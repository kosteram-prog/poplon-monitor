"""Logika: ULDK -> geometria, CDSE -> seria NDVI, ocena poplonu."""

import base64
import math
import os
import threading
import time
from datetime import date, timedelta

import requests
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

# Reużywamy retry/backoff na 429/5xx z sentinel1.py zamiast duplikować logikę.
# Naprawa: fetch_ndvi_series/fetch_true_color_image dotąd robiły gołe
# requests.post bez żadnego retry — w trybie wsadowym (kilka wątków naraz
# walących w Sentinel Hub) pojedynczy 429 od razu wywalał cały wpis batcha
# jako błąd, zamiast spróbować ponownie za chwilę jak reszta modułów.
from sentinel1 import _post_with_retry

# --- Progi i okna (kalibrowalne) -------------------------------------------

CFG = {
    "scierniskowy": {
        "t_start": lambda rok: date(rok, 7, 1),
        "t_end":   lambda rok: date(rok, 11, 15),
        "baseline": lambda rok: (date(rok, 7, 15), date(rok, 9, 5)),
        "cover":    lambda rok: (date(rok, 9, 15), date(rok, 10, 15)),
    },
    "ozimy": {
        "t_start": lambda rok: date(rok, 8, 1),
        # Okno DANYCH/WYŚWIETLANIA — jak daleko pobieramy i pokazujemy na
        # wykresie sceny S2/S1, niezależnie od tego, do kiedy sięga ocena
        # zgodności (patrz "winter" niżej). Naprawa: dawniej t_end i koniec
        # okna oceny były tą samą datą (20.02) — teraz dane wyświetlają się
        # do 10.03, żeby operator widział trend jeszcze kawałek za granicą
        # wymogu, a sam werdykt nadal ocenia tylko do wymaganego terminu.
        "t_end":   lambda rok: date(rok + 1, 3, 10),
        "baseline": lambda rok: (date(rok, 8, 1), date(rok, 9, 20)),
        "cover":    lambda rok: (date(rok, 10, 10), date(rok, 11, 30)),
        # Okno OCENY zgodności — musi kończyć się na realnym wymogu programu
        # LEN (FLD_KSR_004f pkt 3: poplon ozimy musi pozostać do 15.02), nie
        # na dacie granicznej danych. Naprawa: dawniej używano tu osobnej,
        # zahardkodowanej daty 20.02, niespójnej z już istniejącą stałą
        # OZIMY_MAINTENANCE_END (15.02) zdefiniowaną niżej, ale nigdzie
        # nieużywaną — teraz "winter" faktycznie z niej korzysta.
        "winter":   lambda rok: (date(rok + 1, 1, 10), OZIMY_MAINTENANCE_END(rok)),
    },
}

DEFAULT_NDVI_COVER_HIGH = 0.42
DEFAULT_NDVI_COVER_MID  = 0.30
NDVI_COVER_HIGH = DEFAULT_NDVI_COVER_HIGH
NDVI_COVER_MID  = DEFAULT_NDVI_COVER_MID
NDVI_BARE_MAX   = 0.28
MIN_GROWTH      = 0.15
MIN_VALID_FRAC       = 0.35   # minimalna granica — poniżej scena odrzucana całkowicie
MIN_VALID_FRAC_AUTO  = 0.50   # granica automatycznego werdyktu — poniżej wymaga zatwierdzenia operatora
MIN_SCENES_COVER = 2


def reload_thresholds():
    """Wczytuje kalibrowalne progi NDVI z bazy (tabela `settings`), jeśli są
    zapisane; w przeciwnym razie zostają wartości domyślne. Progi są trzymane
    w pamięci procesu (nie modyfikujemy pliku źródłowego core.py w runtime —
    patrz calibration.py), więc ta funkcja jest bezpieczna do wywołania
    wielokrotnie i współbieżnie."""
    global NDVI_COVER_HIGH, NDVI_COVER_MID
    try:
        import db
        high = db.get_setting("ndvi_cover_high")
        mid = db.get_setting("ndvi_cover_mid")
        NDVI_COVER_HIGH = float(high) if high is not None else DEFAULT_NDVI_COVER_HIGH
        NDVI_COVER_MID = float(mid) if mid is not None else DEFAULT_NDVI_COVER_MID
    except Exception:
        # Baza niedostępna (np. wywołane przed init_db()) — zostają domyślne.
        NDVI_COVER_HIGH = DEFAULT_NDVI_COVER_HIGH
        NDVI_COVER_MID = DEFAULT_NDVI_COVER_MID

# --- Kategorie zbiorcze (do trybu wsadowego / wielu działek na raz) --------
CATEGORY_LABELS = {
    "green":  "OK — poplon potwierdzony",
    "yellow": "Do interpretacji agronoma",
    "red":    "Odrzucony — brak poplonu",
}


def categorize_verdict(verdict: dict) -> str:
    """Mapuje werdykt na jedną z 3 kategorii zbiorczych: green/yellow/red."""
    status = verdict.get("status")
    confidence = verdict.get("confidence")
    if status == "BRAK":
        return "red"
    if verdict.get("compliance_issue"):
        return "yellow"
    if verdict.get("requires_operator_approval"):
        # Sceny ze strefy szarej (0.35–0.50 valid_frac) — wymaga ręcznej weryfikacji
        return "yellow"
    if status == "OBECNY" and confidence == "wysoka":
        return "green"
    return "yellow"


# --- Wymogi programu LEN (FLD_KSR_004e — poplon ścierniskowy/letni,
#     FLD_KSR_004f — poplon jesienny/ozimy; sezon 2026-2027). Te terminy
# zmieniają się między sezonami — jeśli działasz w innym sezonie/programie,
# popraw stałe poniżej.
SCIERNISKOWY_MIN_WEEKS = 8  # FLD_KSR_004e pkt 2: min. 8 tyg. od siewu
OZIMY_SOW_DEADLINE_NORMAL = lambda rok: date(rok, 10, 1)      # FLD_KSR_004f pkt 2
OZIMY_SOW_DEADLINE_EXCEPTION = lambda rok: date(rok, 10, 31)  # tylko po konsultacji z doradcą LEN
OZIMY_MAINTENANCE_END = lambda rok: date(rok + 1, 2, 15)      # FLD_KSR_004f pkt 3
OZIMY_MULCH_ALLOWED_FROM = lambda rok: date(rok, 11, 15)      # FLD_KSR_004f pkt 3
# Minimalny okres między siewem a możliwością wiarygodnej oceny okrywy dla
# poplonu ozimego (kiełkowanie + wschody zbóż ozimych trwają zwykle 2-3 tyg.,
# doliczamy margines bezpieczeństwa). Nie jest to termin z FLD_KSR — to
# techniczne założenie analogiczne do SCIERNISKOWY_MIN_WEEKS, żeby okno oceny
# nie zaczynało się przed siewem przy legalnie późnym (do 31.10) terminie.
OZIMY_MIN_WEEKS = 3


# --- Likwidacja poplonu (opcjonalna informacja od użytkownika) -------------
# Jeśli poplon został świadomie zlikwidowany (np. wałowany, talerzowany
# i pozostawiony na polu), spadek NDVI po tej dacie jest oczekiwany i nie
# powinien być odczytywany jako dowód braku poplonu.

TERMINATION_METHOD_LABELS = {
    "walowanie": "wałowanie",
    "talerzowanie": "lekkie talerzowanie",
    "przyoranie": "przyoranie",
    "oprysk": "oprysk herbicydem",
    "inne": "inny sposób",
}


def _clip_window(window, termination_date):
    """Skraca okno (a,b) do daty likwidacji, jeśli ta w nie wchodzi.
    Zwraca (nowe_okno_lub_None, czy_skrocone)."""
    if window is None or termination_date is None:
        return window, False
    a, b = window
    if termination_date <= a:
        return None, True  # cała likwidacja przed oknem — pomiń całe okno
    if termination_date < b:
        return (a, termination_date), True  # skrócone
    return window, False  # likwidacja po oknie — bez zmian

ULDK_URL = "https://uldk.gugik.gov.pl/"
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                  "protocol/openid-connect/token")
CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

TRUE_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02", "dataMask"],
    output: { bands: 4 }
  };
}
function evaluatePixel(s) {
  // Listopadowe sceny mają niską odbijalność — pierwiastek sześcienny
  // podnosi cienie, żeby zdjęcie nie wychodziło prawie czarne.
  function vis(v) { return Math.cbrt(Math.max(0, v) * 0.55); }
  return [vis(s.B04), vis(s.B03), vis(s.B02), s.dataMask];
}
"""

# --- Zdjęcie satelitarne z dnia szczytu NDVI --------------------------------
IMAGE_PADDING_RATIO = 0.15   # margines wokół działki, żeby było widać kontekst
IMAGE_TARGET_RES_M = 10.0    # docelowa rozdzielczość (natywna dla Sentinel-2 10m)
IMAGE_MAX_DIM = 640
# Optymalizacja PU (listopad 2026): dawniej IMAGE_MIN_DIM = 512, czyli obraz
# ~50x40 pikseli natywnych 10 m był po stronie serwera powiększany do 512 px
# — Sentinel Hub liczy PU od liczby pikseli WYJŚCIA, więc płaciliśmy za
# piksele bez żadnej nowej informacji (zmierzone: 0,74 PU vs 0,03 PU za ten
# sam obraz w natywnej rozdzielczości). Teraz pobieramy obraz w 10 m, a
# powiększa go przeglądarka (Leaflet imageOverlay, CSS pixelated — ten sam
# wygląd co dawne powiększanie NEAREST po stronie serwera).
IMAGE_MIN_DIM = 32


def _compute_padded_bbox_4326(geom_4326, padding_ratio=IMAGE_PADDING_RATIO):
    minx, miny, maxx, maxy = geom_4326.bounds
    w, h = maxx - minx, maxy - miny
    pad = padding_ratio * max(w, h)
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def _compute_image_size(width_m, height_m, target_res_m=IMAGE_TARGET_RES_M,
                         max_dim=IMAGE_MAX_DIM, min_dim=IMAGE_MIN_DIM):
    w_px, h_px = width_m / target_res_m, height_m / target_res_m
    largest = max(w_px, h_px)
    scale = 1.0
    if largest > max_dim:
        scale = max_dim / largest
    elif largest < min_dim:
        scale = min_dim / largest
    return (max(int(round(w_px * scale)), 32),
            max(int(round(h_px * scale)), 32))


def fetch_true_color_image(geom_4326, target_date: date):
    """Pobiera prawdziwokolorowe zdjęcie Sentinel-2 dla działki (z marginesem)
    z konkretnego dnia. Zwraca dict {date, bbox (WGS84), png_base64} albo
    None, jeśli się nie uda (błąd nie przerywa całej oceny działki)."""
    bbox_4326 = _compute_padded_bbox_4326(geom_4326)
    to_3857 = Transformer.from_crs(4326, 3857, always_xy=True).transform
    minx3857, miny3857 = to_3857(bbox_4326[0], bbox_4326[1])
    maxx3857, maxy3857 = to_3857(bbox_4326[2], bbox_4326[3])
    # Obraz zostaje w EPSG:3857 (natywna projekcja Leafleta — nakładka trafia
    # idealnie w mapę), ale rozmiar liczymy w METRACH TERENOWYCH: jednostka
    # 3857 na tej szerokości to ok. cos(lat) metra w terenie.
    ground_scale = math.cos(math.radians((bbox_4326[1] + bbox_4326[3]) / 2.0))
    width_px, height_px = _compute_image_size((maxx3857 - minx3857) * ground_scale,
                                              (maxy3857 - miny3857) * ground_scale)

    payload = {
        "input": {
            "bounds": {
                "bbox": [minx3857, miny3857, maxx3857, maxy3857],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{target_date.isoformat()}T00:00:00Z",
                        "to": f"{target_date.isoformat()}T23:59:59Z",
                    },
                    "mosaickingOrder": "mostRecent",
                    "maxCloudCoverage": 100,
                },
            }],
        },
        "output": {
            "width": width_px, "height": height_px,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": TRUE_COLOR_EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {cdse_token()}"}
    try:
        r = _post_with_retry(CDSE_PROCESS_URL, payload, headers, timeout=60,
                              refresh_token_fn=cdse_token)
        if r.status_code != 200 or not r.content:
            return None
        return {
            "date": target_date.isoformat(),
            "bbox": list(bbox_4326),
            "png_base64": base64.b64encode(r.content).decode("ascii"),
        }
    except (requests.RequestException, RuntimeError):
        return None

# Referencja maski NDVI (te same klasy SCL co EVALSCRIPT_MULTI w analytics.py).
# Od listopada 2026 nie jest już wysyłany do Sentinel Hub — NDVI pochodzi
# z wyniku zapytania wielowskaźnikowego (patrz fetch_ndvi_series).
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  const bad = [0, 1, 3, 8, 9, 10, 11]; // no-data, saturacja, cienie, chmury, snieg
  let valid = s.dataMask;
  if (bad.indexOf(s.SCL) !== -1) valid = 0;
  const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-9);
  return { ndvi: [ndvi], dataMask: [valid] };
}
"""


def safe_float(x):
    """Konwertuje wartość na float; zwraca None jeśli None, niekonwertowalna lub nieskończona."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class AppError(Exception):
    """Błąd z komunikatem do pokazania użytkownikowi."""


# --- ULDK -------------------------------------------------------------------

# Krótkotrwały cache geometrii działki (ULDK). W jednym przebiegu wsadowym
# check_parcel() i check_multi() dla tej samej działki dotąd osobno odpytywały
# ULDK o tę samą, niezmienną w tym oknie czasowym geometrię — zbędny
# roundtrip, który przy dużych batchach (np. 20 działek) realnie dokłada się
# do czasu przetwarzania. TTL krótki (10 min), żeby nie trzymać nieaktualnych
# danych, gdyby ktoś poprawił granice działki w międzyczasie.
_geom_cache: dict = {}
_geom_cache_lock = threading.Lock()
_GEOM_CACHE_TTL_SECONDS = 600
# Blokady per działka: w batchu NDVI i analiza wielowskaźnikowa startują
# równolegle i obie trafiały w pusty cache — szły 2 identyczne zapytania do
# ULDK. Teraz drugi wątek czeka na wynik pierwszego.
_geom_fetch_locks: dict = {}


def fetch_parcel_geometry(parcel_id: str):
    key = parcel_id.strip()
    now = time.time()
    with _geom_cache_lock:
        cached = _geom_cache.get(key)
        if cached and now - cached[2] < _GEOM_CACHE_TTL_SECONDS:
            return cached[0], cached[1]
        fetch_lock = _geom_fetch_locks.setdefault(key, threading.Lock())

    with fetch_lock:
        with _geom_cache_lock:
            cached = _geom_cache.get(key)
            if cached and time.time() - cached[2] < _GEOM_CACHE_TTL_SECONDS:
                return cached[0], cached[1]
        return _fetch_parcel_geometry_uldk(parcel_id, key)


def _fetch_parcel_geometry_uldk(parcel_id: str, key: str):
    now = time.time()
    params = {"request": "GetParcelById", "id": key,
              "result": "geom_wkt,teryt,voivodeship,county,commune",
              "srid": "4326"}
    try:
        r = requests.get(ULDK_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AppError(f"Brak połączenia z ULDK (GUGiK): {e}")
    lines = r.text.strip().split("\n")
    if lines[0].strip() != "0" or len(lines) < 2:
        raise AppError(f"Nie znaleziono działki o ID „{parcel_id}”. "
                       "Sprawdź identyfikator TERYT.")
    fields = lines[1].split("|")
    wkt_str = fields[0]
    if wkt_str.upper().startswith("SRID="):
        wkt_str = wkt_str.split(";", 1)[1]
    geom = shapely_wkt.loads(wkt_str)
    if geom.is_empty:
        raise AppError("ULDK zwrócił pustą geometrię działki.")
    meta = {}
    if len(fields) >= 5:
        meta = {"teryt": fields[1], "wojewodztwo": fields[2],
                "powiat": fields[3], "gmina": fields[4]}

    with _geom_cache_lock:
        _geom_cache[key] = (geom, meta, now)
    return geom, meta


def fetch_parcel_at_point(lon: float, lat: float):
    """Działka ewidencyjna zawierająca punkt (ULDK GetParcelByXY, tylko Polska).
    Zwraca (geom_4326, meta) z meta["parcel_id"] = identyfikator TERYT działki."""
    params = {"request": "GetParcelByXY", "xy": f"{lon},{lat},4326",
              "result": "geom_wkt,teryt,voivodeship,county,commune", "srid": "4326"}
    try:
        r = requests.get(ULDK_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AppError(f"Brak połączenia z ULDK (GUGiK): {e}")
    lines = r.text.strip().split("\n")
    if lines[0].strip() != "0" or len(lines) < 2:
        raise AppError("W tym miejscu ULDK nie zwrócił działki ewidencyjnej "
                       "(punkt poza Polską albo poza działką).")
    fields = lines[1].split("|")
    wkt_str = fields[0]
    if wkt_str.upper().startswith("SRID="):
        wkt_str = wkt_str.split(";", 1)[1]
    geom = shapely_wkt.loads(wkt_str)
    if geom.is_empty:
        raise AppError("ULDK zwrócił pustą geometrię działki.")
    meta = {}
    if len(fields) >= 5:
        meta = {"parcel_id": fields[1], "teryt": fields[1], "wojewodztwo": fields[2],
                "powiat": fields[3], "gmina": fields[4]}
    return geom, meta


def inner_buffer(geom_4326, meters: float = 10.0):
    to_2180 = Transformer.from_crs(4326, 2180, always_xy=True).transform
    to_4326 = Transformer.from_crs(2180, 4326, always_xy=True).transform
    g_m = shp_transform(to_2180, geom_4326)
    g_buf = g_m.buffer(-meters)
    buffered = True
    if g_buf.is_empty or g_buf.area < 400:
        g_buf, buffered = g_m, False
    return shp_transform(to_4326, g_buf), g_m.area / 10000.0, buffered


# --- CDSE / Sentinel Hub -----------------------------------------------------

_token_cache = {"token": None, "exp": 0.0}
_token_lock = threading.Lock()


def cdse_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["exp"] - 60:
        return _token_cache["token"]
    # Przy starcie batcha kilka wątków naraz widziało pusty cache i każdy
    # pobierał własny token — teraz pobiera jeden, reszta bierze z cache.
    with _token_lock:
        if _token_cache["token"] and time.time() < _token_cache["exp"] - 60:
            return _token_cache["token"]
        return _cdse_token_fetch()


def _cdse_token_fetch() -> str:
    cid = (os.environ.get("CDSE_CLIENT_ID") or "").strip().strip('"').strip("'")
    sec = (os.environ.get("CDSE_CLIENT_SECRET") or "").strip().strip('"').strip("'")
    if not cid or not sec:
        raise AppError("Serwer nie ma skonfigurowanych CDSE_CLIENT_ID / "
                       "CDSE_CLIENT_SECRET (dane z panelu Copernicus Data "
                       "Space → Sentinel Hub → OAuth clients).")
    try:
        session = requests.Session()
        session.trust_env = False
        r = session.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": cid,
                "client_secret": sec,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                "Accept": "application/json",
            },
            timeout=30,
        )
        if r.status_code >= 400:
            body = (r.text or "").strip()[:400]
            raise AppError(
                "Autoryzacja w Copernicus Data Space nie powiodła się: "
                f"HTTP {r.status_code} (client_id={cid[:12]}…, "
                f"secret_len={len(sec)}). Odpowiedź: {body or r.reason}")
        r.raise_for_status()
    except AppError:
        raise
    except requests.RequestException as e:
        raise AppError(f"Autoryzacja w Copernicus Data Space nie powiodła "
                       f"się: {e}")
    tok = r.json()
    _token_cache["token"] = tok["access_token"]
    _token_cache["exp"] = time.time() + tok.get("expires_in", 600)
    return _token_cache["token"]


def fetch_ndvi_series(geom_4326, t_start: date, t_end: date):
    """Szereg NDVI działki (date, ndvi_mean, valid_frac).

    Optymalizacja (listopad 2026): NDVI nie jest już pobierany osobnym
    zapytaniem do Sentinel Hub. Liczymy go z wyniku analytics.
    fetch_multi_series_strict — ta sama maska SCL (klasy 0,1,3,8,9,10,11),
    ten sam limit zachmurzenia (90%), ta sama geometria, a NDVI jest jednym
    z jego wyjść (zweryfikowane na żywo: identyczne sceny i valid_frac,
    różnice NDVI tylko z zaokrąglenia, <= 0,001). Dzięki wspólnej pamięci
    podręcznej (sh_utils.cached_call) w batchu, przy pojedynczej działce
    i w schedulerze NDVI i analiza wielowskaźnikowa dzielą JEDNO zapytanie.

    Daty z przyszłości nie mają scen, więc koniec zakresu przycinamy do
    dziś — dzięki temu klucz pamięci podręcznej jest ten sam co w
    /api/check_multi (które zawsze przycinało do dziś).
    Celowo bez łagodniejszego fallbacku bez maski SCL (jak dawniej).
    """
    import analytics
    t_end = min(t_end, date.today())
    if t_end < t_start:
        return []
    try:
        multi = analytics.fetch_multi_series_strict(geom_4326, t_start, t_end)
    except AppError:
        raise
    except (requests.RequestException, RuntimeError) as e:
        raise AppError(f"Sentinel Hub: {e}")
    records = []
    for r in multi:
        ndvi = safe_float(r.get("ndvi"))
        vf = safe_float(r.get("valid_frac"))
        if ndvi is None or vf is None:
            continue
        records.append({"date": r["date"], "ndvi_mean": round(ndvi, 3),
                        "valid_frac": round(vf, 3)})
    return sorted(records, key=lambda x: x["date"])


# --- Werdykt -----------------------------------------------------------------

def _in_window(recs, win):
    """Zwraca sceny w oknie z valid_frac >= MIN_VALID_FRAC (wszystkie używane)."""
    if win is None:
        return []
    a, b = win
    return [r for r in recs
            if a <= r["date"] <= b and r["valid_frac"] >= MIN_VALID_FRAC]


def _low_quality_in_window(recs, win):
    """Zwraca sceny ze strefy szarej (MIN_VALID_FRAC <= vf < MIN_VALID_FRAC_AUTO)."""
    if win is None:
        return []
    a, b = win
    return [r for r in recs
            if a <= r["date"] <= b
            and MIN_VALID_FRAC <= r["valid_frac"] < MIN_VALID_FRAC_AUTO]


def assess(records, typ: str, rok: int, sowing_date=None,
           termination_date=None, termination_method=None) -> dict:
    cfg = CFG[typ]
    baseline_window = cfg["baseline"](rok)
    cover_window = cfg["cover"](rok)
    winter_window = cfg["winter"](rok) if typ == "ozimy" else None

    notes = []
    compliance_issue = False
    term_label = None
    if termination_date is not None:
        term_label = (f"{termination_date.isoformat()}"
                      + (f" ({TERMINATION_METHOD_LABELS.get(termination_method, termination_method)})"
                         if termination_method else ""))

    # --- Sprawdzenia zgodności z terminami programu LEN --------------------
    if typ == "scierniskowy" and sowing_date is not None and termination_date is not None:
        min_end = sowing_date + timedelta(weeks=SCIERNISKOWY_MIN_WEEKS)
        if termination_date < min_end:
            compliance_issue = True
            notes.append(
                f"⚠ Zgodność: likwidacja ({term_label}) nastąpiła przed wymaganym "
                f"minimalnym okresem {SCIERNISKOWY_MIN_WEEKS} tygodni od siewu "
                f"({sowing_date.isoformat()} + {SCIERNISKOWY_MIN_WEEKS} tyg. = "
                f"{min_end.isoformat()}), zgodnie z FLD_KSR_004e pkt 2. Skrócenie "
                "dopuszczalne wyłącznie po konsultacji z doradcą LEN — wymaga potwierdzenia.")

    # --- Korekta okna oceny o datę siewu -----------------------------------
    # Ścierniskowy: okno oceny nie może zaczynać się wcześniej niż sowing_date + 8 tyg.
    # Bo poplon nie może mieć wymaganej wegetacji przed tym terminem.
    if typ == "scierniskowy" and sowing_date is not None:
        min_cover_start = sowing_date + timedelta(weeks=SCIERNISKOWY_MIN_WEEKS)
        cover_a, cover_b = cover_window
        if min_cover_start > cover_a:
            if min_cover_start >= cover_b:
                # siew tak późny że 8 tygodni wypada poza oknem oceny
                notes.append(
                    f"⚠ Data siewu ({sowing_date.isoformat()}) jest późna — 8 tygodni "
                    f"wymaganej wegetacji mija {min_cover_start.isoformat()}, "
                    f"co wykracza poza standardowe okno oceny ({cover_b.isoformat()}). "
                    "Ocena opiera się na dostępnych danych z całego okna.")
            else:
                cover_window = (min_cover_start, cover_b)
                notes.append(
                    f"Okno oceny skrócone od strony początku do {min_cover_start.isoformat()} "
                    f"(siew {sowing_date.isoformat()} + 8 tyg.) — sceny przed tym terminem "
                    "nie są uwzględniane, bo poplon nie mógł jeszcze osiągnąć "
                    "wymaganego poziomu wegetacji.")

    # Ozimy: okno oceny nie może zaczynać się przed sowing_date + OZIMY_MIN_WEEKS.
    # Bez tej korekty spóźniony, ale w pełni dopuszczalny siew (do 31.10) mógł
    # być oceniany w oknie zaczynającym się przed wschodami — dawało to
    # fałszywy werdykt "BRAK" u zgodnego rolnika (analogiczna korekta jak
    # wyżej dla poplonu ścierniskowego).
    if typ == "ozimy" and sowing_date is not None:
        min_cover_start = sowing_date + timedelta(weeks=OZIMY_MIN_WEEKS)
        cover_a, cover_b = cover_window
        if min_cover_start > cover_a:
            if min_cover_start >= cover_b:
                notes.append(
                    f"⚠ Data siewu ({sowing_date.isoformat()}) jest późna — "
                    f"{OZIMY_MIN_WEEKS} tyg. wymagane do wschodów mija "
                    f"{min_cover_start.isoformat()}, co wykracza poza standardowe "
                    f"okno oceny ({cover_b.isoformat()}). Ocena opiera się na "
                    "dostępnych danych z całego okna (w tym okna zimowego, jeśli dotyczy).")
            else:
                cover_window = (min_cover_start, cover_b)
                notes.append(
                    f"Okno oceny skrócone od strony początku do {min_cover_start.isoformat()} "
                    f"(siew {sowing_date.isoformat()} + {OZIMY_MIN_WEEKS} tyg.) — sceny przed "
                    "tym terminem nie są uwzględniane, bo poplon nie mógł jeszcze wzejść.")

    if typ == "ozimy" and sowing_date is not None:
        deadline_normal = OZIMY_SOW_DEADLINE_NORMAL(rok)
        deadline_exception = OZIMY_SOW_DEADLINE_EXCEPTION(rok)
        if sowing_date > deadline_exception:
            compliance_issue = True
            notes.append(
                f"⚠ Zgodność: data siewu ({sowing_date.isoformat()}) wykracza poza "
                f"dopuszczalny termin ({deadline_exception.isoformat()}), zgodnie "
                "z FLD_KSR_004f pkt 2.")
        elif sowing_date > deadline_normal:
            compliance_issue = True
            notes.append(
                f"⚠ Zgodność: data siewu ({sowing_date.isoformat()}) po "
                f"{deadline_normal.isoformat()} — zgodnie z FLD_KSR_004f dopuszczalne "
                "tylko po wcześniejszej konsultacji z doradcą LEN. Wymaga potwierdzenia.")

    if typ == "ozimy" and termination_method == "walowanie" and termination_date is not None:
        mulch_from = OZIMY_MULCH_ALLOWED_FROM(rok)
        if termination_date < mulch_from:
            compliance_issue = True
            notes.append(
                f"⚠ Zgodność: wałowanie ({termination_date.isoformat()}) przed "
                f"{mulch_from.isoformat()} — niezgodne z FLD_KSR_004f pkt 3 "
                "(mulczowanie dopuszczalne dopiero po 15 listopada).")
        notes.append(
            "Fizyczna obecność ściółki po wałowaniu do 15 lutego nie jest "
            "weryfikowalna przez NDVI (martwa okrywa ma niską zieloność) — "
            "zgodnie z metodą weryfikacji w FLD_KSR_004f sprawdza się to na "
            "podstawie zdjęć i dziennika gospodarstwa, nie satelity.")

    if typ == "ozimy" and termination_method == "oprysk":
        notes.append(
            "FLD_KSR_004f zaleca likwidację metodami mechanicznymi, bez "
            "glifosatu ani innych herbicydów — warto potwierdzić zasadność "
            "zastosowania oprysku.")

    if termination_date is not None:
        cover_window, cover_clipped = _clip_window(cover_window, termination_date)
        if cover_window is None:
            return {
                "status": "NIEROZSTRZYGNIETE", "confidence": None,
                "details": [
                    f"Zgłoszona likwidacja poplonu ({term_label}) nastąpiła przed "
                    f"rozpoczęciem standardowego okna oceny "
                    f"({cfg['cover'](rok)[0].isoformat()}) — nie da się ocenić "
                    "rozwoju poplonu w tym okresie."] + notes,
                "metrics": {},
                "termination": {"date": termination_date.isoformat(),
                                "method": termination_method},
                "compliance_issue": True,
            }
        if cover_clipped:
            notes.append(f"Okno oceny okrywy skrócone do daty likwidacji "
                        f"poplonu ({term_label}) — spadek NDVI po tej dacie "
                        "jest oczekiwany i nie obniża oceny.")

        if winter_window is not None:
            winter_window, winter_clipped = _clip_window(winter_window, termination_date)
            if winter_window is None:
                notes.append(f"Sprawdzanie okrywy zimowej pominięte — poplon "
                            f"zlikwidowany ({term_label}) przed tym okresem, "
                            "zgodnie z podaną datą.")
            elif winter_clipped:
                notes.append(f"Okno oceny zimowej skrócone do daty likwidacji "
                            f"({term_label}).")

    base = _in_window(records, baseline_window)
    cover = _in_window(records, cover_window)

    # wykryj sceny ze strefy szarej (0.35-0.50 valid_frac) w oknie oceny
    low_quality_cover = _low_quality_in_window(records, cover_window)
    requires_operator_approval = False

    if len(cover) < MIN_SCENES_COVER:
        return {"status": "NIEROZSTRZYGNIETE", "confidence": None,
                "details": [f"Za mało bezchmurnych scen w oknie oceny "
                            f"({len(cover)}, wymagane {MIN_SCENES_COVER}).",
                            "Zachmurzenie uniemożliwia wiarygodną ocenę "
                            "z Sentinel-2 w tym okresie."] + notes,
                "detail_codes": [("too_few_scenes", {"n": len(cover), "required": MIN_SCENES_COVER}),
                                  ("cloud_obstruction", {})],
                "metrics": {}, "compliance_issue": compliance_issue,
                "requires_operator_approval": False,
                "low_quality_scenes": len(low_quality_cover)}

    baseline_ndvi = safe_float(min((r["ndvi_mean"] for r in base), default=None))
    cover_ndvi = safe_float(max(r["ndvi_mean"] for r in cover))
    peak_date = None
    if cover_ndvi is not None:
        peak_candidates = [r["date"] for r in cover if r["ndvi_mean"] == cover_ndvi]
        if peak_candidates:
            peak_date = min(peak_candidates)
    growth = safe_float((cover_ndvi - baseline_ndvi) if (cover_ndvi is not None and baseline_ndvi is not None) else None)
    metrics = {"ndvi_baza": baseline_ndvi, "ndvi_szczyt": cover_ndvi,
               "przyrost": round(growth, 3) if growth is not None else None,
               "sceny_w_oknie": len(cover)}
    details = []
    detail_codes = []

    if cover_ndvi >= NDVI_COVER_HIGH and (growth is None or growth >= MIN_GROWTH):
        res = {"status": "OBECNY", "confidence": "wysoka"}
        details.append("Wyraźna okrywa roślinna w oknie oceny — przebieg "
                       "NDVI typowy dla wysianego poplonu.")
        detail_codes.append(("cover_clear", {"ndvi": round(cover_ndvi, 2)}))
    elif cover_ndvi >= NDVI_COVER_MID:
        res = {"status": "PRAWDOPODOBNY", "confidence": "średnia"}
        details.append("Wegetacja obecna, ale NDVI w strefie pośredniej — "
                       "możliwe słabe wschody, samosiewy lub zachwaszczenie.")
        detail_codes.append(("cover_mid", {"ndvi": round(cover_ndvi, 2)}))
    elif cover_ndvi <= NDVI_BARE_MAX:
        res = {"status": "BRAK", "confidence": "wysoka"}
        details.append("NDVI na poziomie gołej gleby / ścierniska przez "
                       "całe okno oceny.")
        detail_codes.append(("bare_soil", {"ndvi": round(cover_ndvi, 2)}))
    else:
        res = {"status": "BRAK", "confidence": "średnia"}
        details.append("Wegetacja szczątkowa, poniżej progu rozwiniętego "
                       "poplonu.")
        detail_codes.append(("sparse_veg", {"ndvi": round(cover_ndvi, 2)}))

    if typ == "ozimy" and res["status"] != "BRAK":
        if winter_window is None and termination_date is not None:
            pass
        else:
            winter = _in_window(records, winter_window) if winter_window else []
            if not winter:
                res["confidence"] = "obniżona"
                details.append("Brak użytecznych scen w oknie zimowym "
                               "(chmury/śnieg) — utrzymanie okrywy do połowy "
                               "lutego niepotwierdzone.")
                detail_codes.append(("no_winter_scenes", {}))
            else:
                w = safe_float(max(r["ndvi_mean"] for r in winter))
                metrics["ndvi_zima"] = w
                if w >= 0.30:
                    details.append("Okrywa zimowa potwierdzona.")
                    detail_codes.append(("winter_ok", {"ndvi": round(w, 2)}))
                elif w <= 0.18:
                    res = {"status": "BRAK", "confidence": "średnia"}
                    details.append("Jesienią wegetacja była obecna, ale zimą "
                                   "NDVI spadł do poziomu gleby — możliwa "
                                   "przedwczesna likwidacja poplonu.")
                    detail_codes.append(("winter_drop", {"ndvi": round(w, 2)}))
                else:
                    res = {"status": "PRAWDOPODOBNY", "confidence": "niska"}
                    details.append("Okrywa zimowa niejednoznaczna — przymrozki "
                                   "i śnieg mogą zaniżać NDVI.")
                    detail_codes.append(("winter_ambiguous", {"ndvi": round(w, 2)}))

    # --- Wszystkie sceny słabej jakości → BRAK DANYCH, nie żółta -----------
    # Jeśli każda scena w oknie oceny pochodzi ze strefy szarej (valid_frac 0.35–0.50)
    # i nie ma żadnej wiarygodnej sceny, wynik jest zbyt niepewny do oceny.
    good_cover = [r for r in cover if r["valid_frac"] >= MIN_VALID_FRAC_AUTO]
    if len(good_cover) == 0 and len(low_quality_cover) > 0:
        return {"status": "NIEROZSTRZYGNIETE", "confidence": None,
                "details": [f"Wszystkie {len(cover)} scen w oknie oceny pochodzi ze strefy "
                            f"obniżonej jakości (valid_frac 0.35–0.50, czyli 50–65% chmur). "
                            "Brak wiarygodnych danych do oceny poplonu — wynik niemożliwy do "
                            "wyznaczenia. Sprawdź ponownie po poprawie warunków atmosferycznych."]
                            + notes,
                "detail_codes": [("no_data_all_low_quality",
                                  {"n": len(cover), "min_vf": MIN_VALID_FRAC,
                                   "auto_vf": MIN_VALID_FRAC_AUTO}),
                                  ("cloud_obstruction", {})],
                "metrics": {}, "compliance_issue": compliance_issue,
                "requires_operator_approval": False,
                "low_quality_scenes": len(low_quality_cover)}

    # --- Mądrzejsza logika zatwierdzenia operatora -------------------------
    # Zatwierdzenie wymagane TYLKO gdy wynik niejednoznaczny LUB >50% scen słabej jakości.
    # Gdy OBECNY z wysoką pewnością i dominują dobre sceny — bez zatwierdzenia.
    if low_quality_cover:
        n_all = len(cover)
        n_low = len(low_quality_cover)
        low_fraction = n_low / n_all if n_all > 0 else 0
        is_ambiguous = res["status"] in ("PRAWDOPODOBNY", "NIEROZSTRZYGNIETE")
        majority_low = low_fraction > 0.5
        if is_ambiguous or majority_low:
            requires_operator_approval = True
            details.append(
                f"⚠ Dane częściowe: {n_low}/{n_all} scen obniżonej jakości "
                f"(valid_frac 0.35–0.50). "
                + ("Wynik niejednoznaczny — " if is_ambiguous else "")
                + "Werdykt wymaga zatwierdzenia operatora.")
            detail_codes.append(("low_quality_data",
                                  {"n_low": n_low, "n_all": n_all, "ambiguous": is_ambiguous}))
        else:
            details.append(
                f"ℹ {n_low}/{n_all} scen obniżonej jakości, "
                "ale dominują sceny wiarygodne — bez konieczności zatwierdzenia.")
            detail_codes.append(("low_quality_minor", {"n_low": n_low, "n_all": n_all}))

    # informacyjna metryka: ile dni od siewu do szczytu NDVI
    if sowing_date is not None and peak_date is not None and peak_date >= sowing_date:
        metrics["dni_od_siewu_do_szczytu"] = (peak_date - sowing_date).days

    details.extend(notes)
    detail_codes.extend([("compliance_note", {"text": n}) for n in notes])
    result = {**res, "details": details, "detail_codes": detail_codes,
              "metrics": metrics,
              "compliance_issue": compliance_issue,
              "requires_operator_approval": requires_operator_approval,
              "low_quality_scenes": len(low_quality_cover)}
    if peak_date is not None:
        result["peak_date"] = peak_date.isoformat()
    if termination_date is not None:
        result["termination"] = {"date": termination_date.isoformat(),
                                  "method": termination_method}
    if sowing_date is not None:
        result["sowing_date"] = sowing_date.isoformat()
    return result


# --- Pełny przebieg ----------------------------------------------------------

def check_parcel(parcel_id: str, typ: str, rok: int,
                  sowing_date_str: str = None,
                  termination_date_str: str = None,
                  termination_method: str = None,
                  fetch_image: bool = True) -> dict:
    if typ not in CFG:
        raise AppError("Nieznany typ poplonu.")

    sowing_date = None
    if sowing_date_str:
        try:
            sowing_date = date.fromisoformat(sowing_date_str)
        except ValueError:
            raise AppError(f"Nieprawidłowy format daty siewu: "
                           f"„{sowing_date_str}” (oczekiwano RRRR-MM-DD).")

    termination_date = None
    if termination_date_str:
        try:
            termination_date = date.fromisoformat(termination_date_str)
        except ValueError:
            raise AppError(f"Nieprawidłowy format daty likwidacji: "
                           f"„{termination_date_str}” (oczekiwano RRRR-MM-DD).")

    geom, meta = fetch_parcel_geometry(parcel_id)
    geom_buf, area_ha, buffered = inner_buffer(geom, 10.0)
    cfg = CFG[typ]
    t0, t1 = cfg["t_start"](rok), cfg["t_end"](rok)
    records = fetch_ndvi_series(geom_buf, t0, t1)
    verdict = assess(records, typ, rok, sowing_date, termination_date, termination_method)

    windows = {"baseline": cfg["baseline"](rok), "cover": cfg["cover"](rok)}
    if typ == "ozimy":
        windows["winter"] = cfg["winter"](rok)

    warnings = []
    if area_ha < 0.3:
        warnings.append("Mała działka (< 0,3 ha) — statystyka NDVI z pikseli "
                        "10 m może być niepewna.")
    if not buffered:
        warnings.append("Działka zbyt mała na bufor wewnętrzny — do "
                        "statystyki weszły piksele brzegowe.")

    satellite_image = None
    peak_date_iso = verdict.get("peak_date")
    # fetch_image=False (tryb wsadowy): zdjęcie prawdziwokolorowe to 4. z rzędu
    # sekwencyjny call do Sentinel Hub dla tej działki, a w widoku listy
    # wsadowej i tak nie jest wyświetlane (patrz static/index.html
    # showSatelliteImage — używane tylko po wejściu w szczegóły pojedynczej
    # działki). Tam dociągamy je na żądanie przez /api/satellite_image,
    # korzystając z tego samego peak_date i zcache'owanej geometrii.
    if fetch_image and peak_date_iso:
        try:
            satellite_image = fetch_true_color_image(geom, date.fromisoformat(peak_date_iso))
        except Exception:
            satellite_image = None
        if satellite_image is None:
            warnings.append("Nie udało się pobrać zdjęcia satelitarnego z dnia "
                            f"szczytu NDVI ({peak_date_iso}) — pokazany zostanie "
                            "tylko standardowy podkład mapy.")

    return {
        "parcel": {"id": parcel_id, "area_ha": round(safe_float(area_ha) or 0.0, 2),
                   "geojson": mapping(geom), **meta},
        "typ": typ, "rok": rok,
        "range": [t0.isoformat(), t1.isoformat()],
        "windows": {k: [a.isoformat(), b.isoformat()]
                    for k, (a, b) in windows.items()},
        "thresholds": {"cover_high": NDVI_COVER_HIGH,
                       "cover_mid": NDVI_COVER_MID,
                       "bare_max": NDVI_BARE_MAX,
                       "min_valid_frac": MIN_VALID_FRAC},
        "series": [{**r, "date": r["date"].isoformat()} for r in records],
        "verdict": verdict,
        "satellite_image": satellite_image,
        "warnings": warnings,
        "category": categorize_verdict(verdict),
        "category_label": CATEGORY_LABELS[categorize_verdict(verdict)],
    }
