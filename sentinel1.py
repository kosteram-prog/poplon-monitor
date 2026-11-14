"""
sentinel1.py

Moduł fuzji danych radarowych Sentinel-1 (SAR) z istniejącą logiką NDVI.

Dostarcza:
  - fetch_s1_series()     — pobieranie VV/VH z Sentinel Hub Statistical API
  - compute_sar_indices() — obliczanie Cross-Ratio i RVI
  - detect_tillage()      — detekcja orki/likwidacji poplonu w szeregu czasowym
  - assess_with_sar()     — wzbogacenie istniejącego werdyktu o dane SAR

Integruje się z core.py bez modyfikacji istniejącego kodu.
"""

import collections
import logging
import math
import os
import random
import threading
import time
from datetime import date, datetime
from typing import Optional

import requests

# Timeouty i retry na chwilowe błędy Sentinel Hub (429 rate-limit, 5xx).
# Bez tego pojedynczy rate-limit rutynowo degradował cały batch do
# NDVI-only, mimo że ponowna próba za chwilę zwykle się udaje.
#
# Optymalizacja (listopad 2026): limit konta CDSE to 300 PU/min i 300
# zapytań/min. Dawniej każdy wątek po 429 czekał niezależnie 2/4/8 s
# i po 3 próbach się poddawał — kolejne wątki dalej waliły w serwer, więc
# w logu były serie 429, a działki spadały do analizy „tylko NDVI”.
# Teraz:
#   * na 429 czytamy nagłówek Retry-After (Sentinel Hub podaje go w ms)
#     i wstrzymujemy WSZYSTKIE wątki naraz (globalna pauza),
#   * przed każdym zapytaniem pilnujemy budżetu PU i liczby zapytań
#     w ostatnich 60 s (PU z nagłówka x-processingunits-spent), żeby w ogóle
#     nie dobijać do limitu,
#   * na 429 próbujemy do _MAX_RETRIES razy (łącznie do ok. 1,5 min).
_RETRYABLE_5XX = {500, 502, 503, 504}
_RETRYABLE_STATUS = {429} | _RETRYABLE_5XX
_MAX_RETRIES = 6            # dla 429
_MAX_RETRIES_5XX = 3        # dla 5xx i błędów połączenia
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 30.0

# Budżet na minutę z zapasem względem limitu konta (300/300). Konfigurowalny
# przez .env, gdyby konto miało inny plan.
_PU_BUDGET_PER_MIN = float(os.environ.get("SENTINEL_HUB_PU_PER_MIN", "250"))
_REQ_BUDGET_PER_MIN = int(os.environ.get("SENTINEL_HUB_REQ_PER_MIN", "250"))
_PU_ESTIMATE_DEFAULT = 1.0   # szacunek PU zapytania "w locie", zanim przyjdzie odpowiedź

# Globalny limit RÓWNOCZESNYCH requestów do Sentinel Hub, wspólny dla
# core.py/analytics.py/sentinel1.py (wszystkie idą przez _post_with_retry).
# Semafor ogranicza liczbę jednocześnie "w locie" requestów, niezależnie od
# tego, ile wątków próbuje jednocześnie wystartować. Domyślnie 6 (dawniej 4)
# — przy ~2 PU na działkę i budżecie PU/min powyżej to bezpieczne.
_SH_MAX_CONCURRENT = int(os.environ.get("SENTINEL_HUB_MAX_CONCURRENT", "6"))
_sh_gate = threading.Semaphore(_SH_MAX_CONCURRENT)

log = logging.getLogger("sentinel_hub")

_rl_lock = threading.Lock()
_rl_state = {"pause_until": 0.0}
_rl_window: collections.deque = collections.deque()   # [t_monotonic, pu]
_pu_stats = {"pu": 0.0, "requests": 0, "rate_limited": 0}


def _reset_rate_limiter():
    """Zeruje stan limitera i liczniki (testy, restart)."""
    with _rl_lock:
        _rl_state["pause_until"] = 0.0
        _rl_window.clear()
        _pu_stats.update({"pu": 0.0, "requests": 0, "rate_limited": 0})


def pu_stats() -> dict:
    """Kopia liczników: zużyte PU, liczba zapytań, liczba odpowiedzi 429."""
    with _rl_lock:
        return dict(_pu_stats)


def _header_float(r, name):
    try:
        raw = r.headers.get(name)
    except Exception:
        return None
    if not isinstance(raw, (str, bytes, int, float)):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _reserve_slot():
    """Wylicza (raz, bez pętli) ile trzeba poczekać — globalna pauza po 429
    albo wyczerpany budżet PU/zapytań w ostatnich 60 s — rezerwuje miejsce
    w oknie i zwraca (czas_oczekiwania, wpis_okna). Brak pętli opartej na
    upływie czasu jest celowy: przy zamockowanym time.sleep testy nie mogą
    się zawiesić."""
    with _rl_lock:
        now = time.monotonic()
        while _rl_window and now - _rl_window[0][0] > 60.0:
            _rl_window.popleft()
        wait = max(0.0, _rl_state["pause_until"] - now)
        if len(_rl_window) >= _REQ_BUDGET_PER_MIN:
            wait = max(wait, 60.0 - (now - _rl_window[0][0]))
        pu_sum = sum(e[1] for e in _rl_window)
        if pu_sum >= _PU_BUDGET_PER_MIN:
            acc = pu_sum
            for t_entry, pu in _rl_window:
                acc -= pu
                if acc < _PU_BUDGET_PER_MIN:
                    wait = max(wait, 60.0 - (now - t_entry))
                    break
        n = _pu_stats["requests"]
        est = (_pu_stats["pu"] / n) if n else _PU_ESTIMATE_DEFAULT
        entry = [now + wait, est]
        _rl_window.append(entry)
        return min(wait, 60.0), entry


def _retry_after_seconds(r):
    """Retry-After z odpowiedzi 429. Sentinel Hub podaje go w milisekundach
    (tak też interpretuje go oficjalna biblioteka sentinelhub-py)."""
    v = _header_float(r, "Retry-After")
    if v is None or v <= 0:
        return None
    return max(1.0, min(v / 1000.0, 60.0))


def _post_with_retry(url, json_payload, headers, timeout, refresh_token_fn=None):
    """POST z retry na 429/5xx i jednorazowym odświeżeniem tokenu na 401.
    `refresh_token_fn`, jeśli podane, powinno zwrócić świeży token Bearer.
    Patrz komentarz wyżej: globalna pauza na 429 i budżet PU/zapytań."""
    last_exc = None
    r = None
    for attempt in range(_MAX_RETRIES + 1):
        wait, entry = _reserve_slot()
        if wait > 0:
            time.sleep(wait)
        _sh_gate.acquire()
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            with _rl_lock:
                entry[1] = 0.0
            if attempt >= _MAX_RETRIES_5XX:
                raise RuntimeError(f"Brak połączenia z Sentinel Hub: {e}")
            time.sleep(min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt)))
            continue
        finally:
            _sh_gate.release()

        pu = _header_float(r, "x-processingunits-spent")
        with _rl_lock:
            _pu_stats["requests"] += 1
            if r.status_code == 200 and pu is not None:
                _pu_stats["pu"] += pu
                entry[1] = pu
            elif r.status_code != 200:
                entry[1] = 0.0   # PU są odliczane tylko za udane zapytania
            if r.status_code == 429:
                _pu_stats["rate_limited"] += 1

        if r.status_code == 401 and refresh_token_fn is not None:
            fresh = refresh_token_fn()
            headers["Authorization"] = f"Bearer {fresh}"
            continue

        if r.status_code == 429 and attempt < _MAX_RETRIES:
            delay = _retry_after_seconds(r)
            if delay is None:
                delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt))
            delay += random.uniform(0.0, 0.5)   # rozproszenie wątków po pauzie
            with _rl_lock:
                _rl_state["pause_until"] = max(_rl_state["pause_until"],
                                               time.monotonic() + delay)
            log.warning("Sentinel Hub 429 — pauza %.1f s (próba %d/%d)",
                        delay, attempt + 1, _MAX_RETRIES)
            continue

        if r.status_code in _RETRYABLE_5XX and attempt < _MAX_RETRIES_5XX:
            time.sleep(min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt)))
            continue

        return r

    if last_exc is not None and r is None:
        raise RuntimeError(f"Brak połączenia z Sentinel Hub: {last_exc}")
    return r


# ============================================================
# EVALSCRIPT — Sentinel-1 GRD, IW, VV+VH, sigma-nought (liniowo)
# Przetwarzanie serwerowe: ortorektyfikacja + (opcjonalnie) filtr Lee 5x5
# ============================================================
#
# Poprawka (listopad 2026). Poprzednia wersja:
#   * nie pobierała dataMask i zawsze zwracała dataMask=1, więc piksele
#     prostokąta otaczającego SPOZA działki wchodziły do średniej jako 0
#     (-100 dB po zabezpieczeniu log10) — VV wychodziło ok. -40 dB zamiast
#     ok. -10 dB, a skoki VV i cross-ratio były rozcieńczone o udział tych
#     pikseli (RVI był poprawny, bo to iloraz),
#   * miała własny "filtr Lee" w JS, który nic nie robił: evaluatePixel
#     dostaje kolejne próbki w CZASIE (mosaicking ORBIT), nie okno 5x5
#     sąsiednich pikseli — filtrowała wyłącznie opcja serwerowa,
#   * zwracała średnią z wartości w dB (zaniżoną przez speckle) zamiast dB
#     ze średniej liniowej.
# Teraz evalscript zwraca sigma0 liniowe x10000 jako UINT16 (16 bit zamiast
# FLOAT32 = połowa PU), a dB liczymy w Pythonie ze średniej liniowej działki.
S1_LIN_SCALE = 10000.0

EVALSCRIPT_S1 = """
//VERSION=3
// Sentinel-1 GRD | IW | VV + VH | sigma0 liniowe x10000 (UINT16)
// Ortorektyfikacja i filtr speckle — po stronie serwera (processing).
// dataMask z wejścia: piksele spoza działki / poza pasem zobrazowania
// NIE wchodzą do statystyk.
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"], units: "LINEAR_POWER" }],
    output: [
      { id: "vv_lin",   bands: 1, sampleType: "UINT16" },
      { id: "vh_lin",   bands: 1, sampleType: "UINT16" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(s) {
  if (s.dataMask === 0 || !(s.VV > 0) || !(s.VH > 0)) {
    return { vv_lin: [0], vh_lin: [0], dataMask: [0] };
  }
  return {
    vv_lin: [Math.round(Math.min(s.VV, 6.5) * 10000)],
    vh_lin: [Math.round(Math.min(s.VH, 6.5) * 10000)],
    dataMask: [1]
  };
}
"""

# Filtr speckle po stronie serwera (x2 PU dla zapytania S1). Dla średniej
# z całej działki samo uśrednienie setek pikseli tłumi speckle — zmierzone:
# średnie VV działki z filtrem i bez różnią się średnio o 0,13 dB (maks.
# 0,19 dB). Domyślnie zostaje WŁĄCZONY (bez zmiany metodyki); wyłączenie:
# S1_SPECKLE_FILTER=NONE w .env.
S1_SPECKLE_FILTER = (os.environ.get("S1_SPECKLE_FILTER", "LEE") or "LEE").strip().upper()

# ============================================================
# STAŁE API
# ============================================================

CDSE_STATS_URL  = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
CDSE_TOKEN_URL  = ("https://identity.dataspace.copernicus.eu/auth/realms/"
                   "CDSE/protocol/openid-connect/token")

# Progi detekcji zdarzeń agrotechnicznych (kalibrowane empirycznie)
# VV [dB] dla gołej gleby typowo: –12 do –8 dB
# VV [dB] dla poplonu:            –16 do –12 dB
# Gwałtowny skok VV po orce:       ≥ 3 dB w ciągu ≤ 12 dni
TILLAGE_VV_JUMP_DB       = 3.0     # minimalny skok VV świadczący o orce
TILLAGE_VV_BARE_SOIL_MIN = -13.0   # VV > tej wartości = prawdopodobnie goła gleba po orce
TILLAGE_WINDOW_DAYS      = 12      # max przerwa między obserwacjami dla detekcji zdarzenia
CROSS_RATIO_JUMP_DB      = 1.5     # min. zmiana Cross-Ratio (VH-VV) traktowana jako sygnał pomocniczy

# Podbija pewność o jeden stopień, gdy Cross-Ratio potwierdza skok VV (patrz
# detect_tillage). .get(x, x) w miejscu użycia to celowy fallback — gdyby w
# przyszłości pojawiła się nowa etykieta pewności nieujęta tutaj, zostajemy
# przy niezmienionej wartości zamiast wywalić się KeyError.
_CONF_UP = {"low": "medium", "medium": "high", "high": "high"}
RVI_VEGETATION_MIN       = 0.3     # RVI ≥ tej wartości = wyraźna wegetacja
RVI_BARE_MAX             = 0.1     # RVI < tej wartości = brak wegetacji


# ============================================================
# TOKEN (reuses core.py cache gdy możliwe)
# ============================================================

_s1_token_cache: dict = {"token": None, "exp": 0.0}


def _cdse_token(force_refresh: bool = False) -> str:
    """Pobiera token OAuth z CDSE. Używa oddzielnego cache niż core.py."""
    if (not force_refresh
            and _s1_token_cache["token"]
            and time.time() < _s1_token_cache["exp"] - 60):
        return _s1_token_cache["token"]

    # Preferuj współdzielony cache z core.py, gdy dostępny.
    if not force_refresh:
        try:
            import core as _core
            tok = _core.cdse_token()
            if tok:
                _s1_token_cache["token"] = tok
                _s1_token_cache["exp"] = time.time() + 300
                return tok
        except Exception:
            pass

    cid = os.environ.get("CDSE_CLIENT_ID")
    sec = os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("Brak CDSE_CLIENT_ID / CDSE_CLIENT_SECRET.")
    r = requests.post(CDSE_TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": cid, "client_secret": sec,
    }, timeout=30)
    r.raise_for_status()
    tok = r.json()
    _s1_token_cache["token"] = tok["access_token"]
    _s1_token_cache["exp"]   = time.time() + tok.get("expires_in", 600)
    return _s1_token_cache["token"]


# ============================================================
# POBIERANIE DANYCH S1
# ============================================================

def fetch_s1_series(
    geom_4326,
    t_start: date,
    t_end: date,
    orbit_direction: str = "DESCENDING",   # ASCENDING lub DESCENDING
    resolution_m: int = 10,
) -> list[dict]:
    """
    Pobiera szereg czasowy VV i VH dla poligonu z Sentinel-1 GRD IW.

    Zwraca listę słowników:
    {
        "date": date,
        "vv_db":  float,   # sigma0 VV [dB] = 10*log10(średnia liniowa działki)
        "vh_db":  float,   # sigma0 VH [dB] = 10*log10(średnia liniowa działki)
        "vv_lin": float,   # sigma0 VV [wartość liniowa], średnia działki
        "vh_lin": float,   # sigma0 VH [wartość liniowa], średnia działki
        "cross_ratio_db": float,   # VH - VV [dB]
        "rvi":            float,   # Radar Vegetation Index
        "valid":          bool,    # czy obserwacja wiarygodna
    }

    Optymalizacja (listopad 2026): geometria w UTM, wyjście UINT16, pełny
    dzień t_end, pamięć podręczna z single-flight (patrz sh_utils.py).
    """
    from sh_utils import cached_call, geom_key, text_key
    key = ("s1", geom_key(geom_4326), t_start, t_end, orbit_direction,
           resolution_m, S1_SPECKLE_FILTER, text_key(EVALSCRIPT_S1))
    return cached_call(key, lambda: _fetch_s1_raw(
        geom_4326, t_start, t_end, orbit_direction, resolution_m))


def _fetch_s1_raw(geom_4326, t_start, t_end, orbit_direction, resolution_m) -> list[dict]:
    from sh_utils import stat_bounds, stats_time_range

    bounds, _n_poly_px = stat_bounds(geom_4326, res_m=float(resolution_m))

    processing = {
        "orthorectify":    True,                 # korekcja terenowa (DEM)
        "backCoeff":       "SIGMA0_ELLIPSOID",   # sigma-nought
        "demInstance":     "COPERNICUS",
    }
    if S1_SPECKLE_FILTER == "LEE":
        processing["speckleFilter"] = {"type": "LEE", "windowSizeX": 5, "windowSizeY": 5}

    payload = {
        "input": {
            "bounds": bounds,
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "acquisitionMode":  "IW",
                    "polarization":     "DV",         # Dual VV+VH
                    "orbitDirection":   orbit_direction,
                    "resolution":       "HIGH",        # GRD High = 10m
                },
                "processing": processing,
            }],
        },
        "aggregation": {
            "timeRange": stats_time_range(t_start, t_end),
            "aggregationInterval": {"of": "P1D"},
            "resx": resolution_m,
            "resy": resolution_m,
            "evalscript": EVALSCRIPT_S1,
        },
    }

    headers = {"Authorization": f"Bearer {_cdse_token()}",
               "Content-Type": "application/json"}
    r = _post_with_retry(CDSE_STATS_URL, payload, headers, timeout=180,
                          refresh_token_fn=lambda: _cdse_token(force_refresh=True))

    if r.status_code != 200:
        raise RuntimeError(
            f"Sentinel Hub S1 Statistical API błąd {r.status_code}: {r.text[:400]}"
        )

    records = []
    for item in r.json().get("data", []):
        try:
            outputs = item["outputs"]
            vvl_stats = outputs["vv_lin"]["bands"]["B0"]["stats"]
            vhl_stats = outputs["vh_lin"]["bands"]["B0"]["stats"]
        except (KeyError, TypeError):
            continue

        try:
            vv_lin = float(vvl_stats.get("mean")) / S1_LIN_SCALE
            vh_lin = float(vhl_stats.get("mean")) / S1_LIN_SCALE
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(vv_lin) and math.isfinite(vh_lin)) or vv_lin <= 0 or vh_lin <= 0:
            continue

        vv_db = 10.0 * math.log10(vv_lin)
        vh_db = 10.0 * math.log10(vh_lin)

        d = datetime.fromisoformat(
            item["interval"]["from"].replace("Z", "+00:00")
        ).date()

        sar = compute_sar_indices(vv_db, vh_db, vv_lin, vh_lin)
        records.append({
            "date":    d,
            "vv_db":   round(vv_db,  2),
            "vh_db":   round(vh_db,  2),
            "vv_lin":  round(vv_lin, 6),
            "vh_lin":  round(vh_lin, 6),
            "cross_ratio_db": sar["cross_ratio_db"],
            "rvi":            sar["rvi"],
            "valid":          True,
        })

    return sorted(records, key=lambda x: x["date"])


# ============================================================
# ORBITY WZGLĘDNE (publiczny katalog STAC CDSE, bez kosztu PU)
# ============================================================
#
# Statistical API z agregacją P1D nie zwraca numeru orbity. Nad Polską ten
# sam punkt obserwuje kilka orbit względnych (ścieżek) pod różnymi kątami
# padania — VV z dwóch różnych ścieżek różni się o 1-2 dB bez żadnej zmiany
# na polu. Porównania VV "dziś vs historia" (moduł wodny) mają sens tylko w
# obrębie TEJ SAMEJ orbity względnej. Publiczny katalog STAC CDSE podaje
# `sat:relative_orbit` dla każdej sceny, bez uwierzytelnienia i bez PU.

CDSE_STAC_ITEMS_URL = "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-1-grd/items"
STAC_MAX_PAGES = 5


def fetch_s1_relative_orbits(geom_4326, t_start: date, t_end: date,
                              orbit_direction: str = "DESCENDING") -> dict:
    """{data: numer orbity względnej} dla scen S1 GRD IW pokrywających
    działkę w oknie dat (tylko podany kierunek orbity). Przy kilku scenach
    jednego dnia — pierwsza (w praktyce ta sama ścieżka, kolejne ramki).
    NIEBLOKUJĄCE: przy błędzie sieci zwraca {} (wołający traktuje orbitę
    jako nieznaną)."""
    from sh_utils import cached_call, geom_key
    key = ("s1_relorb", geom_key(geom_4326), t_start, t_end, orbit_direction)
    try:
        return cached_call(key, lambda: _fetch_s1_relative_orbits_raw(
            geom_4326, t_start, t_end, orbit_direction))
    except Exception as e:
        log.warning("Katalog STAC S1 niedostępny (orbity względne): %s", e)
        return {}


def _fetch_s1_relative_orbits_raw(geom_4326, t_start, t_end, orbit_direction) -> dict:
    minx, miny, maxx, maxy = geom_4326.bounds
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "datetime": f"{t_start.isoformat()}T00:00:00Z/{t_end.isoformat()}T23:59:59Z",
        "limit": 100,
    }
    want_state = orbit_direction.lower()
    out: dict = {}
    url = CDSE_STAC_ITEMS_URL
    for _ in range(STAC_MAX_PAGES):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"STAC CDSE błąd {r.status_code}: {r.text[:200]}")
        js = r.json()
        for f in js.get("features", []):
            p = f.get("properties", {})
            if p.get("sar:instrument_mode") not in (None, "IW"):
                continue
            if (p.get("sat:orbit_state") or "").lower() != want_state:
                continue
            rel = p.get("sat:relative_orbit")
            dt = p.get("datetime")
            if rel is None or not dt:
                continue
            d = datetime.fromisoformat(dt.replace("Z", "+00:00")).date()
            out.setdefault(d, int(rel))
        nxt = next((l.get("href") for l in js.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url, params = nxt, None   # link "next" zawiera już wszystkie parametry
    return out


# ============================================================
# OBLICZENIA WSKAŹNIKÓW SAR
# ============================================================

def compute_sar_indices(
    vv_db: float, vh_db: float,
    vv_lin: float, vh_lin: float,
) -> dict:
    """
    Oblicza wskaźniki SAR na podstawie surowych wartości.

    Cross-Ratio (CR):
        CR_dB = VH_dB - VV_dB
        Interpretacja: wyższy CR = więcej rozproszenia objętościowego = więcej wegetacji.
        Typowe wartości: –5 do +2 dB. Nagły wzrost CR po orce: objętość → powierzchnia.

    RVI (Radar Vegetation Index):
        RVI = (4 * VH_lin) / (VV_lin + VH_lin)
        Zakres: [0, 1]. 0 = goła gleba, 1 = gęsta wegetacja.
        Kim & van Zyl (2009): RVI > 0.3 → rozwinięta wegetacja.
    """
    # Cross-Ratio w dB (prosta różnica — działa bo oboje już w dB)
    cr_db = round(vh_db - vv_db, 3)

    # RVI na wartościach liniowych
    denom = vv_lin + vh_lin
    rvi = round((4 * vh_lin) / denom, 4) if denom > 1e-12 else 0.0

    return {
        "cross_ratio_db": cr_db,
        "rvi": rvi,
    }


# ============================================================
# DETEKTOR ORKI / ZDARZEŃ AGROTECHNICZNYCH
# ============================================================

def detect_tillage(
    s1_records: list[dict],
    sowing_date: Optional[date] = None,
    termination_date: Optional[date] = None,
) -> list[dict]:
    """
    Wykrywa zdarzenia agrotechniczne (orka, likwidacja poplonu) w szeregu S1.

    Algorytm oparty na trzech komplementarnych sygnałach:

    1. SKOK VV (główny sygnał):
       Gwałtowny wzrost VV ≥ TILLAGE_VV_JUMP_DB w ciągu ≤ TILLAGE_WINDOW_DAYS
       sugeruje zmianę struktury powierzchni (wegetacja → goła gleba po orce).
       Orka zwiększa chropowatość → silniejsze odbicie VV.

    2. POZIOM VV PO ZDARZENIU (filtr potwierdzający):
       Po skoku VV musi przekroczyć TILLAGE_VV_BARE_SOIL_MIN dB.
       Eliminuje fałszywe alarmy (np. po deszczu lub przy dużej rosy).

    3. SPADEK RVI (sygnał pomocniczy):
       Jednoczesny spadek RVI poniżej RVI_BARE_MAX potwierdza brak wegetacji.

    4. SKOK CROSS-RATIO (sygnał pomocniczy, nigdy samodzielnie nie wyzwala
       zdarzenia): zmiana Cross-Ratio (VH-VV) o ≥ CROSS_RATIO_JUMP_DB w tym
       samym oknie co skok VV — dawniej liczona i pokazywana operatorowi
       wyłącznie w opisie zdarzenia, ale nie wpływała na pewność. Teraz,
       jeśli towarzyszy skokowi VV, podnosi pewność o jeden stopień
       (low→medium, medium→high) jako dodatkowe potwierdzenie zmiany
       struktury odbicia (rozpraszanie objętościowe → powierzchniowe).

    Zwraca listę wykrytych zdarzeń:
    {
        "date":        date,     # data wykrycia (data późniejszej obserwacji)
        "type":        str,      # "likely_tillage" | "possible_tillage" | "vegetation_loss"
        "vv_before":   float,    # VV dB przed zdarzeniem
        "vv_after":    float,    # VV dB po zdarzeniu
        "vv_jump":     float,    # skok VV [dB]
        "rvi_after":   float,    # RVI po zdarzeniu
        "confidence":  str,      # "high" | "medium" | "low"
        "days_gap":    int,      # przerwa między obserwacjami
        "message":     str,      # opis dla operatora
        "within_termination_window": bool,  # czy zgodne z zadeklarowaną likwidacją
    }
    """
    if len(s1_records) < 2:
        return []

    events = []

    for i in range(1, len(s1_records)):
        prev = s1_records[i - 1]
        curr = s1_records[i]

        try:
            d_prev = prev["date"] if isinstance(prev["date"], date) else date.fromisoformat(str(prev["date"]))
            d_curr = curr["date"] if isinstance(curr["date"], date) else date.fromisoformat(str(curr["date"]))
        except (ValueError, KeyError):
            continue

        days_gap = (d_curr - d_prev).days
        if days_gap > TILLAGE_WINDOW_DAYS * 2:
            # Zbyt duża przerwa — nie możemy twierdzić że znamy moment zdarzenia
            continue

        vv_jump = curr["vv_db"] - prev["vv_db"]
        rvi_curr = curr.get("rvi", 1.0)
        vv_curr  = curr["vv_db"]

        # --- Ocena czy to zdarzenie agrotechniczne ---
        event_type   = None
        confidence   = "low"
        messages     = []

        # Sygnał 1: skok VV
        big_vv_jump  = vv_jump >= TILLAGE_VV_JUMP_DB
        # Sygnał 2: VV po zdarzeniu wskazuje na gołą glebę
        bare_vv      = vv_curr >= TILLAGE_VV_BARE_SOIL_MIN
        # Sygnał 3: RVI wskazuje brak wegetacji
        low_rvi      = rvi_curr <= RVI_BARE_MAX
        # Sygnał 4 (pomocniczy, nigdy samodzielnie nie wyzwala zdarzenia):
        # skok Cross-Ratio w tym samym oknie co skok VV.
        cr_curr = curr.get("cross_ratio_db")
        cr_prev = prev.get("cross_ratio_db")
        cr_jump = (cr_curr - cr_prev) if (cr_curr is not None and cr_prev is not None) else None
        cr_confirms = cr_jump is not None and abs(cr_jump) >= CROSS_RATIO_JUMP_DB

        if big_vv_jump and bare_vv and low_rvi:
            event_type = "likely_tillage"
            confidence = "high"
            messages.append(
                f"Skok VV +{vv_jump:.1f} dB → {vv_curr:.1f} dB "
                f"przy VH/VV={curr.get('cross_ratio_db', '?')} dB "
                f"i RVI={rvi_curr:.2f}. "
                "Wszystkie trzy sygnały wskazują na orkę / likwidację poplonu."
            )
        elif big_vv_jump and bare_vv:
            event_type = "likely_tillage"
            confidence = "medium"
            messages.append(
                f"Skok VV +{vv_jump:.1f} dB → {vv_curr:.1f} dB "
                f"(RVI={rvi_curr:.2f} — brak potwierdzenia braku wegetacji). "
                "Prawdopodobna orka lub zmiana pokrycia."
            )
            if cr_confirms:
                confidence = _CONF_UP.get(confidence, confidence)
                messages.append(
                    f"Dodatkowe potwierdzenie: Cross-Ratio zmieniło się o {cr_jump:+.1f} dB "
                    "w tym samym oknie — zgodne ze zmianą struktury odbicia po orce."
                )
        elif big_vv_jump and not bare_vv:
            event_type = "possible_tillage"
            confidence = "low"
            messages.append(
                f"Skok VV +{vv_jump:.1f} dB, ale poziom VV={vv_curr:.1f} dB "
                "nie wskazuje jednoznacznie gołej gleby. Możliwy deszcz lub rosa."
            )
            if cr_confirms:
                confidence = _CONF_UP.get(confidence, confidence)
                messages.append(
                    f"Dodatkowe potwierdzenie: Cross-Ratio zmieniło się o {cr_jump:+.1f} dB "
                    "w tym samym oknie — podnosi to pewność mimo niejednoznacznego poziomu VV."
                )
        elif not big_vv_jump and prev.get("rvi", 0) >= RVI_VEGETATION_MIN and low_rvi:
            # Nagły spadek wegetacji bez skoku VV (np. chemiczna likwidacja)
            event_type = "vegetation_loss"
            confidence = "medium"
            messages.append(
                f"Spadek RVI {prev.get('rvi', '?'):.2f} → {rvi_curr:.2f} "
                f"bez znaczącego skoku VV ({vv_jump:+.1f} dB). "
                "Możliwa chemiczna likwidacja poplonu lub wymarznięcie."
            )

        if event_type is None:
            continue

        # Sprawdź czy zdarzenie jest zgodne z zadeklarowaną datą likwidacji
        within_termination = False
        if termination_date:
            delta = abs((d_curr - termination_date).days)
            within_termination = delta <= 14  # ±14 dni

        # Pomiń jako alarm jeśli siew nieznany lub wydarzenie jest PRZED siewem
        if sowing_date and d_curr < sowing_date:
            continue

        events.append({
            "date":       d_curr.isoformat(),
            "type":       event_type,
            "vv_before":  round(prev["vv_db"], 2),
            "vv_after":   round(curr["vv_db"], 2),
            "vv_jump":    round(vv_jump, 2),
            "rvi_after":  round(rvi_curr, 4),
            "confidence": confidence,
            "days_gap":   days_gap,
            "message":    " ".join(messages),
            "within_termination_window": within_termination,
        })

    return events


# ============================================================
# INTEGRACJA Z WERDYKTEM NDVI
# ============================================================

def assess_with_sar(
    ndvi_verdict: dict,
    s1_records: list[dict],
    sowing_date: Optional[date] = None,
    termination_date: Optional[date] = None,
) -> dict:
    """
    Wzbogaca istniejący werdykt NDVI o dane SAR.

    Logika fuzji:
    - Jeśli NDVI = OBECNY, ale SAR wykrył orkę bez deklaracji likwidacji
      → obniż pewność do PRAWDOPODOBNY i dodaj ostrzeżenie
    - Jeśli NDVI = NIEROZSTRZYGNIETE (chmury), ale SAR ma dane
      → użyj RVI z SAR do oceny poplonu
    - Jeśli oba sygnały zgodne (NDVI wysoki + RVI wysoki)
      → podwyższ pewność do "wysoka (S1+S2)"

    Zwraca wzbogacony słownik werdyktu.
    """
    if not s1_records:
        return {**ndvi_verdict, "sar": None, "sar_events": []}

    tillage_events = detect_tillage(s1_records, sowing_date, termination_date)

    # Najnowszy RVI
    last_rvi = s1_records[-1].get("rvi") if s1_records else None

    # Zdarzenia alarmowe (pewność medium lub high, bez zadeklarowanej likwidacji)
    alarm_events = [
        e for e in tillage_events
        if e["confidence"] in ("high", "medium")
        and not e["within_termination_window"]
    ]

    verdict = dict(ndvi_verdict)
    detail_codes = list(verdict.get("detail_codes") or [])
    sar_summary  = {
        "observations": len(s1_records),
        "last_vv_db":   s1_records[-1]["vv_db"] if s1_records else None,
        "last_vh_db":   s1_records[-1]["vh_db"] if s1_records else None,
        "last_rvi":     last_rvi,
        "tillage_events": len(alarm_events),
    }

    ndvi_status = verdict.get("status")

    # --- Przypadek 1: NDVI mówi OK, ale SAR widzi orkę ---
    if alarm_events and ndvi_status in ("OBECNY", "PRAWDOPODOBNY"):
        verdict["status"]     = "PRAWDOPODOBNY"
        verdict["confidence"] = "obniżona (S1 alarm)"
        verdict["compliance_issue"] = True
        detail_codes.append(("sar_tillage_alarm", {
            "events": len(alarm_events),
            "last_date": alarm_events[-1]["date"],
            "vv_jump":   alarm_events[-1]["vv_jump"],
            "confidence": alarm_events[-1]["confidence"],
        }))

    # --- Przypadek 2: NDVI nierozstrzygnięte (chmury), SAR ma dane ---
    elif ndvi_status == "NIEROZSTRZYGNIETE" and last_rvi is not None:
        if last_rvi >= RVI_VEGETATION_MIN:
            verdict["status"]     = "PRAWDOPODOBNY"
            verdict["confidence"] = "średnia (SAR backup)"
            detail_codes.append(("sar_backup_vegetation", {
                "rvi": last_rvi, "threshold": RVI_VEGETATION_MIN,
            }))
        elif last_rvi <= RVI_BARE_MAX:
            verdict["status"]     = "BRAK"
            verdict["confidence"] = "średnia (SAR backup)"
            detail_codes.append(("sar_backup_bare", {
                "rvi": last_rvi, "threshold": RVI_BARE_MAX,
            }))
        else:
            detail_codes.append(("sar_inconclusive", {"rvi": last_rvi}))

    # --- Przypadek 3: Oba sygnały zgodne = wyższa pewność ---
    elif (ndvi_status == "OBECNY"
          and verdict.get("confidence") == "wysoka"
          and last_rvi is not None
          and last_rvi >= RVI_VEGETATION_MIN
          and not alarm_events):
        verdict["confidence"] = "wysoka (S1+S2)"
        detail_codes.append(("sar_confirms_vegetation", {"rvi": last_rvi}))

    verdict["detail_codes"] = detail_codes
    verdict["sar"]          = sar_summary
    verdict["sar_events"]   = alarm_events

    return verdict


# ============================================================
# TABELA SQLite — schemat rozszerzony o S1
# ============================================================

S1_DB_SCHEMA = """
-- Obserwacje Sentinel-1 (SAR) — tabela uzupełniająca ndvi_observations
CREATE TABLE IF NOT EXISTS s1_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    parcel_db_id     INTEGER NOT NULL,
    obs_date         TEXT NOT NULL,
    vv_db            REAL NOT NULL,   -- sigma0 VV [dB]
    vh_db            REAL NOT NULL,   -- sigma0 VH [dB]
    vv_lin           REAL,            -- sigma0 VV [liniowy]
    vh_lin           REAL,            -- sigma0 VH [liniowy]
    cross_ratio_db   REAL,            -- VH - VV [dB]
    rvi              REAL,            -- Radar Vegetation Index [0-1]
    orbit_direction  TEXT DEFAULT 'DESCENDING',
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE(parcel_db_id, obs_date, orbit_direction),
    FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
);

-- Zdarzenia agrotechniczne wykryte przez SAR
CREATE TABLE IF NOT EXISTS sar_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    parcel_db_id     INTEGER NOT NULL,
    event_date       TEXT NOT NULL,
    event_type       TEXT NOT NULL,    -- likely_tillage | possible_tillage | vegetation_loss
    vv_before        REAL,
    vv_after         REAL,
    vv_jump          REAL,
    rvi_after        REAL,
    confidence       TEXT,             -- high | medium | low
    message          TEXT,
    reviewed         INTEGER DEFAULT 0,
    created_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
);
"""


def init_s1_tables(db_path: str = "monitoring.db"):
    """Tworzy tabele S1 w bazie jeśli nie istnieją."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.executescript(S1_DB_SCHEMA)


def save_s1_observations(parcel_db_id: int, records: list[dict],
                          orbit: str = "DESCENDING", db_path: str = "monitoring.db"):
    """Zapisuje obserwacje S1 do bazy (INSERT OR IGNORE)."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        for r in records:
            obs_date = r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"])
            conn.execute("""
                INSERT OR IGNORE INTO s1_observations
                (parcel_db_id, obs_date, vv_db, vh_db, vv_lin, vh_lin,
                 cross_ratio_db, rvi, orbit_direction)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (parcel_db_id, obs_date, r["vv_db"], r["vh_db"],
                  r.get("vv_lin"), r.get("vh_lin"),
                  r.get("cross_ratio_db"), r.get("rvi"), orbit))


def get_s1_observations(parcel_db_id: int, db_path: str = "monitoring.db") -> list[dict]:
    """Pobiera obserwacje S1 z bazy."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM s1_observations WHERE parcel_db_id=? ORDER BY obs_date",
            (parcel_db_id,)
        ).fetchall()
    return [dict(r) for r in rows]
