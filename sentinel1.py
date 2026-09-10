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

import math
import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests

# Timeouty i retry na chwilowe błędy Sentinel Hub (429 rate-limit, 5xx).
# Bez tego pojedynczy rate-limit rutynowo degradował cały batch do
# NDVI-only, mimo że ponowna próba za chwilę zwykle się udaje.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0

# Globalny limit RÓWNOCZESNYCH requestów do Sentinel Hub, wspólny dla
# core.py/analytics.py/sentinel1.py (wszystkie idą przez _post_with_retry).
# Naprawa regresji: równoległe odpalenie NDVI+multi/S1 w obrębie jednej
# działki (patrz app.py _process_one) samo w sobie podwoiło realne
# obciążenie Sentinel Hub, a każda kolejna pozycja batcha (do
# BATCH_CONCURRENCY naraz) dokłada swoje. Bez twardego globalnego limitu to
# skutkowało seriami 429, które _post_with_retry owszem próbuje ponowić, ale
# kosztem narastającego backoffu (do ok. 14 s na wywołanie) — czyli batch
# robił się WOLNIEJSZY niż przed równoległością, a po wyczerpaniu retry
# check_multi finalnie się wywalał i wynik spadał z powrotem do samego NDVI.
# Semafor ogranicza to do stałej liczby jednocześnie "w locie" requestów,
# niezależnie od tego, ile wątków próbuje jednocześnie wystartować.
_SH_MAX_CONCURRENT = int(os.environ.get("SENTINEL_HUB_MAX_CONCURRENT", "4"))
_sh_gate = threading.Semaphore(_SH_MAX_CONCURRENT)


def _post_with_retry(url, json_payload, headers, timeout, refresh_token_fn=None):
    """POST z retry na 429/5xx (backoff wykładniczy) i jednorazowym
    odświeżeniem tokenu na 401. `refresh_token_fn`, jeśli podane, powinno
    zwrócić świeży token Bearer."""
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        _sh_gate.acquire()
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            if attempt >= _MAX_RETRIES:
                raise RuntimeError(f"Brak połączenia z Sentinel Hub: {e}")
            time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
            continue
        finally:
            _sh_gate.release()

        if r.status_code == 401 and refresh_token_fn is not None:
            fresh = refresh_token_fn()
            headers["Authorization"] = f"Bearer {fresh}"
            continue

        if r.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
            time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
            continue

        return r

    if last_exc:
        raise RuntimeError(f"Brak połączenia z Sentinel Hub: {last_exc}")
    return r


# ============================================================
# EVALSCRIPT — Sentinel-1 GRD, IW, VV+VH, sigma-nought [dB]
# Przetwarzanie serwerowe: Orthorectification + Lee Speckle Filter (5×5)
# ============================================================

EVALSCRIPT_S1 = """
//VERSION=3
// Sentinel-1 GRD | IW | VV + VH | sigma0 [dB]
// Server-side: Orthorectification (DEM) + Lee speckle filter 5x5

function setup() {
  return {
    input: [{
      bands: ["VV", "VH"],
      units: "LINEAR_POWER",      // liniowe wartości mocy (nie dB) — konwersja po stronie evalscript
    }],
    output: [
      { id: "vv_db",    bands: 1, sampleType: "FLOAT32" },
      { id: "vh_db",    bands: 1, sampleType: "FLOAT32" },
      { id: "vv_lin",   bands: 1, sampleType: "FLOAT32" },
      { id: "vh_lin",   bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8"   },
    ],
    mosaicking: "ORBIT",         // jedna orbita na interwał — unikamy mieszania orbit
  };
}

// Filtr Lee 5×5 (klasyczna implementacja na liniowych wartościach mocy)
function leeFilter(samples, targetBand, wSize) {
  const half = Math.floor(wSize / 2);
  const n = wSize * wSize;

  let sum = 0, sumSq = 0, count = 0;
  for (let dy = -half; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++) {
      const idx = (dy + half) * wSize + (dx + half);
      const s = samples[idx];
      if (s && s[targetBand] !== undefined && !isNaN(s[targetBand])) {
        sum   += s[targetBand];
        sumSq += s[targetBand] * s[targetBand];
        count++;
      }
    }
  }
  if (count === 0) return NaN;

  const mean    = sum / count;
  const varNoise = mean * mean * (1 / n);      // model szumu Rayleigha
  const varImage = Math.max(0, sumSq / count - mean * mean);
  const weight   = varImage / (varImage + varNoise);
  const center   = samples[Math.floor(n / 2)];
  return mean + weight * ((center ? center[targetBand] : mean) - mean);
}

function evaluatePixel(samples) {
  // samples to tablica sąsiadujących pikseli (okno 5×5 = 25 elementów)
  // Centrum okna to samples[12]
  const center = samples[12] || samples[0];

  if (!center || center.VV === undefined || center.dataMask === 0) {
    return {
      vv_db:    [NaN],
      vh_db:    [NaN],
      vv_lin:   [NaN],
      vh_lin:   [NaN],
      dataMask: [0],
    };
  }

  // Filtr Lee 5×5 na wartościach liniowych
  const vv_filt = leeFilter(samples, "VV", 5);
  const vh_filt = leeFilter(samples, "VH", 5);

  // Zabezpieczenie przed wartościami ≤ 0 (log10 nie zdefiniowany)
  const vv_safe = vv_filt > 0 ? vv_filt : 1e-10;
  const vh_safe = vh_filt > 0 ? vh_filt : 1e-10;

  // Konwersja do dB: sigma0_dB = 10 * log10(sigma0_linear)
  const vv_db = 10 * Math.log10(vv_safe);
  const vh_db = 10 * Math.log10(vh_safe);

  return {
    vv_db:    [vv_db],
    vh_db:    [vh_db],
    vv_lin:   [vv_safe],
    vh_lin:   [vh_safe],
    dataMask: [1],
  };
}
"""

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
        "vv_db":  float,   # sigma0 VV [dB], po filtrze Lee
        "vh_db":  float,   # sigma0 VH [dB], po filtrze Lee
        "vv_lin": float,   # sigma0 VV [wartość liniowa]
        "vh_lin": float,   # sigma0 VH [wartość liniowa]
        "cross_ratio_db": float,   # VH - VV [dB]
        "rvi":            float,   # Radar Vegetation Index
        "valid":          bool,    # czy obserwacja wiarygodna
    }
    """
    from shapely.geometry import mapping as shp_mapping
    from shapely.ops      import transform as shp_transform
    from pyproj import Transformer

    to_3857 = Transformer.from_crs(4326, 3857, always_xy=True).transform
    geom_3857 = shp_transform(to_3857, geom_4326)

    payload = {
        "input": {
            "bounds": {
                "geometry":   shp_mapping(geom_3857),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "acquisitionMode":  "IW",
                    "polarization":     "DV",         # Dual VV+VH
                    "orbitDirection":   orbit_direction,
                    "resolution":       "HIGH",        # GRD High = 10m
                    # Orthorectification i speckle filter wymuszamy przez
                    # processing options (server-side)
                },
                "processing": {
                    "orthorectify":    True,            # korekcja terenowa (DEM)
                    "backCoeff":       "SIGMA0_ELLIPSOID",  # sigma-nought
                    "speckleFilter": {
                        "type":       "LEE",
                        "windowSizeX": 5,
                        "windowSizeY": 5,
                    },
                    "demInstance": "COPERNICUS",
                },
            }],
        },
        "aggregation": {
            "timeRange": {
                "from": f"{t_start.isoformat()}T00:00:00Z",
                "to":   f"{t_end.isoformat()}T23:59:59Z",
            },
            "aggregationInterval": {"of": "P1D"},
            "resx": resolution_m,
            "resy": resolution_m,
            "evalscript": EVALSCRIPT_S1,
        },
        "calculations": {
            # Statistical API liczy średnie per output band automatycznie
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
            vv_stats  = outputs["vv_db"]["bands"]["B0"]["stats"]
            vh_stats  = outputs["vh_db"]["bands"]["B0"]["stats"]
            vvl_stats = outputs["vv_lin"]["bands"]["B0"]["stats"]
            vhl_stats = outputs["vh_lin"]["bands"]["B0"]["stats"]
        except (KeyError, TypeError):
            continue

        vv_db  = vv_stats.get("mean")
        vh_db  = vh_stats.get("mean")
        vv_lin = vvl_stats.get("mean")
        vh_lin = vhl_stats.get("mean")

        if any(v is None or math.isnan(v) for v in (vv_db, vh_db, vv_lin, vh_lin)):
            continue
        if vv_db == 0 and vh_db == 0:
            continue

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
