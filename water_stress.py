"""Silnik anomalii stresu wodnego — Krok 2 planu modułu teledetekcji i wody.

Liczy sygnał ryzyka stresu wodnego / nieprawidłowego nawodnienia dla jednego
pola z water_monitored_fields (patrz db.py, Krok 1), fuzjonując źródła, które
w tym repo już istnieją i są używane w module poplonów — żadna z tych
integracji nie jest tu nowa:

  - NDWI (Sentinel-2, canopy water content) — analytics.fetch_multi_series,
    ten sam evalscript co przy ocenie poplonu.
  - VV backscatter (Sentinel-1) — sentinel1.fetch_s1_series. Używamy VV (nie
    RVI), bo to VV jest wrażliwe na wilgotność gleby i zdarzenia nawadniania;
    RVI (już liczone w sentinel1.py) mierzy gęstość wegetacji — przydatne w
    module poplonów, ale to inny sygnał fizyczny, nie proxy wilgotności.
  - Opady / posucha (Open-Meteo) — weather.fetch_weather_series /
    detect_dry_spell_dates, już zintegrowane w module poplonów.
  - Log nawodnienia (irrigation_events) — Krok 1 tego planu.

ZASADA Z PLANU MODUŁU (sekcja 3.1-3.2): anomalia liczona względem WŁASNEJ
HISTORII pola i względem KOHORTY podobnych pól — nie względem sztywnego
globalnego progu. Wynik to FLAGA RYZYKA, nie diagnoza przyczyny (moduł nie
rozróżnia stresu wodnego od zasolenia/choroby/azotu — to zadanie doradcy w
terenie). Każdy wynik niesie status "niekalibrowane / wymaga walidacji
terenowej" — to świadomie nieskalibrowany silnik pierwszej wersji, budowany
po to, żeby zacząć zbierać potwierdzenia terenowe pod przyszłą rekalibrację
(Krok 3: pętla zwrotna, analogicznie do calibration.py w module poplonów).

DEFINICJA "PODOBNEGO POLA" (kohorta): pola aktywne o tym samym `region`,
tym samym `irrigation_mode` i tej samej uprawie `crop_type` (porównanie
bez rozróżnienia wielkości liter; pola bez podanej uprawy porównujemy tylko
z innymi polami bez uprawy). Wartość kohorty bierzemy z obserwacji
najbliższej dacie obserwacji pola, maks. COHORT_DATE_TOLERANCE_DAYS dni
różnicy — porównanie NDMI z dziś z NDMI sąsiada sprzed 10 dni nie mówi
nic o tym, czy problem jest regionalny. W repo nie ma jeszcze
danych glebowych/klimatycznych do lepszego dopasowania — jeśli w praktyce
grupowanie po samym `region` okaże się za szerokie albo za wąskie, to jest
jedyne miejsce do zmiany (get_cohort_fields).
"""
import json
import logging
import statistics
from datetime import date, timedelta
from typing import Optional

from shapely import wkt as shapely_wkt

import core
import db

WATER_ALARM_RECENT_DAYS = 7  # patrz db.water_alarm_exists_recent
UNIFORMITY_RISK_TYPES = ("irrigation_uniformity_anomaly", "uneven_moisture_risk")
BALANCE_RISK_TYPE = "water_balance_deficit"
# Typy alarmów, których NIE wyzwala z-score spadku sygnału — nie wchodzą do
# kalibracji Z_SCORE_RISK_THRESHOLD (compute_suggested_water_threshold).
NON_DECLINE_RISK_TYPES = UNIFORMITY_RISK_TYPES + (BALANCE_RISK_TYPE,)
# Alarm z modelu bilansu wodnego: stres już trwa albo nastąpi w ciągu tylu dni.
BALANCE_ALERT_DAYS = 3

log = logging.getLogger("water_stress")

# --- progi robocze (NIEKALIBROWANE — patrz nagłówek modułu) -----------------

OWN_HISTORY_WINDOW = 6            # ile poprzednich obserwacji pola bierzemy jako linię bazową
OWN_HISTORY_MIN_SAMPLES = 3       # poniżej tego nie liczymy z-score (za mało historii)
COHORT_MIN_FIELDS = 2             # poniżej tego nie liczymy porównania z kohortą
Z_SCORE_RISK_THRESHOLD = -1.5     # spadek >= 1.5 odch. std. poniżej własnej historii pola
COHORT_FIELD_SPECIFIC_Z = -1.0    # pole wyraźnie gorsze niż kohorta -> sygnał punktowy, nie regionalny
COHORT_DATE_TOLERANCE_DAYS = 3    # maks. różnica dat obserwacji pola i pola z kohorty
LOOKBACK_DAYS = 20                # okno pobierania danych dla jednej obserwacji
WEATHER_LOOKBACK_DAYS = 32        # okno pogody: bilans P-ET0 do 30 dni przed obserwacją
IRRIGATION_RECENT_DAYS = 5        # "niedawne" nawodnienie -> kontekst irrigation-performance

# Minimalny udział czystych (po masce SCL) pikseli pola, żeby scena mogła
# zostać obserwacją modułu wodnego. Chmura/mgiełka/cień obniżają NDMI tak samo
# jak przesuszenie łanu — bez tego progu częściowo zachmurzona scena dawała
# fałszywy alarm "stres wodny". W module poplonów wystarczy, że jakaś część
# pola jest widoczna (pytanie "czy jest okrywa"), tu mierzymy subtelny spadek
# wilgotności, więc wymagamy większości pola.
WATER_MIN_VALID_FRAC = 0.6

# Odporna anomalia własnej historii (listopad 2026): mediana + MAD zamiast
# średniej + odchylenia standardowego. Przy 3-6 próbkach odchylenie
# standardowe jest bardzo niestabilne (jedna odstająca scena rozdmuchuje albo
# zeruje skalę). SIGMA_FLOOR to dolna granica skali = typowy szum pomiaru
# średniej z pola (NDMI z S2 ok. 0.02, VV z S1 ok. 0.5 dB) — bez niej
# idealnie płaska historia dawała z-score -10 przy zmianie o 0.01.
MAD_TO_SIGMA = 1.4826
SIGMA_FLOOR = {"ndwi_mean": 0.02, "vv_db": 0.5, "ndwi_residual": 0.02}
# Minimalny spadek bezwzględny względem mediany historii — drugi, niezależny
# od skali warunek alarmu (z-score bywa duży także przy zmianie bez
# znaczenia agronomicznego).
MIN_ABS_DROP = {"ndwi_mean": 0.04, "vv_db": 1.0, "ndwi_residual": 0.04}

# NDMI rezydualny względem NDVI: niski NDMI może znaczyć "sucho" albo po
# prostu "mniej liści" (dojrzewanie, zbiór, koszenie). Oczekiwaną zmianę
# NDMI liczymy z nachylenia NDMI~NDVI historii pola (k) i porównujemy z
# obserwowaną: rezyduum = ΔNDMI − k·ΔNDVI (Δ względem mediany historii).
# Spadek NDMI przy stabilnym NDVI = wczesny sygnał stresu wodnego; spadek
# obu proporcjonalnie = raczej fenologia/zbiór.
RESIDUAL_SLOPE_DEFAULT = 0.8          # gdy NDVI w historii prawie stałe (nachylenia nie da się dopasować)
RESIDUAL_SLOPE_BOUNDS = (0.3, 1.2)    # fizycznie sensowny zakres nachylenia NDMI~NDVI
RESIDUAL_MIN_NDVI_RANGE = 0.08        # min. rozpiętość NDVI w historii do dopasowania nachylenia

# VV (Sentinel-1) jako proxy wilgotności: tylko w obrębie tej samej orbity
# względnej (inny kąt padania = inny poziom VV, patrz
# sentinel1.fetch_s1_relative_orbits) i tylko przy rzadszym łanie — pasmo C
# pod gęstą roślinnością rozprasza się głównie w łanie i prawie nie "widzi"
# gleby, więc spadek VV mówi wtedy o łanie, nie o wilgotności. Próg NDVI
# roboczy (niekalibrowany).
VV_MAX_NDVI = 0.5

# Minimalne NDVI, przy którym NDMI opisuje wilgotność łanu (a nie gleby /
# resztek pożniwnych). Roboczo; powtórka sezonu 2026 — alarmy na
# ścierniskach z NDVI ~0.2 były bez sensu agronomicznego.
MIN_CANOPY_NDVI = 0.35
# Spadek NDVI względem mediany historii pola, od którego traktujemy zmianę
# jako zbiór / koszenie / likwidację, a nie stres (roboczo).
HARVEST_NDVI_DROP = 0.25

# Równomierność wilgotności w obrębie pola (rozkład NDMI z Statistical API,
# patrz analytics._ndwi_distribution). Średnia z pola może być w normie,
# gdy część pola jest sucha (koniec bruzdy, niedolany sektor, piaszczysta
# plama). Rozrzut = mediana − 10. percentyl NDMI. Alarm, gdy rozrzut wyraźnie
# wzrósł względem historii pola ORAZ istotna część pola jest wyraźnie sucha.
# Progi robocze (niekalibrowane).
UNIFORMITY_MIN_SPREAD_INCREASE = 0.05   # wzrost rozrzutu p50−p10 względem mediany historii
UNIFORMITY_MIN_LOW_FRAC = 0.15          # min. udział "suchych" pikseli (NDMI < mediana − 0.10)
SIGMA_FLOOR["ndwi_spread"] = 0.015

# Z_SCORE_RISK_THRESHOLD jest od tego miejsca KALIBROWALNE w runtime (patrz
# reload_water_threshold/apply_water_threshold nizej) -- ta stala to tylko
# wartosc startowa, analogicznie do core.DEFAULT_NDVI_COVER_HIGH.
DEFAULT_Z_SCORE_RISK_THRESHOLD = Z_SCORE_RISK_THRESHOLD


def reload_water_threshold():
    """Wczytuje skalibrowany Z_SCORE_RISK_THRESHOLD z tabeli `settings`, jesli
    zostal kiedys zatwierdzony przez operatora (patrz apply_water_threshold) --
    analogicznie do core.reload_thresholds() w module poplonow. Bezpieczne do
    wywolania wielokrotnie/przed init_db (wtedy zostaje wartosc domyslna)."""
    global Z_SCORE_RISK_THRESHOLD
    try:
        val = db.get_setting("water_zscore_risk_threshold")
        Z_SCORE_RISK_THRESHOLD = float(val) if val is not None else DEFAULT_Z_SCORE_RISK_THRESHOLD
    except Exception:
        Z_SCORE_RISK_THRESHOLD = DEFAULT_Z_SCORE_RISK_THRESHOLD


def _zscore(current: Optional[float], history: list) -> Optional[float]:
    """Z-score `current` względem `history` (current NIE wchodzi do
    history). None, gdy za mało próbek albo brak wariancji w historii."""
    values = [v for v in history if v is not None]
    if current is None or len(values) < OWN_HISTORY_MIN_SAMPLES:
        return None
    mean = statistics.mean(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        return None
    if stdev == 0:
        # Idealnie plaska historia (brak wariancji) -> z-score nieokreslony;
        # w praktyce (szum sensora S1/S2, pogoda) to praktycznie sie nie
        # zdarza, wiec nie warto dorabiac specjalnej logiki pod ten
        # przypadek brzegowy kosztem false-positive na naturalny mikro-szum.
        return None
    return round((current - mean) / stdev, 3)


def _norm_crop(crop) -> Optional[str]:
    if crop is None:
        return None
    crop = str(crop).strip().lower()
    return crop or None


def get_cohort_fields(field: dict) -> list:
    """Pola 'podobne' do porównania — patrz docstring modułu: ten sam
    region + ten sam irrigation_mode + ta sama uprawa, aktywne, bez samego pola."""
    region = field.get("region")
    if not region:
        return []
    crop = _norm_crop(field.get("crop_type"))
    active = db.get_water_fields(status="active")
    return [f for f in active
            if f["id"] != field["id"]
            and f.get("region") == region
            and f.get("irrigation_mode") == field.get("irrigation_mode")
            and _norm_crop(f.get("crop_type")) == crop]


def cohort_values_near_date(cohort: list, obs_date_iso: str, key: str = "ndwi_mean",
                            tolerance_days: int = None) -> list:
    """Dla każdego pola z kohorty: wartość `key` z obserwacji najbliższej
    `obs_date_iso`, o ile różnica dat <= tolerance_days. Pola bez takiej
    obserwacji są pomijane (zamiast porównywać z nieaktualną wartością)."""
    tol = COHORT_DATE_TOLERANCE_DAYS if tolerance_days is None else tolerance_days
    target = date.fromisoformat(obs_date_iso)
    values = []
    for cf in cohort:
        best = None
        for o in db.get_water_observations(cf["id"]):
            if o.get(key) is None:
                continue
            diff = abs((date.fromisoformat(o["obs_date"]) - target).days)
            if diff <= tol and (best is None or diff < best[0]):
                best = (diff, o[key])
        if best is not None:
            values.append(best[1])
    return values


def _robust_zscore(current: Optional[float], history: list, floor: float) -> Optional[tuple]:
    """(z, median) — odporny z-score: (current − mediana) / max(1.4826·MAD, floor).
    None, gdy za mało próbek."""
    values = [v for v in history if v is not None]
    if current is None or len(values) < OWN_HISTORY_MIN_SAMPLES:
        return None
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    scale = max(MAD_TO_SIGMA * mad, floor)
    return round((current - med) / scale, 3), med


def own_history_anomaly(observations: list, key: str) -> Optional[dict]:
    """Odporny z-score ostatniej obserwacji względem do OWN_HISTORY_WINDOW
    poprzednich obserwacji TEGO SAMEGO pola i tego samego wskaźnika
    (mediana/MAD z dolną granicą skali SIGMA_FLOOR). `significant_drop`
    = z-score poniżej progu ORAZ spadek bezwzględny >= MIN_ABS_DROP."""
    if len(observations) < OWN_HISTORY_MIN_SAMPLES + 1:
        return None
    history = [o.get(key) for o in observations[:-1]][-OWN_HISTORY_WINDOW:]
    current = observations[-1].get(key)
    res = _robust_zscore(current, history, SIGMA_FLOOR.get(key, 0.0))
    if res is None:
        return None
    z, med = res
    valid_hist = [v for v in history if v is not None]
    delta = round(current - med, 4)
    return {"current": current, "history_median": round(med, 4),
            "history_mean": round(statistics.mean(valid_hist), 4),
            "delta": delta, "z_score": z, "n_samples": len(valid_hist),
            "significant_drop": bool(z <= Z_SCORE_RISK_THRESHOLD
                                     and delta <= -MIN_ABS_DROP.get(key, 0.0))}


def own_vv_anomaly(observations: list) -> Optional[dict]:
    """Anomalia VV względem historii pola z TEJ SAMEJ orbity względnej S1
    (gdy orbita bieżącej obserwacji jest znana). Przy gęstym łanie
    (NDVI > VV_MAX_NDVI) wynik jest liczony, ale oznaczony jako
    nieużywany do decyzji (`used_for_decision=False`)."""
    if not observations:
        return None
    cur = observations[-1]
    orbit = cur.get("s1_rel_orbit")
    if orbit is not None:
        hist = [o for o in observations[:-1] if o.get("s1_rel_orbit") == orbit]
        series = hist + [cur]
    else:
        series = observations
    res = own_history_anomaly(series, "vv_db")
    if res is None:
        return None
    ndvi = cur.get("ndvi_mean")
    dense = ndvi is not None and ndvi > VV_MAX_NDVI
    res["rel_orbit"] = orbit
    res["dense_canopy"] = dense
    res["used_for_decision"] = not dense
    return res


def uniformity_anomaly(observations: list) -> Optional[dict]:
    """Wzrost niejednorodności NDMI w polu względem historii pola.
    None, gdy brak percentyli w bieżącej obserwacji / za krótka historia."""
    if len(observations) < OWN_HISTORY_MIN_SAMPLES + 1:
        return None
    cur = observations[-1]
    if cur.get("ndwi_p50") is None or cur.get("ndwi_p10") is None:
        return None

    def spread(o):
        if o.get("ndwi_p50") is None or o.get("ndwi_p10") is None:
            return None
        return o["ndwi_p50"] - o["ndwi_p10"]

    cur_spread = spread(cur)
    hist = [spread(o) for o in observations[:-1][-OWN_HISTORY_WINDOW:]]
    res = _robust_zscore(cur_spread, hist, SIGMA_FLOOR["ndwi_spread"])
    if res is None:
        return None
    z, med = res
    low_frac = cur.get("ndwi_low_frac")
    increase = cur_spread - med
    flagged = bool(z >= -Z_SCORE_RISK_THRESHOLD
                   and increase >= UNIFORMITY_MIN_SPREAD_INCREASE
                   and low_frac is not None and low_frac >= UNIFORMITY_MIN_LOW_FRAC)
    return {"current_spread": round(cur_spread, 4), "history_median_spread": round(med, 4),
            "spread_increase": round(increase, 4), "z_score": z,
            "low_frac": low_frac, "n_samples": len([h for h in hist if h is not None]),
            "flagged": flagged}


def _ndmi_ndvi_slope(pairs: list) -> float:
    """Nachylenie NDMI~NDVI (MNK) z historii pola, przycięte do
    RESIDUAL_SLOPE_BOUNDS; domyślne, gdy NDVI w historii prawie stałe."""
    xs = [p[0] for p in pairs]
    if len(pairs) < OWN_HISTORY_MIN_SAMPLES or max(xs) - min(xs) < RESIDUAL_MIN_NDVI_RANGE:
        return RESIDUAL_SLOPE_DEFAULT
    mx = statistics.mean(xs)
    my = statistics.mean(p[1] for p in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    k = sxy / sxx if sxx > 0 else RESIDUAL_SLOPE_DEFAULT
    lo, hi = RESIDUAL_SLOPE_BOUNDS
    return min(hi, max(lo, k))


def ndmi_residual_anomaly(observations: list) -> Optional[dict]:
    """Anomalia NDMI po odjęciu zmiany wyjaśnionej zmianą NDVI (patrz
    RESIDUAL_* wyżej). None, gdy brakuje NDVI/NDMI w bieżącej obserwacji
    albo w historii jest za mało par."""
    if len(observations) < OWN_HISTORY_MIN_SAMPLES + 1:
        return None
    cur = observations[-1]
    if cur.get("ndwi_mean") is None or cur.get("ndvi_mean") is None:
        return None
    pairs = [(o["ndvi_mean"], o["ndwi_mean"]) for o in observations[:-1][-OWN_HISTORY_WINDOW:]
             if o.get("ndvi_mean") is not None and o.get("ndwi_mean") is not None]
    if len(pairs) < OWN_HISTORY_MIN_SAMPLES:
        return None
    k = _ndmi_ndvi_slope(pairs)
    med_x = statistics.median(p[0] for p in pairs)
    med_y = statistics.median(p[1] for p in pairs)
    hist_res = [(y - med_y) - k * (x - med_x) for x, y in pairs]
    d_ndvi = cur["ndvi_mean"] - med_x
    d_ndmi = cur["ndwi_mean"] - med_y
    residual = d_ndmi - k * d_ndvi
    res = _robust_zscore(residual, hist_res, SIGMA_FLOOR["ndwi_residual"])
    if res is None:
        return None
    z, med_res = res
    delta = residual - med_res
    return {"current_residual": round(residual, 4), "slope_k": round(k, 3),
            "delta_ndvi": round(d_ndvi, 4), "delta_ndmi": round(d_ndmi, 4),
            "expected_delta_ndmi": round(k * d_ndvi, 4),
            "z_score": z, "n_samples": len(pairs),
            "significant_drop": bool(z <= Z_SCORE_RISK_THRESHOLD
                                     and delta <= -MIN_ABS_DROP["ndwi_residual"])}


def cohort_anomaly(current: Optional[float], cohort_values: list) -> Optional[dict]:
    """Z-score bieżącej wartości pola względem wartości KONKURENCYJNYCH
    (z tego samego okresu) pól z kohorty."""
    values = [v for v in cohort_values if v is not None]
    if current is None or len(values) < COHORT_MIN_FIELDS:
        return None
    z = _zscore(current, values)
    if z is None:
        return None
    return {"current": current, "cohort_mean": round(statistics.mean(values), 4),
            "cohort_n": len(values), "z_score": z}


def last_irrigation_context(events: list, obs_date_iso: str) -> dict:
    """Ile dni temu było ostatnie zarejestrowane nawodnienie przed dniem
    obserwacji (nie po) i jaką metodą — kontekst do odróżnienia
    'water-stress risk' od 'irrigation-performance anomaly' (sekcja 3.1
    planu: to ten sam silnik, różna interpretacja w trybie irygacyjnym)."""
    if not events:
        return {"days_since_last_event": None, "method": None}
    obs_d = date.fromisoformat(obs_date_iso)
    past = [e for e in events if date.fromisoformat(e["event_date"]) <= obs_d]
    if not past:
        return {"days_since_last_event": None, "method": None}
    last = max(past, key=lambda e: e["event_date"])
    days = (obs_d - date.fromisoformat(last["event_date"])).days
    return {"days_since_last_event": days, "method": last["method"]}


def _cwb_state(obs: dict) -> Optional[str]:
    """'deficit' / 'surplus' / 'neutral' wg bilansu P-ET0 z 14 dni, None gdy brak danych."""
    import weather as weather_mod
    cwb = obs.get("cwb_14d_mm")
    if cwb is None:
        return None
    if cwb <= weather_mod.CWB_DEFICIT_14D_MM:
        return "deficit"
    if cwb >= weather_mod.CWB_SURPLUS_14D_MM:
        return "surplus"
    return "neutral"


def _climate_sentence(obs: dict) -> str:
    """Bilans 30 dni na tle normy 1991–2020 (gdy policzony)."""
    import weather as weather_mod
    pct = obs.get("cwb_30d_pct")
    if pct is None:
        return ""
    label = weather_mod.cwb_class(pct)
    return (f" Na tle lat 1991–2020 ostatnie 30 dni: {label} "
            f"({pct:.0f}. percentyl bilansu opad−parowanie dla tej pory roku).")


def _heat_sentence(obs: dict) -> str:
    """Wysokie zapotrzebowanie atmosfery — stres możliwy mimo wody w glebie."""
    import weather as weather_mod
    vpd, tmax = obs.get("vpd_max_kpa"), obs.get("tmax_c")
    hot = (vpd is not None and vpd >= weather_mod.VPD_HIGH_KPA) or (
        tmax is not None and tmax >= weather_mod.HEAT_TMAX_C)
    if not hot:
        return ""
    parts = []
    if vpd is not None:
        parts.append(f"VPD do {vpd:.1f} kPa")
    if tmax is not None:
        parts.append(f"Tmax {tmax:.0f} °C")
    return (f" Wysokie zapotrzebowanie atmosfery ({', '.join(parts)}) — przy takim upale "
            f"rośliny zamykają aparaty szparkowe nawet przy wilgotnej glebie.")


def _weather_sentence(obs: dict) -> str:
    """Zdanie kontekstu pogodowego do komunikatu — interpretacja spadku
    sygnału na tle bilansu opad−parowanie (nie zmienia flagi ryzyka)."""
    return _cwb14_sentence(obs) + _climate_sentence(obs) + _heat_sentence(obs)


def _cwb14_sentence(obs: dict) -> str:
    state = _cwb_state(obs)
    cwb = obs.get("cwb_14d_mm")
    if state == "deficit":
        return (f" Spadek zgodny z pogodą: bilans opad−parowanie z 14 dni "
                f"wynosi {cwb:+.0f} mm.")
    if state == "surplus":
        return (f" Bilans opad−parowanie z 14 dni jest dodatni ({cwb:+.0f} mm) — "
                f"spadek raczej nie wynika z braku wody z opadów; sprawdź inne "
                f"przyczyny (zbiór, choroba, nawodnienie, zasolenie).")
    if state == "neutral":
        return f" Bilans opad−parowanie z 14 dni: {cwb:+.0f} mm."
    return ""


def classify_water_signal(field: dict, observations: list, cohort_ndwi_values: list,
                           irrigation_events: list) -> dict:
    """Fuzja: własna historia (NDWI + VV) + kohorta (NDWI) + log nawodnienia
    + kontekst pogodowy dnia obserwacji. Zwraca FLAGĘ RYZYKA, nie diagnozę
    (patrz docstring modułu) — zawsze ze statusem "niekalibrowane"."""
    calibration_status = "niekalibrowane / wymaga walidacji terenowej"

    if not observations:
        return {"calibration_status": calibration_status, "risk_flag": False,
                "risk_type": None, "message_pl": "Brak wystarczających danych do oceny.",
                "components": {}}

    latest = observations[-1]
    own_ndwi = own_history_anomaly(observations, "ndwi_mean")
    own_resid = ndmi_residual_anomaly(observations)
    own_vv = own_vv_anomaly(observations)
    cohort_ndwi = cohort_anomaly(latest.get("ndwi_mean"), cohort_ndwi_values)
    irrigation_ctx = last_irrigation_context(irrigation_events, latest["obs_date"])

    raw_drop = bool(own_ndwi and own_ndwi["significant_drop"])
    # Warunek konieczny alarmu: istotny SPADEK surowego NDMI względem własnej
    # historii. NDMI rezydualny (względem NDVI) tylko ODSIEWA spadki wyjaśnione
    # spadkiem zieleni (dojrzewanie, zbiór, likwidacja poplonu) — sam nie
    # wywołuje alarmu. (Powtórka sezonu 2026: rezyduum jako samodzielny
    # wyzwalacz flagowało wschody poplonu, gdy NDMI rósł wolniej niż NDVI.)
    if own_resid is not None:
        # Wyjątek: przy deficycie atmosferycznym (P−ET0) silny spadek NDMI
        # flagujemy także, gdy NDVI spada razem z nim — tak wygląda
        # zaawansowany stres (więdnięcie), nie tylko dojrzewanie.
        ndmi_decline = raw_drop and (own_resid["significant_drop"]
                                     or _cwb_state(latest) == "deficit")
    else:
        ndmi_decline = raw_drop
    phenology_explained = bool(raw_drop and not ndmi_decline)

    # Gwałtowny spadek zieleni względem historii pola = zbiór, koszenie albo
    # likwidacja poplonu. Spadek NDMI jest wtedy skutkiem usunięcia łanu, nie
    # braku wody (powtórka sezonu 2026: zbiór kukurydzy flagowany jako stres,
    # bo nachylenie NDMI~NDVI jest przycięte z góry i rezyduum go nie odsiało).
    harvest_like = bool(raw_drop and own_resid is not None
                        and own_resid["delta_ndvi"] <= -HARVEST_NDVI_DROP)
    if harvest_like:
        ndmi_decline = False
        phenology_explained = True

    # Ocena wilgotności ŁANU ma sens tylko, gdy łan jest. Przy NDVI poniżej
    # MIN_CANOPY_NDVI (ściernisko, goła gleba, wschody) NDMI opisuje
    # wilgotność powierzchni gleby i resztek, nie stres roślin.
    ndvi_now = latest.get("ndvi_mean")
    canopy = ndvi_now is None or ndvi_now >= MIN_CANOPY_NDVI
    if not canopy:
        ndmi_decline = False
        phenology_explained = False

    # VV (Sentinel-1) tylko POTWIERDZA spadek optyczny — samodzielnie nie
    # wywołuje alarmu: na ścierniskach i po uprawkach VV skacze od zmian
    # szorstkości gleby, nie od wilgotności (powtórka sezonu 2026).
    vv_confirms = bool(ndmi_decline and own_vv and own_vv["used_for_decision"]
                       and own_vv["significant_drop"])
    own_decline = ndmi_decline
    # Równomierność: tylko przy łanie i nie w trakcie zbioru/likwidacji
    # (pole częściowo zebrane ma z natury duży rozrzut NDMI).
    uniformity = uniformity_anomaly(observations) if (canopy and not harvest_like) else None
    uneven = bool(uniformity and uniformity["flagged"])

    risk_type = None
    if own_decline:
        if (field.get("irrigation_mode") == "irrigation"
                and irrigation_ctx["days_since_last_event"] is not None
                and irrigation_ctx["days_since_last_event"] <= IRRIGATION_RECENT_DAYS):
            # Nawodnienie było niedawno, a sygnał mimo to spada — to bardziej
            # pytanie o skuteczność nawodnienia niż o sam fakt jego braku.
            risk_type = "irrigation_performance_anomaly"
        else:
            risk_type = "water_stress_risk"

    if not own_decline and uneven:
        risk_type = ("irrigation_uniformity_anomaly"
                     if field.get("irrigation_mode") == "irrigation" else "uneven_moisture_risk")
        pct = round(100 * uniformity["low_frac"])
        message = (f"Pole {field.get('parcel_id')}: średnia wilgotność w normie, ale ok. "
                   f"{pct}% powierzchni jest wyraźnie suchsze od reszty pola "
                   f"(od {latest['obs_date']}).")
        if field.get("irrigation_mode") == "irrigation":
            message += " Sprawdź równomierność nawodnienia (koniec bruzd, niedolane sektory)."
        else:
            message += " Sprawdź, czy to stała cecha gleby (piaszczyste płaty) czy nowe zjawisko."
        message += " (sygnał niekalibrowany — wymaga potwierdzenia w terenie)"
    elif not canopy:
        message = (f"Brak zwartej okrywy roślinnej (NDVI {ndvi_now:.2f}) — wilgotności łanu "
                   f"nie da się ocenić z satelity; o zapasie wody w glebie mówi bilans wodny.")
    elif not own_decline and harvest_like:
        message = (f"Brak sygnału stresu wodnego — gwałtowny spadek zieleni (NDVI "
                   f"{own_resid['delta_ndvi']:+.2f} względem historii pola) wygląda na zbiór, "
                   f"koszenie albo likwidację poplonu.")
    elif not own_decline and phenology_explained:
        message = ("Brak sygnału stresu wodnego — spadek NDMI jest proporcjonalny "
                   "do spadku zieleni (NDVI), co wskazuje raczej na dojrzewanie, "
                   "zbiór lub koszenie niż na brak wody.")
    elif not own_decline:
        message = ("Brak sygnału anomalii — obserwacja w granicach własnej "
                    "historii pola.")
    else:
        if field.get("irrigation_mode") == "irrigation":
            message = (f"Pole {field.get('parcel_id')} może wykazywać stres wodny "
                       f"od {latest['obs_date']} — warto sprawdzić nawodnienie.")
        else:
            message = (f"Pole {field.get('parcel_id')} wykazuje deficyt wilgoci — "
                       f"rozważ mulczowanie / poplon.")
        message += _weather_sentence(latest)
        if vv_confirms:
            message += " Spadek potwierdza też radar (Sentinel-1 VV)."
        if cohort_ndwi is None:
            message += " Brak wystarczającej liczby podobnych pól w regionie do porównania."
        elif cohort_ndwi["z_score"] <= COHORT_FIELD_SPECIFIC_Z:
            message += (" Sąsiednie pola w regionie nie wykazują tego samego spadku — "
                        "możliwa przyczyna specyficzna dla tego pola.")
        else:
            message += (" Podobny spadek widoczny też na sąsiednich polach w regionie — "
                        "może to być efekt pogodowy/regionalny, nie problem punktowy.")
        message += " (sygnał niekalibrowany — wymaga potwierdzenia w terenie)"

    return {
        "calibration_status": calibration_status,
        "risk_flag": own_decline or uneven,
        "risk_type": risk_type,
        "message_pl": message,
        "components": {
            "own_ndwi": own_ndwi,
            "own_ndwi_residual": own_resid,
            "own_vv": own_vv,
            "cohort_ndwi": cohort_ndwi,
            "uniformity": uniformity,
        },
        "phenology_explained": phenology_explained,
        "harvest_like": harvest_like,
        "canopy_present": canopy,
        "vv_confirms": vv_confirms,
        "weather_context": {
            "dry_spell_at_obs": bool(latest.get("dry_spell")),
            "cwb_14d_mm": latest.get("cwb_14d_mm"),
            "cwb_30d_mm": latest.get("cwb_30d_mm"),
            "atmospheric_deficit": _cwb_state(latest) == "deficit",
            "cwb_30d_pct": latest.get("cwb_30d_pct"),
            "vpd_max_kpa": latest.get("vpd_max_kpa"),
            "tmax_c": latest.get("tmax_c"),
            "soil_moisture_era5land": {k: latest.get(k) for k in ("sm_0_7", "sm_7_28", "sm_28_100")},
        },
        "irrigation_context": irrigation_ctx,
        "obs_date": latest["obs_date"],
    }


def select_water_scene(optical: list, min_valid_frac: float = None) -> Optional[dict]:
    """Najnowsza scena z wystarczającym udziałem czystych pikseli i z
    policzonym NDMI. None, gdy żadna scena w oknie nie spełnia progu —
    wtedy lepiej nie mieć obserwacji niż mieć obserwację zafałszowaną chmurą."""
    thr = WATER_MIN_VALID_FRAC if min_valid_frac is None else min_valid_frac
    ok = [r for r in optical
          if (r.get("valid_frac") or 0.0) >= thr and r.get("ndwi") is not None]
    if not ok:
        return None
    return max(ok, key=lambda r: r["date"])


def inner_buffer_utm(geom_4326, meters: float = 10.0):
    """Bufor wewnętrzny w lokalnej strefie UTM pola. core.inner_buffer liczy
    w EPSG:2180 (odwzorowanie dla Polski, południk osiowy 19°E) — dla pól
    w Uzbekistanie (~67°E) zniekształcenie jest duże (zmierzone: bufor 8.3 m
    zamiast 10 m, powierzchnia zawyżona o ~48%). Moduł wodny działa na obu
    kontynentach, więc liczy bufor w UTM. Za mała działka po buforze
    (< 400 m²) -> geometria bez bufora, jak w core.inner_buffer."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    from sh_utils import utm_epsg
    c = geom_4326.centroid
    epsg = utm_epsg(c.x, c.y)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    g_m = shp_transform(fwd, geom_4326)
    g_buf = g_m.buffer(-meters)
    if g_buf.is_empty or g_buf.area < 400:
        return geom_4326
    return shp_transform(inv, g_buf)


S1_TRACK_REPEAT_DAYS = 6   # konstelacja S1 (dwa satelity w fazie 180°): ta sama ścieżka co 6 dni


def infer_missing_orbits(radar_dates: list, orbits: dict) -> dict:
    """Uzupełnia orbity względne dla scen S1, których brakuje w katalogu STAC
    (zdarza się: Sentinel Hub ma scenę, STAC nie). Ta sama ścieżka nad
    punktem wraca co S1_TRACK_REPEAT_DAYS dni, więc scena oddalona o
    wielokrotność 6 dni od sceny o znanej orbicie ma tę samą orbitę —
    o ile wskazanie jest jednoznaczne."""
    out = dict(orbits)
    for d in radar_dates:
        if d in out:
            continue
        cands = {o for d0, o in orbits.items() if (d - d0).days % S1_TRACK_REPEAT_DAYS == 0}
        if len(cands) == 1:
            out[d] = cands.pop()
    return out


def fetch_sources(field: dict, t_start: date, t_end: date) -> dict:
    """Pobiera wszystkie źródła dla pola w oknie [t_start, t_end]: S2 (tylko
    z maską chmur), S1, orbity względne S1, pogodę (z zapasem na bilans
    P−ET0). Każde źródło poza S2 jest NIEBLOKUJĄCE. Wspólne dla bieżącego
    sprawdzenia (fetch_and_store_observation) i uzupełniania historii
    (water_replay.backfill_field) — jedna ścieżka danych, jeden wynik."""
    from analytics import fetch_multi_series_strict
    from sentinel1 import fetch_s1_series, fetch_s1_relative_orbits
    import weather as weather_mod

    geom = shapely_wkt.loads(field["geometry_wkt"])
    import water_geometry as wg
    try:
        wg.check_field_extent(geom, label=f"Pole {field.get('parcel_id')}")
    except ValueError as e:
        # Czytelny komunikat zamiast błędu 400 Sentinel Hub (obszar > 2500 px).
        raise RuntimeError(str(e) + " Usuń to pole z modułu i dodaj działki osobno.")
    geom_buf = inner_buffer_utm(geom, 10.0)

    # TYLKO zapytanie z maską chmur SCL (bez łagodniejszego fallbacku
    # fetch_multi_series, który przy braku czystych scen sięga po sceny bez
    # maski do 95% zachmurzenia) — patrz WATER_MIN_VALID_FRAC.
    optical = fetch_multi_series_strict(geom_buf, t_start, t_end, token=core.cdse_token()) or []
    if select_water_scene(optical) is None:
        # Bez żadnej czystej sceny nie ma czego budować — nie pytamy S1/pogody.
        return {"optical": optical, "radar": [], "orbits": {}, "weather": [], "dry_dates": set()}

    radar, orbits = [], {}
    try:
        radar = fetch_s1_series(geom_buf, t_start - timedelta(days=3), t_end)
    except Exception as e:
        log.warning("S1 niedostepne dla pola wodnego id=%s: %s", field.get("id"), e)
    if radar:
        try:
            orbits = fetch_s1_relative_orbits(geom_buf, radar[0]["date"], radar[-1]["date"]) or {}
        except Exception as e:
            log.warning("Orbity S1 nieznane dla pola wodnego id=%s: %s", field.get("id"), e)
        orbits = infer_missing_orbits([r["date"] for r in radar], orbits)

    weather_series, dry_dates = [], set()
    try:
        # Okno pogodowe dłuższe niż optyczne — bilans P-ET0 liczymy do 30 dni wstecz.
        w_start = t_start - timedelta(days=WEATHER_LOOKBACK_DAYS)
        weather_series = weather_mod.fetch_weather_series(geom_buf, w_start, t_end)
        dry_dates = weather_mod.detect_dry_spell_dates(weather_series)
    except Exception as e:
        log.warning("Pogoda niedostepna dla pola wodnego id=%s: %s", field.get("id"), e)

    climate = {}
    try:
        c = geom_buf.centroid
        climate = weather_mod.fetch_climatology(c.y, c.x)
    except Exception as e:
        log.warning("Norma wieloletnia niedostepna dla pola wodnego id=%s: %s", field.get("id"), e)

    return {"optical": optical, "radar": radar, "orbits": orbits,
            "weather": weather_series, "dry_dates": dry_dates, "climate": climate}


def build_observation(field: dict, opt: dict, sources: dict) -> dict:
    """Jedna obserwacja modułu wodnego dla sceny optycznej `opt`: dopasowanie
    S1 (±3 dni, z numerem orbity), pogoda dnia, bilans P−ET0, rozkład NDMI."""
    import weather as weather_mod
    d = opt["date"]
    radar = sources.get("radar") or []
    rad = None
    candidates = [r for r in radar if abs((r["date"] - d).days) <= 3]
    if candidates:
        rad = min(candidates, key=lambda r: abs((r["date"] - d).days))
    weather_series = sources.get("weather") or []
    day_rec = next((w for w in weather_series if w.get("date") == d), {})
    return {
        "water_field_id": field["id"],
        "obs_date": d.isoformat(),
        "ndwi_mean": opt.get("ndwi"),
        "ndvi_mean": opt.get("ndvi"),
        "vv_db": rad["vv_db"] if rad else None,
        "vh_db": rad["vh_db"] if rad else None,
        "rvi": rad["rvi"] if rad else None,
        "valid_frac": opt.get("valid_frac"),
        "precip_mm": day_rec.get("precip_mm"),
        "dry_spell": d in (sources.get("dry_dates") or set()),
        "et0_mm": day_rec.get("et0_mm"),
        "cwb_14d_mm": weather_mod.climatic_water_balance(weather_series, d, 14),
        "cwb_30d_mm": (cwb30 := weather_mod.climatic_water_balance(weather_series, d, 30)),
        "cwb_30d_pct": weather_mod.cwb_percentile(cwb30, d, 30, sources.get("climate") or {}),
        "vpd_max_kpa": day_rec.get("vpd_max_kpa"),
        "tmax_c": day_rec.get("tmax_c"),
        "sm_0_7": day_rec.get("sm_0_7"),
        "sm_7_28": day_rec.get("sm_7_28"),
        "sm_28_100": day_rec.get("sm_28_100"),
        "ndwi_p10": opt.get("ndwi_p10"),
        "ndwi_p50": opt.get("ndwi_p50"),
        "ndwi_low_frac": opt.get("ndwi_low_frac"),
        "s1_rel_orbit": (sources.get("orbits") or {}).get(rad["date"]) if rad else None,
        "s1_date": rad["date"].isoformat() if rad else None,
    }


def fetch_and_store_observation(field: dict, lookback_days: int = LOOKBACK_DAYS) -> Optional[dict]:
    """Pobiera świeże dane S2/S1/pogoda dla jednego pola wodnego, zapisuje
    JEDNĄ obserwację (najnowszy dostępny czysty dzień optyczny w oknie) do
    water_observations i zwraca ją. None, jeśli w oknie nie ma żadnej
    czystej sceny — analogicznie do core.check_parcel, brak danych nie jest
    błędem, tylko brakiem wyniku dla tego przebiegu."""
    t_end = date.today()
    t_start = t_end - timedelta(days=lookback_days)
    sources = fetch_sources(field, t_start, t_end)
    if not sources["optical"]:
        log.warning("Brak danych optycznych dla pola wodnego id=%s w oknie %s..%s",
                    field.get("id"), t_start, t_end)
        return None
    latest_opt = select_water_scene(sources["optical"])
    if latest_opt is None:
        log.warning("Pole wodne id=%s: brak sceny z >= %.0f%% czystych pikseli w oknie %s..%s",
                    field.get("id"), WATER_MIN_VALID_FRAC * 100, t_start, t_end)
        return None
    # Zapisujemy WSZYSTKIE czyste sceny z okna, których jeszcze nie ma w
    # historii (audyt 11.2026: wcześniej tylko najnowszą — pozostałe, już
    # opłacone w PU, przepadały, a historia do z-score była rzadsza).
    existing = {o["obs_date"] for o in db.get_water_observations(field["id"])}
    clean = sorted((r for r in sources["optical"]
                    if (r.get("valid_frac") or 0.0) >= WATER_MIN_VALID_FRAC and r.get("ndwi") is not None),
                   key=lambda r: r["date"])
    obs = None
    for opt in clean:
        o = build_observation(field, opt, sources)
        if o["obs_date"] not in existing:
            db.add_water_observation(**o)
        if opt is latest_opt:
            obs = o
    db.update_water_field_last_checked(field["id"])
    return obs


def build_verdict_for_field(field: dict) -> dict:
    """Buduje werdykt silnika dla pola NA PODSTAWIE JUŻ ZAPISANEJ historii
    (bez sięgania po świeże dane) — używane zarówno przez check_field_now
    (po fetch_and_store_observation), jak i przez scheduler (Krok 3), żeby
    logika fuzji istniała w jednym miejscu."""
    observations = db.get_water_observations(field["id"])
    if not observations:
        return {"calibration_status": "niekalibrowane / wymaga walidacji terenowej",
                "risk_flag": False, "risk_type": None,
                "message_pl": "Brak wystarczających danych do oceny.", "components": {}}

    cohort = get_cohort_fields(field)
    cohort_ndwi_values = cohort_values_near_date(cohort, observations[-1]["obs_date"])

    irrigation_events = db.get_irrigation_events(field["id"])
    return classify_water_signal(field, observations, cohort_ndwi_values, irrigation_events)


def check_field_now(field: dict) -> dict:
    """Orkiestracja pojedynczego sprawdzenia 'na żądanie' (analogicznie do
    monitor_check_now w app.py): pobierz+zapisz świeżą obserwację, zbuduj
    werdykt z pełnej (odświeżonej) historii. Kosztowne (odpytuje Sentinel
    Hub) — wołający ma to chronić auth-em. NIE zapisuje alarmu do kolejki
    triage — to robi scheduler (record_alarm_if_risky), żeby samo
    sprawdzenie podglądowe operatora nie zapychało kolejki."""
    fresh = fetch_and_store_observation(field)
    refresh_optram(field)
    refresh_thermal(field)
    verdict = build_verdict_for_field(field)
    verdict["fresh_observation_fetched"] = fresh is not None
    attach_water_balance(field, verdict)
    attach_optram(field, verdict)
    attach_thermal(field, verdict)
    return verdict


def refresh_thermal(field: dict) -> int:
    """Dociąga świeże sceny Landsat, jeśli pole ma już termikę (nieblokujące)."""
    try:
        import thermal
        return thermal.update_recent(field)
    except Exception as e:
        log.warning("Termika: odświeżenie pola id=%s nieudane: %s", field.get("id"), e)
        return 0


def attach_thermal(field: dict, verdict: dict) -> dict:
    import thermal
    return thermal.attach_thermal(field, verdict)


def refresh_optram(field: dict) -> int:
    """Dopisuje świeże W z OPTRAM, jeśli pole ma kalibrację (nieblokujące)."""
    try:
        import optram
        return optram.update_recent(field)
    except Exception as e:
        log.warning("OPTRAM: odświeżenie pola id=%s nieudane: %s", field.get("id"), e)
        return 0


def attach_optram(field: dict, verdict: dict) -> dict:
    import optram
    return optram.attach_optram(field, verdict)


def attach_water_balance(field: dict, verdict: dict) -> dict:
    """Dokleja do werdyktu podsumowanie bilansu wodnego (water_balance.py)
    i zdanie łączące oba źródła: satelita vs model. NIEBLOKUJĄCE — brak
    pogody nie psuje werdyktu satelitarnego. Nie zmienia risk_flag."""
    try:
        import water_balance
        bal = water_balance.compute_field_balance(field)["summary"]
    except Exception as e:
        log.warning("Bilans wodny niedostępny dla pola id=%s: %s", field.get("id"), e)
        return verdict
    verdict["water_balance"] = bal
    st = bal.get("status")
    try:
        import water_overview
        trust = water_overview.balance_trust(field)
    except Exception:
        trust = {"trusted": True, "reason": None}
    if not trust["trusted"]:
        verdict["balance_untrusted"] = trust["reason"]
        verdict["message_pl"] += (" Bilans wodny pomijamy w ocenie: " + (
            "w dzienniku nie ma nawodnień z ostatnich 3 miesięcy, więc model zakłada, że pole nie było podlewane."
            if trust["reason"] == "no_irrigation_log" else
            "satelita (wilgotność wierzchnia albo temperatura łanu) pokazuje wodę, której model nie zna."))
        return verdict
    if verdict.get("risk_flag") and verdict.get("risk_type") in (
            "water_stress_risk", "irrigation_performance_anomaly") and st == "ok":
        verdict["message_pl"] += (" Model bilansu wodnego nie wskazuje deficytu w glebie — "
                                  "rozważ przyczyny inne niż brak wody.")
    elif not verdict.get("risk_flag") and st == "stress":
        verdict["message_pl"] += (" Uwaga: model bilansu wodnego wskazuje już stres — "
                                  "satelita może go jeszcze nie pokazywać.")
    elif verdict.get("risk_flag") and st == "stress":
        verdict["message_pl"] += " Model bilansu wodnego potwierdza deficyt wody w glebie."
    return verdict


def balance_alarm_decision(field: dict, verdict: dict) -> Optional[dict]:
    """Czy model bilansu wodnego (verdict["water_balance"], patrz
    attach_water_balance) uzasadnia osobny alarm w kolejce triage.
    Tylko gdy satelita NIE zgłosił ryzyka (wtedy jego alarm już zawiera
    zdanie o bilansie) i model mówi: stres trwa albo przyjdzie w ciągu
    BALANCE_ALERT_DAYS dni. Zwraca {risk_type, message_pl} albo None."""
    bal = verdict.get("water_balance") or {}
    if verdict.get("risk_flag") or not bal:
        return None
    if field.get("irrigation_mode") != "irrigation":
        # Na polu deszczowym nie ma działania do podjęcia (nie da się
        # podlać) — bilans zostaje kontekstem werdyktu, bez alarmu w kolejce.
        return None
    dts = bal.get("days_to_stress")
    fc = bal.get("forecast") or {}
    hot_soon = bool(fc.get("dry_hot")) and bal.get("status") == "approaching"
    if bal.get("status") != "stress" and (dts is None or dts > BALANCE_ALERT_DAYS) and not hot_soon:
        return None
    pid = field.get("parcel_id")
    if bal.get("status") == "stress":
        head = f"Pole {pid}: model bilansu wodnego wskazuje stres wodny (zużyto {bal.get('depletion_mm', 0):.0f} mm, próg {bal.get('raw_mm', 0):.0f} mm)."
    elif hot_soon and (dts is None or dts > BALANCE_ALERT_DAYS):
        head = (f"Pole {pid}: zapas łatwo dostępnej wody prawie wyczerpany "
                f"(zużyto {bal.get('depletion_mm', 0):.0f} z {bal.get('raw_mm', 0):.0f} mm), "
                f"a prognoza zapowiada suchy i gorący tydzień"
                + (f" (Tmax do {fc['tmax_c']:.0f} °C)" if fc.get("tmax_c") is not None else "") + ".")
    else:
        when = "jutro" if dts == 1 else f"za {dts} dni"
        if dts == 0:
            when = "dziś"
        head = (f"Pole {pid}: według modelu bilansu wodnego próg stresu zostanie "
                f"przekroczony {when} ({bal.get('suggested_irrigation_date')}).")
    if field.get("irrigation_mode") == "irrigation" and bal.get("suggested_dose_mm") is not None:
        head += (f" Sugerowana dawka ok. {bal['suggested_dose_mm']:.0f} mm "
                 f"({bal['suggested_dose_m3_ha']:.0f} m³/ha).")
    head += " Satelita nie pokazuje jeszcze spadku. (sygnał niekalibrowany — wymaga potwierdzenia w terenie)"
    return {"risk_type": BALANCE_RISK_TYPE, "message_pl": head}


def record_balance_alarm_if_needed(field: dict, verdict: dict) -> Optional[int]:
    decision = balance_alarm_decision(field, verdict)
    if decision is None:
        return None
    import water_overview
    trust = water_overview.balance_trust(field)
    if not trust["trusted"]:
        # Pole nawadniane bez dziennika albo satelita widzi wodę — bilans nie
        # wie o nawodnieniach, więc jego „stres” nie trafia do kolejki doradcy.
        log.info("Pole id=%s: alarm bilansowy pominięty (%s)", field.get("id"), trust["reason"])
        return None
    if db.water_alarm_exists_recent(field["id"], decision["risk_type"], days=WATER_ALARM_RECENT_DAYS):
        return None
    return db.add_water_alarm(
        field["id"], verdict.get("obs_date") or date.today().isoformat(),
        decision["risk_type"], decision["message_pl"],
        "niekalibrowane / wymaga walidacji terenowej",
        components_json=json.dumps({"water_balance": verdict.get("water_balance")}, default=str))


def record_alarm_if_risky(field: dict, verdict: dict) -> Optional[int]:
    """Zapisuje werdykt do kolejki triage (water_alarms), jeśli
    risk_flag=True i nie ma już świeżego alarmu tego samego typu dla tego
    pola (db.water_alarm_exists_recent) — chroni kolejkę przed zalaniem tym
    samym sygnałem co 48h przy kolejnych przebiegach schedulera. Zwraca
    id nowego alarmu albo None (brak ryzyka / już zgłoszone niedawno)."""
    if not verdict.get("risk_flag"):
        return None
    risk_type = verdict["risk_type"]
    if db.water_alarm_exists_recent(field["id"], risk_type, days=WATER_ALARM_RECENT_DAYS):
        return None
    components = verdict.get("components", {})
    return db.add_water_alarm(
        field["id"], verdict["obs_date"], risk_type, verdict["message_pl"],
        verdict["calibration_status"],
        ndwi_value=(components.get("own_ndwi") or {}).get("current"),
        vv_value=(components.get("own_vv") or {}).get("current"),
        components_json=json.dumps(components, default=str),
    )


# ---------------------------------------------------------------------------
# Petla zwrotna / rekalibracja (ostatni element Etapu 1-3 planu modulu
# wodnego) -- analogicznie do calibration.compute_suggested_thresholds() w
# module poplonow, ale zamiast zdjec terenowych porownuje z-score, ktory
# WYZWOLIL alarm, miedzy alarmami POTWIERDZONYMI (prawdziwy sygnal) i
# ODRZUCONYMI (falszywy alarm) w kolejce triage (water_alarms.triage_status).
# needs_visit/pending sa pomijane -- jeszcze nie ma rozstrzygniecia.
#
# WAZNE OGRANICZENIE (w odroznieniu od kalibracji poplonow): mamy tu proby
# WYLACZNIE z populacji juz OZNACZONEJ jako ryzyko (alarm powstaje tylko gdy
# risk_flag=True) -- nie mamy z-score dla obserwacji, ktore NIE zostaly
# oflagowane. To znaczy, ze ta kalibracja moze tylko ZAOSTRZYC prog (zmniejszyc
# liczbe falszywych alarmow wsrod juz zglaszanych), nie moze ocenic, czy prog
# jest za surowy i przepuszcza realne przypadki stresu ponizej niego --
# to wymagaloby osobnego mechanizmu (np. losowej probki niealarmowych
# obserwacji do przegladu), celowo NIE zrobionego w tym kroku.
# ---------------------------------------------------------------------------

def _trigger_z_from_alarm(alarm: dict) -> Optional[float]:
    """Wyciaga z-score, ktory faktycznie wyzwolil dany alarm (najbardziej
    ujemny sposrod own_ndwi/own_vv zapisanych w components_json przy tworzeniu
    alarmu -- patrz record_alarm_if_risky) -- to jest zmienna, ktora
    kalibrujemy, nie surowe wartosci NDWI/VV."""
    raw = alarm.get("components_json")
    if not raw:
        return None
    try:
        components = json.loads(raw)
    except (ValueError, TypeError):
        return None
    zs = [components[k]["z_score"] for k in ("own_ndwi", "own_ndwi_residual", "own_vv")
          if components.get(k) and components[k].get("z_score") is not None
          and components[k].get("used_for_decision", True)]
    return min(zs) if zs else None


def compute_suggested_water_threshold(min_samples: int = 10) -> Optional[dict]:
    """Sugeruje nowy Z_SCORE_RISK_THRESHOLD na podstawie decyzji doradcy w
    kolejce triage. Zwraca None, jesli lacznie potwierdzonych+odrzuconych
    alarmow z odczytywalnym z-score jest mniej niz min_samples."""
    # Kalibrujemy próg z-score spadku sygnału — alarmy równomierności mają
    # inną zmienną wyzwalającą i nie wchodzą do tej próbki.
    alarms = [a for a in db.get_water_alarms() if a.get("risk_type") not in NON_DECLINE_RISK_TYPES]
    confirmed_z = sorted(
        z for z in (_trigger_z_from_alarm(a) for a in alarms if a["triage_status"] == "confirmed")
        if z is not None)
    rejected_z = sorted(
        z for z in (_trigger_z_from_alarm(a) for a in alarms if a["triage_status"] == "rejected")
        if z is not None)
    total = len(confirmed_z) + len(rejected_z)
    if total < min_samples:
        return None

    def percentile(lst, pct):
        if not lst:
            return None
        idx = max(0, min(int(len(lst) * pct / 100), len(lst) - 1))
        return lst[idx]

    current = Z_SCORE_RISK_THRESHOLD
    # confirmed_z posortowane rosnaco (najbardziej ujemne = najsilniejszy
    # sygnal, pierwsze). Percentyl 75 to "najslabszy" koniec potwierdzonych --
    # granica, ktora nadal lapie ~75% z nich (mirror poplonowego
    # percentile(veg_ndvi, 25), tylko odwrocony kierunek skali).
    from_confirmed = percentile(confirmed_z, 75)
    # rejected_z posortowane rosnaco (najbardziej ujemne = najbardziej
    # "przekonujacy" falszywy alarm, pierwsze). Percentyl 10 lapie ten
    # skrajny koniec odrzuconych -- margines 0.05 W GLAB (bardziej ujemnie),
    # zeby prog byl surowszy niz najsilniejszy zaobserwowany falszywy alarm,
    # a nie tylko rowny mu.
    from_rejected = percentile(rejected_z, 10)
    if from_rejected is not None:
        from_rejected = round(from_rejected - 0.05, 3)

    candidates = [v for v in (from_confirmed, from_rejected) if v is not None]
    if not candidates:
        return None

    # Priorytet: nie tracic potwierdzonych przypadkow. Przy praktycznie
    # zerowej ilosci danych na starcie falszywy alarm kosztuje doradce chwile
    # w kolejce triage, przeoczony realny stres wodny kosztuje wiecej -- wiec
    # bierzemy MNIEJ rygorystyczna (blizsza zeru) z dwoch propozycji.
    new_threshold = max(candidates)
    new_threshold = max(-4.0, min(-0.5, round(new_threshold, 2)))

    return {
        "suggested_threshold": new_threshold,
        "current_threshold": current,
        "confirmed_samples": len(confirmed_z),
        "rejected_samples": len(rejected_z),
        "total_samples": total,
        "change_magnitude": round(abs(new_threshold - current), 3),
    }


def apply_water_threshold(new_threshold: float) -> bool:
    """Zapisuje nowy prog w tabeli `settings` i przeladowuje go w pamieci
    procesu (reload_water_threshold) -- operator zatwierdza recznie, tak jak
    przy progach NDVI w module poplonow (calibration.apply_thresholds)."""
    try:
        old = Z_SCORE_RISK_THRESHOLD
        db.set_setting("water_zscore_risk_threshold", f"{new_threshold:.2f}")
        reload_water_threshold()
        log.info(f"Z_SCORE_RISK_THRESHOLD zaktualizowany: {old} -> {new_threshold}")
        return True
    except Exception as e:
        log.error(f"Blad aktualizacji progu wodnego: {e}")
        return False


def get_water_calibration_status() -> dict:
    alarms = db.get_water_alarms()
    by_status: dict = {}
    for a in alarms:
        by_status[a["triage_status"]] = by_status.get(a["triage_status"], 0) + 1
    return {
        "current_z_score_threshold": Z_SCORE_RISK_THRESHOLD,
        "default_z_score_threshold": DEFAULT_Z_SCORE_RISK_THRESHOLD,
        "total_alarms": len(alarms),
        "by_triage_status": by_status,
        "min_samples_required": 10,
    }
