"""Bilans wodny strefy korzeniowej (FAO-56, uproszczony) dla pól modułu wodnego.

Po co: silnik anomalii (water_stress.py) mówi "sygnał satelitarny spadł".
Bilans mówi "ile wody zostało w glebie i kiedy roślina wejdzie w stres" —
z pogody (opad, ET0), zieleni łanu (NDVI z obserwacji pola) i logu
nawodnień. Satelita staje się wtedy niezależną weryfikacją modelu:
  - model: deficyt  + satelita: spadek   -> stres wodny potwierdzony z dwóch stron,
  - model: OK       + satelita: spadek   -> raczej inna przyczyna (choroba,
                                            zasolenie, nieskuteczne nawodnienie),
  - model: deficyt  + satelita: bez zmian -> stres się zbliża, satelita go
                                            jeszcze nie widzi.

Model (Allen et al. 1998, FAO-56, rozdz. 8 — wersja dzienna, jedna warstwa):
    Kcb  = Kcb_min + (Kcb_max − Kcb_min) · fc,   fc = (NDVI − NDVI_min)/(NDVI_max − NDVI_min)
    Ks   = 1, gdy D <= RAW;  (TAW − D) / ((1 − p) · TAW), gdy D > RAW
    ETc  = (Ks · Kcb + KE_AVG · (1 − fc)) · ET0
    D_t  = max(0, D_{t−1} − P_eff − I + ETc)        (nadwyżka = przesiąk/spływ)
    RAW  = p · TAW
gdzie D = deficyt (zużycie) wody w strefie korzeniowej [mm], TAW = całkowita
woda dostępna [mm], p = frakcja łatwo dostępna (zależna od uprawy).

ZAŁOŻENIA (zwracane jawnie w wyniku, żeby doradca widział, na czym stoi liczba):
  - Start: początek sezonu (1 marca) — wtedy profil jest najbliżej pełnego
    (po zimie w Polsce, przed pierwszym nawodnieniem w Uzbekistanie).
    Stan początkowy z wilgotności ERA5-Land 0–100 cm na ten dzień
    (przeliczonej na % wody dostępnej, patrz weather.soil_limits); gdy
    archiwum niedostępne — dawny tryb: 92 dni wstecz i profil pełny.
  - Punkt kontrolny (ocena wilgotności w polu, np. metodą „gleba w dłoni”
    FAO/USDA) ustawia stan profilu w danym dniu — jak nawodnienie do pełna.
  - Odcinek „szacunkowy”: dni, w których wynik jeszcze zależy od stanu
    początkowego. Liczymy go wprost: bilans z profilem pełnym i pustym na
    starcie zbiega się dopiero po opadzie/nawodnieniu, które w pełni
    uzupełnia profil (albo po punkcie kontrolnym).
  - Nawodnienie bez podanej dawki w mm traktujemy jako uzupełnienie do
    pełnej pojemności (D = 0) — typowe dla nawadniania zalewowego/bruzdowego.
  - TAW domyślnie 100 mm, gdy nie podano (gleba średnia, ~0.7 m korzeni).
  - Opady < P_MIN_EFFECTIVE_MM/d nie uzupełniają strefy korzeniowej
    (intercepcja, parowanie z powierzchni).
  - Parametry upraw: typowe wartości z tabel FAO-56 (Kcb_mid, p) — do
    doprecyzowania lokalnie.

To NARZĘDZIE DORADCZE: wynik to sugestia terminu i dawki, nie automatyczny
wpis do kalendarza — plan dodaje człowiek (patrz irrigation_plans).
"""
import logging
from datetime import date, timedelta
from typing import Optional

log = logging.getLogger("water_balance")

NDVI_MIN = 0.15          # gleba goła
NDVI_MAX = 0.85          # pełne zwarcie łanu
KCB_MIN = 0.15           # Kcb gleby gołej / początku sezonu (FAO-56)
KE_AVG = 0.20            # średni współczynnik parowania z gleby nieokrytej (roboczo)
P_MIN_EFFECTIVE_MM = 2.0 # opady poniżej tej dobowej sumy nie zasilają strefy korzeniowej
DEFAULT_TAW_MM = 100.0   # tylko gdy soil_water nic nie ustali (patrz soil_water.field_taw)
DEFAULT_KCB_MAX = 1.10
DEFAULT_P = 0.50
BALANCE_PAST_DAYS = 92   # tryb awaryjny: maks. historia z API prognozy Open-Meteo (past_days)
FORECAST_DAYS = 10
SEASON_START = (3, 1)    # (miesiąc, dzień) startu bilansu
SEASON_MIN_DAYS = 30     # start co najmniej tyle dni przed dziś (inaczej — rok wcześniej)
NDVI_HOLD_BACK_DAYS = 30 # przed pierwszą obserwacją trzymamy jej NDVI najwyżej tyle dni
ANCHOR_TOL_MM = 0.5

# Ocena wilgotności „w dłoni” (FAO / USDA NRCS „Estimating Soil Moisture by
# Feel and Appearance”) — środek przedziału % wody dostępnej.
SOIL_FEEL_LEVELS = {"wet": 100.0, "moist": 87.0, "medium": 62.0, "dry": 37.0, "very_dry": 12.0}

# (Kcb_mid, p) — FAO-56 tab. 17 i 22, wartości typowe. Klucze: nazwy PL/EN/RU-translit.
CROP_PARAMS = {
    "pszenica": (1.10, 0.55), "wheat": (1.10, 0.55), "bugdoy": (1.10, 0.55),
    "jęczmień": (1.10, 0.55), "jeczmien": (1.10, 0.55), "barley": (1.10, 0.55),
    "kukurydza": (1.15, 0.55), "maize": (1.15, 0.55), "corn": (1.15, 0.55),
    "bawełna": (1.15, 0.65), "bawelna": (1.15, 0.65), "cotton": (1.15, 0.65), "paxta": (1.15, 0.65),
    "burak": (1.15, 0.55), "burak cukrowy": (1.15, 0.55), "sugar beet": (1.15, 0.55),
    "ziemniak": (1.10, 0.35), "ziemniaki": (1.10, 0.35), "potato": (1.10, 0.35),
    "rzepak": (1.05, 0.60), "rapeseed": (1.05, 0.60),
    "soja": (1.10, 0.50), "soybean": (1.10, 0.50),
    "ryż": (1.15, 0.20), "ryz": (1.15, 0.20), "rice": (1.15, 0.20),
    "lucerna": (1.15, 0.55), "alfalfa": (1.15, 0.55),
    "warzywa": (1.00, 0.40), "vegetables": (1.00, 0.40),
}


def crop_params(crop_type) -> tuple:
    """(Kcb_max, p, czy_znana_uprawa)."""
    key = (str(crop_type).strip().lower() if crop_type else "")
    if key in CROP_PARAMS:
        kcb, p = CROP_PARAMS[key]
        return kcb, p, True
    return DEFAULT_KCB_MAX, DEFAULT_P, False


def cover_fraction(ndvi: Optional[float]) -> float:
    if ndvi is None:
        return 0.0
    return min(1.0, max(0.0, (ndvi - NDVI_MIN) / (NDVI_MAX - NDVI_MIN)))


def kcb_from_ndvi(ndvi: Optional[float], kcb_max: float) -> float:
    return KCB_MIN + (kcb_max - KCB_MIN) * cover_fraction(ndvi)


def interpolate_ndvi(points: list, d: date, hold_back_days: Optional[int] = None) -> Optional[float]:
    """Liniowa interpolacja NDVI między obserwacjami; poza zakresem —
    najbliższa wartość (trzymamy ostatnią znaną zieleń w prognozie).
    `hold_back_days`: przed pierwszą obserwacją trzymamy jej wartość najwyżej
    tyle dni — wcześniej None (brak wiedzy o okrywie = gleba goła)."""
    pts = sorted((p for p in points if p[1] is not None), key=lambda p: p[0])
    if not pts:
        return None
    if d <= pts[0][0]:
        if hold_back_days is not None and (pts[0][0] - d).days > hold_back_days:
            return None
        return pts[0][1]
    if d >= pts[-1][0]:
        return pts[-1][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= d <= d1:
            span = (d1 - d0).days or 1
            return v0 + (v1 - v0) * (d - d0).days / span
    return pts[-1][1]


def run_balance(weather: list, ndvi_points: list, irrigation_events: list,
                taw_mm: float, p: float, kcb_max: float,
                start_date: Optional[date] = None, initial_depletion: float = 0.0,
                soil_checks: Optional[list] = None,
                ndvi_hold_back_days: Optional[int] = None, ks_salt: float = 1.0) -> list:
    """Dzienny bilans od start_date (domyślnie pierwszy dzień pogody).
    `weather`: [{"date", "precip_mm", "et0_mm", "is_forecast"}] posortowane.
    `irrigation_events`: [{"event_date": iso, "amount_mm": float|None}].
    `initial_depletion`: zużycie wody [mm] na starcie (0 = profil pełny).
    `soil_checks`: [{"check_date": iso, "fill_pct": 0–100}] — ocena w polu
    ustawia stan profilu na koniec danego dnia.
    `ks_salt`: redukcja pobierania wody przez zasolenie (FAO-56 r. 89/90,
    soil_water.field_salinity) — mnożona przez Ks od niedoboru wody."""
    raw_mm = p * taw_mm
    irr_by_date: dict = {}
    for e in irrigation_events:
        d = date.fromisoformat(e["event_date"])
        irr_by_date.setdefault(d, []).append(e.get("amount_mm"))
    check_by_date: dict = {}
    for c in soil_checks or []:
        check_by_date.setdefault(date.fromisoformat(c["check_date"]), []).append(float(c["fill_pct"]))

    depletion = min(taw_mm, max(0.0, float(initial_depletion)))
    out = []
    for w in sorted(weather, key=lambda r: r["date"]):
        d = w["date"]
        if start_date and d < start_date:
            continue
        et0 = w.get("et0_mm")
        precip = w.get("precip_mm") or 0.0
        if et0 is None:
            continue
        ndvi = interpolate_ndvi(ndvi_points, d, ndvi_hold_back_days)
        fc = cover_fraction(ndvi)
        kcb = kcb_from_ndvi(ndvi, kcb_max)
        p_eff = precip if precip >= P_MIN_EFFECTIVE_MM else 0.0

        irrigation_mm = 0.0
        refill = False
        for amt in irr_by_date.get(d, []):
            if amt is None:
                refill = True
            else:
                irrigation_mm += amt

        ks = (1.0 if depletion <= raw_mm else max(
            0.0, (taw_mm - depletion) / ((1.0 - p) * taw_mm))) * ks_salt
        etc = (ks * kcb + KE_AVG * (1.0 - fc)) * et0
        depletion = depletion - p_eff - irrigation_mm + etc
        if refill:
            depletion = 0.0
        check = None
        if d in check_by_date:
            check = sum(check_by_date[d]) / len(check_by_date[d])
            depletion = (1.0 - min(100.0, max(0.0, check)) / 100.0) * taw_mm
        depletion = min(taw_mm, max(0.0, depletion))
        out.append({
            "date": d, "is_forecast": bool(w.get("is_forecast")),
            "ndvi": None if ndvi is None else round(ndvi, 3),
            "kcb": round(kcb, 3), "ks": round(ks, 3),
            "et0_mm": round(et0, 2), "etc_mm": round(etc, 2),
            "p_eff_mm": round(p_eff, 1), "irrigation_mm": round(irrigation_mm, 1),
            "irrigation_refill": refill,
            "soil_check_pct": None if check is None else round(check, 0),
            "depletion_mm": round(depletion, 1),
            "stress": depletion > raw_mm,
        })
    return out


def summarize_balance(days: list, taw_mm: float, p: float, today: date) -> dict:
    raw_mm = p * taw_mm
    past = [r for r in days if r["date"] <= today and not r["is_forecast"]]
    future = [r for r in days if r["date"] > today or r["is_forecast"]]
    if not past:
        return {"status": "no_data"}
    cur = past[-1]
    status = "stress" if cur["depletion_mm"] > raw_mm else (
        "approaching" if cur["depletion_mm"] > 0.75 * raw_mm else "ok")

    stress_day = cur["date"] if status == "stress" else next(
        (r["date"] for r in future if r["stress"]), None)
    days_to_stress = None if stress_day is None else max(0, (stress_day - today).days)
    dose = None
    if stress_day is not None:
        rec = next((r for r in days if r["date"] == stress_day), cur)
        dose = rec["depletion_mm"]

    wetting = sum(r["p_eff_mm"] + r["irrigation_mm"] for r in past) + (
        taw_mm if any(r["irrigation_refill"] for r in past) else 0.0)
    return {
        "status": status,
        "as_of": cur["date"].isoformat(),
        "depletion_mm": cur["depletion_mm"],
        "depletion_frac": round(cur["depletion_mm"] / taw_mm, 3),
        "raw_mm": round(raw_mm, 1),
        "taw_mm": round(taw_mm, 1),
        "ks": cur["ks"],
        "days_to_stress": days_to_stress,
        "forecast_horizon_days": len(future),
        "suggested_irrigation_date": stress_day.isoformat() if stress_day else None,
        "suggested_dose_mm": None if dose is None else round(dose, 0),
        "suggested_dose_m3_ha": None if dose is None else round(dose * 10, -1),
        # Pewność stanu początkowego: czy w oknie był opad/nawodnienie
        # porównywalne z RAW (wtedy założenie D=0 na starcie przestaje mieć znaczenie).
        "initial_state_reliable": wetting >= raw_mm,
    }


def describe_balance(summary: dict, irrigation_mode: str = "irrigation") -> str:
    """Krótki komunikat po polsku dla doradcy. Dla pól deszczowych bez
    sugestii nawadniania — tylko przewidywany termin wejścia w stres."""
    st = summary.get("status")
    if st == "no_data":
        return "Brak danych do policzenia bilansu wodnego."
    d, raw = summary["depletion_mm"], summary["raw_mm"]
    if st == "stress":
        msg = (f"Bilans wodny: zużyto {d:.0f} mm, powyżej progu łatwo dostępnej wody "
               f"({raw:.0f} mm) — roślina jest w stresie wodnym.")
    elif st == "approaching":
        msg = (f"Bilans wodny: zużyto {d:.0f} mm z {raw:.0f} mm łatwo dostępnej wody "
               f"— zbliża się próg stresu.")
    else:
        msg = f"Bilans wodny: zużyto {d:.0f} mm z {raw:.0f} mm łatwo dostępnej wody."
    if summary.get("suggested_irrigation_date") and irrigation_mode != "irrigation":
        msg += f" Przewidywany próg stresu: {summary['suggested_irrigation_date']}."
    elif summary.get("suggested_irrigation_date"):
        msg += (f" Sugerowane nawodnienie: {summary['suggested_irrigation_date']}, "
                f"ok. {summary['suggested_dose_mm']:.0f} mm "
                f"({summary['suggested_dose_m3_ha']:.0f} m³/ha).")
    elif summary.get("forecast_horizon_days"):
        msg += f" W prognozie na {summary['forecast_horizon_days']} dni próg stresu nie zostanie przekroczony."
    fc = summary.get("forecast")
    if fc:
        msg += (f" Prognoza {fc['days']} dni: opad {fc['precip_mm']:.0f} mm, "
                f"parowanie {fc['et0_mm']:.0f} mm")
        if fc.get("tmax_c") is not None:
            msg += f", Tmax do {fc['tmax_c']:.0f} °C"
        msg += "."
        if fc.get("dry_hot"):
            msg += " Zapowiada się sucho i gorąco — zużycie wody wzrośnie."
    if summary.get("cwb_30d_class"):
        msg += (f" Ostatnie 30 dni na tle lat 1991–2020: {summary['cwb_30d_class']} "
                f"({summary['cwb_30d_pct']:.0f}. percentyl).")
    if not summary.get("initial_state_reliable"):
        if (summary.get("initial_state") or {}).get("source") == "era5":
            msg += (" (Szacunek: od startu bilansu nie było opadu ani nawodnienia, które w pełni uzupełniłoby "
                    "profil — wynik zależy od stanu początkowego z modelu ERA5-Land. Ocena wilgotności "
                    "w polu (punkt kontrolny) go urealni.)")
        else:
            msg += " (Niska pewność: w oknie nie było większego opadu ani nawodnienia — stan początkowy profilu jest założony.)"
    return msg


def apply_rain_gauge(weather: list, readings: list) -> int:
    """Podmienia opad z modelu na odczyty deszczomierza rolnika (suma
    odczytów z danego dnia). Siatka modelu (9-25 km) nie widzi lokalnych
    burz — odczyt z pola jest lepszy. Zwraca liczbę podmienionych dni."""
    by_day: dict = {}
    for r in readings:
        by_day[r["reading_date"]] = by_day.get(r["reading_date"], 0.0) + float(r["precip_mm"])
    n = 0
    for w in weather:
        key = w["date"].isoformat()
        if key in by_day:
            w["precip_mm"] = by_day[key]
            w["precip_source"] = "gauge"
            n += 1
    return n


def weather_context(geom_4326, wx: list, today: date) -> dict:
    """Kontekst pogodowy do podsumowania bilansu (wszystko NIEBLOKUJĄCE):
    prognoza 7 dni, bilans P−ET0 30 dni na tle normy 1991–2020 i ostatnia
    wilgotność gleby z modelu ERA5-Land."""
    import weather as weather_mod
    out = {"forecast": weather_mod.forecast_summary(wx, today),
           "cwb_30d_mm": None, "cwb_30d_pct": None, "cwb_30d_class": None,
           "soil_moisture": None}
    past = [r for r in wx if r["date"] < today]
    if past:
        end = past[-1]["date"]
        cwb30 = weather_mod.climatic_water_balance(past, end, 30)
        out["cwb_30d_mm"] = cwb30
        try:
            c = geom_4326.centroid
            clim = weather_mod.fetch_climatology(c.y, c.x)
            pct = weather_mod.cwb_percentile(cwb30, end, 30, clim)
            out["cwb_30d_pct"], out["cwb_30d_class"] = pct, weather_mod.cwb_class(pct)
        except Exception as e:
            log.warning("Norma wieloletnia niedostępna: %s", e)
    try:
        # Bilans od początku sezonu ma już wilgotność ERA5 w wierszach archiwum —
        # osobne zapytanie tylko w trybie awaryjnym (sama prognoza).
        last = next((r for r in reversed(wx) if r.get("sm_0_7") is not None), None)
        if last is None:
            arch = weather_mod.fetch_weather_series(geom_4326, today - timedelta(days=14), today)
            last = next((r for r in reversed(arch) if r.get("sm_0_7") is not None), None)
        if last:
            out["soil_moisture"] = {"date": last["date"].isoformat(),
                                    "sm_0_7": last["sm_0_7"], "sm_7_28": last.get("sm_7_28"),
                                    "sm_28_100": last.get("sm_28_100")}
    except Exception as e:
        log.warning("Wilgotność gleby ERA5-Land niedostępna: %s", e)
    return out


def season_start(today: date) -> date:
    """Ostatni 1 marca, który jest co najmniej SEASON_MIN_DAYS dni przed dziś."""
    m, d = SEASON_START
    start = date(today.year, m, d)
    if (today - start).days < SEASON_MIN_DAYS:
        start = date(today.year - 1, m, d)
    return start


def anchor_date(weather: list, ndvi_points: list, events: list, taw: float, p: float,
                kcb_max: float, start: date, checks: list,
                hold_back: Optional[int], ks_salt: float = 1.0) -> Optional[date]:
    """Pierwszy dzień, od którego wynik NIE zależy od stanu początkowego:
    bilanse z profilem pełnym i pustym na starcie są wtedy równe (i już
    zostaną równe — te same dane wejściowe). None = do końca okna zależy."""
    lo = run_balance(weather, ndvi_points, events, taw, p, kcb_max, start_date=start,
                     initial_depletion=0.0, soil_checks=checks, ndvi_hold_back_days=hold_back,
                     ks_salt=ks_salt)
    hi = run_balance(weather, ndvi_points, events, taw, p, kcb_max, start_date=start,
                     initial_depletion=taw, soil_checks=checks, ndvi_hold_back_days=hold_back,
                     ks_salt=ks_salt)
    for a, b in zip(lo, hi):
        if abs(a["depletion_mm"] - b["depletion_mm"]) <= ANCHOR_TOL_MM:
            return a["date"]
    return None


def initial_state_from_era5(wx: list, start: date, taw: float) -> Optional[dict]:
    """Stan profilu na starcie z wilgotności ERA5-Land 0–100 cm (średnia z 3 dni),
    przeliczonej na % wody dostępnej (weather.soil_limits). None, gdy brak
    danych — wtedy wołający zakłada profil pełny."""
    import weather as weather_mod
    th = [weather_mod.root_zone_theta(r) for r in wx if start <= r["date"] <= start + timedelta(days=2)]
    th = [v for v in th if v is not None]
    peak = weather_mod.peak_theta(wx)
    if not th or peak is None:
        return None
    lim = weather_mod.soil_limits(peak)
    theta = sum(th) / len(th)
    fill = min(1.0, max(0.0, (theta - lim["wp"]) / (lim["fc"] - lim["wp"])))
    return {"source": "era5", "date": start.isoformat(), "fill_pct": round(100 * fill),
            "depletion_mm": round((1.0 - fill) * taw, 1), "theta": round(theta, 3), **lim}


def ndvi_points_for_field(field_id: int) -> list:
    """Zieleń łanu do bilansu: obserwacje modułu wodnego + NDVI ze scen OPTRAM
    (kalibracja obejmuje do 24 mies., więc pokrywa początek sezonu)."""
    import db
    pts = {o["obs_date"]: o["ndvi_mean"] for o in db.get_water_observations(field_id)
           if o.get("ndvi_mean") is not None}
    for r in db.get_optram_series(field_id):
        if r.get("ndvi") is not None and (r.get("valid_frac") or 0) >= 0.9 and r["obs_date"] not in pts:
            pts[r["obs_date"]] = r["ndvi"]
    return sorted((date.fromisoformat(d), v) for d, v in pts.items())


def compute_field_balance(field: dict, today: Optional[date] = None) -> dict:
    """Pełny bilans dla pola modułu wodnego: od początku sezonu (1 marca,
    archiwum ERA5 + prognoza 10 dni), stan początkowy z ERA5-Land, NDVI
    z obserwacji, log nawodnień, odczyty deszczomierza i punkty kontrolne.
    Rzuca RuntimeError, gdy pogoda niedostępna."""
    import db
    import weather as weather_mod
    from shapely import wkt as shapely_wkt

    today = today or date.today()
    geom = shapely_wkt.loads(field["geometry_wkt"])
    start = season_start(today)
    mode = "season"
    try:
        # Archiwum od 1 października: zima potrzebna do rozpoznania typu gleby ERA5-Land
        # (bilans i tak liczy się od `start` — run_balance pomija wcześniejsze dni).
        wx = weather_mod.fetch_season_weather(geom, weather_mod.hydro_year_start(start),
                                              forecast_days=FORECAST_DAYS, today=today)
    except Exception as e:
        log.warning("Bilans pola %s: archiwum ERA5 niedostępne (%s) — tryb 92 dni.", field.get("id"), e)
        mode = "recent"
        wx = weather_mod.fetch_recent_and_forecast(geom, past_days=BALANCE_PAST_DAYS,
                                                   forecast_days=FORECAST_DAYS)
        start = min((r["date"] for r in wx), default=today)

    ndvi_points = ndvi_points_for_field(field["id"])
    sowing = None
    if field.get("sowing_date"):
        try:
            sowing = date.fromisoformat(field["sowing_date"])
        except ValueError:
            sowing = None
    if sowing and start < sowing <= today:
        # Przed siewem gleba goła — nie pozwalamy NDVI łanu „cofać się” przed siew.
        ndvi_points = [(start, NDVI_MIN), (sowing, NDVI_MIN)] + [pt for pt in ndvi_points if pt[0] > sowing]
    events = db.get_irrigation_events(field["id"])
    checks = [c for c in db.get_soil_checks(field["id"]) if c["check_date"] >= start.isoformat()]
    gauge_days = apply_rain_gauge(wx, db.get_rain_readings(field["id"]))

    import soil_water
    kcb_max, p, crop_known = crop_params(field.get("crop_type"))
    soil = soil_water.field_taw(field)
    salt = soil_water.field_salinity(field, texture=soil.get("texture"))
    taw = float(soil["taw_mm"])
    taw_assumed = soil["source"] == "default"

    init = None
    if mode == "season":
        try:
            init = initial_state_from_era5(wx, start, taw)
        except Exception as e:
            log.warning("Bilans pola %s: wilgotność ERA5-Land na start niedostępna: %s", field.get("id"), e)
    if init is None:
        init = {"source": "assumed_full", "date": start.isoformat(), "fill_pct": 100, "depletion_mm": 0.0}

    # Stan na starcie z ERA5 opisuje osobne pole `initial_state` (UI pokazuje je
    # w jednej linii razem z datą zakotwiczenia); tu tylko tryb awaryjny.
    assumptions = []
    if init["source"] != "era5":
        assumptions.append("Stan początkowy: profil glebowy pełny na początku okresu.")
    if taw_assumed:     # pozostałe źródła UI pokazuje osobną linią z `soil`
        assumptions.append(f"Retencja użyteczna (TAW) nie podana — przyjęto {DEFAULT_TAW_MM:.0f} mm.")
    if not crop_known:
        assumptions.append(f"Uprawa nieznana — przyjęto Kcb_max={DEFAULT_KCB_MAX}, p={DEFAULT_P}.")
    if not ndvi_points:
        assumptions.append("Brak obserwacji NDVI — przyjęto glebę bez okrywy (Kcb minimalne); "
                           "uruchom 'Sprawdź teraz', żeby pobrać dane satelitarne.")
    elif (ndvi_points[0][0] - start).days > NDVI_HOLD_BACK_DAYS:
        assumptions.append(f"Brak zdjęć zieleni łanu przed {ndvi_points[0][0].isoformat()} — wcześniej "
                           f"przyjęto glebę bez okrywy. Uzupełnij historię („Przygotuj wszystko”).")
    if any(e.get("amount_mm") is None for e in events):
        assumptions.append("Nawodnienia bez dawki w mm potraktowano jako uzupełnienie do pełnej pojemności.")
    if gauge_days:
        assumptions.append(f"Opad z deszczomierza rolnika zastąpił opad z modelu w {gauge_days} "
                           f"{'dniu' if gauge_days == 1 else 'dniach'}.")
    if checks:
        assumptions.append(f"Punkty kontrolne wilgotności z pola: {len(checks)} (ostatni "
                           f"{checks[-1]['check_date']}, {checks[-1]['fill_pct']:.0f}%).")

    kw = dict(start_date=start, soil_checks=checks, ndvi_hold_back_days=NDVI_HOLD_BACK_DAYS,
              ks_salt=salt["ks_salt"])
    days = run_balance(wx, ndvi_points, events, taw, p, kcb_max,
                       initial_depletion=init["depletion_mm"], **kw)
    anchored = anchor_date(wx, ndvi_points, events, taw, p, kcb_max, start, checks, NDVI_HOLD_BACK_DAYS,
                           ks_salt=salt["ks_salt"])
    for r in days:
        r["estimated"] = anchored is None or r["date"] < anchored
    summary = summarize_balance(days, taw, p, today)
    summary["initial_state_reliable"] = anchored is not None and anchored <= today
    summary.update(weather_context(geom, wx, today))
    summary.update({
        "crop_type": field.get("crop_type"), "kcb_max": kcb_max, "p": p,
        "irrigation_mode": field.get("irrigation_mode"),
        "taw_assumed": taw_assumed, "soil": soil, "salinity": salt, "assumptions": assumptions,
        "rain_gauge_days_used": gauge_days,
        "season_start": start.isoformat(), "balance_mode": mode,
        "initial_state": init,
        "anchored_from": anchored.isoformat() if anchored else None,
        "last_soil_check": checks[-1] if checks else None,
        "attribution": weather_mod.OPEN_METEO_ATTRIBUTION,
        "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
    })
    summary["message_pl"] = describe_balance(summary, field.get("irrigation_mode"))
    return {"summary": summary, "days": [{**r, "date": r["date"].isoformat()} for r in days]}
