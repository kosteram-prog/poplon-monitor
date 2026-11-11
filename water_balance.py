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
  - Stan początkowy: profil pełny (D = 0) na początku okna. Wpływ tego
    założenia zanika po pierwszym dużym opadzie/nawodnieniu — gdy takiego
    nie było, wynik ma flagę niskiej pewności.
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
DEFAULT_TAW_MM = 100.0
DEFAULT_KCB_MAX = 1.10
DEFAULT_P = 0.50
BALANCE_PAST_DAYS = 92   # maks. historia z API prognozy Open-Meteo (past_days)
FORECAST_DAYS = 10

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


def interpolate_ndvi(points: list, d: date) -> Optional[float]:
    """Liniowa interpolacja NDVI między obserwacjami; poza zakresem —
    najbliższa wartość (trzymamy ostatnią znaną zieleń w prognozie)."""
    pts = sorted((p for p in points if p[1] is not None), key=lambda p: p[0])
    if not pts:
        return None
    if d <= pts[0][0]:
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
                start_date: Optional[date] = None) -> list:
    """Dzienny bilans od start_date (domyślnie pierwszy dzień pogody).
    `weather`: [{"date", "precip_mm", "et0_mm", "is_forecast"}] posortowane.
    `irrigation_events`: [{"event_date": iso, "amount_mm": float|None}]."""
    raw_mm = p * taw_mm
    irr_by_date: dict = {}
    for e in irrigation_events:
        d = date.fromisoformat(e["event_date"])
        irr_by_date.setdefault(d, []).append(e.get("amount_mm"))

    depletion = 0.0
    out = []
    for w in sorted(weather, key=lambda r: r["date"]):
        d = w["date"]
        if start_date and d < start_date:
            continue
        et0 = w.get("et0_mm")
        precip = w.get("precip_mm") or 0.0
        if et0 is None:
            continue
        ndvi = interpolate_ndvi(ndvi_points, d)
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

        ks = 1.0 if depletion <= raw_mm else max(
            0.0, (taw_mm - depletion) / ((1.0 - p) * taw_mm))
        etc = (ks * kcb + KE_AVG * (1.0 - fc)) * et0
        depletion = depletion - p_eff - irrigation_mm + etc
        if refill:
            depletion = 0.0
        depletion = min(taw_mm, max(0.0, depletion))
        out.append({
            "date": d, "is_forecast": bool(w.get("is_forecast")),
            "ndvi": None if ndvi is None else round(ndvi, 3),
            "kcb": round(kcb, 3), "ks": round(ks, 3),
            "et0_mm": round(et0, 2), "etc_mm": round(etc, 2),
            "p_eff_mm": round(p_eff, 1), "irrigation_mm": round(irrigation_mm, 1),
            "irrigation_refill": refill,
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
    if not summary.get("initial_state_reliable"):
        msg += " (Niska pewność: w oknie nie było większego opadu ani nawodnienia — stan początkowy profilu jest założony.)"
    return msg


def compute_field_balance(field: dict, today: Optional[date] = None) -> dict:
    """Pełny bilans dla pola modułu wodnego: pogoda (ostatnie 92 dni +
    prognoza 10 dni), NDVI z water_observations, log nawodnień z db.
    Rzuca RuntimeError, gdy pogoda niedostępna."""
    import db
    import weather as weather_mod
    from shapely import wkt as shapely_wkt

    today = today or date.today()
    geom = shapely_wkt.loads(field["geometry_wkt"])
    wx = weather_mod.fetch_recent_and_forecast(geom, past_days=BALANCE_PAST_DAYS,
                                               forecast_days=FORECAST_DAYS)
    obs = db.get_water_observations(field["id"])
    ndvi_points = [(date.fromisoformat(o["obs_date"]), o["ndvi_mean"])
                   for o in obs if o.get("ndvi_mean") is not None]
    events = db.get_irrigation_events(field["id"])

    kcb_max, p, crop_known = crop_params(field.get("crop_type"))
    taw = field.get("soil_taw_mm")
    taw_assumed = taw is None
    taw = float(taw) if taw is not None else DEFAULT_TAW_MM

    start = None
    if field.get("sowing_date"):
        try:
            start = date.fromisoformat(field["sowing_date"])
        except ValueError:
            start = None

    assumptions = ["Stan początkowy: profil glebowy pełny na początku okresu."]
    if taw_assumed:
        assumptions.append(f"Retencja użyteczna (TAW) nie podana — przyjęto {DEFAULT_TAW_MM:.0f} mm.")
    if not crop_known:
        assumptions.append(f"Uprawa nieznana — przyjęto Kcb_max={DEFAULT_KCB_MAX}, p={DEFAULT_P}.")
    if not ndvi_points:
        assumptions.append("Brak obserwacji NDVI — przyjęto glebę bez okrywy (Kcb minimalne); "
                           "uruchom 'Sprawdź teraz', żeby pobrać dane satelitarne.")
    if any(e.get("amount_mm") is None for e in events):
        assumptions.append("Nawodnienia bez dawki w mm potraktowano jako uzupełnienie do pełnej pojemności.")

    days = run_balance(wx, ndvi_points, events, taw, p, kcb_max, start_date=start)
    summary = summarize_balance(days, taw, p, today)
    summary.update({
        "crop_type": field.get("crop_type"), "kcb_max": kcb_max, "p": p,
        "irrigation_mode": field.get("irrigation_mode"),
        "taw_assumed": taw_assumed, "assumptions": assumptions,
        "message_pl": describe_balance(summary, field.get("irrigation_mode")),
        "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
    })
    return {"summary": summary, "days": [{**r, "date": r["date"].isoformat()} for r in days]}
