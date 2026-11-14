"""„Pełny obraz pola” — jedna, zrozumiała dla każdego ocena stanu wodnego pola
z pięciu źródeł, które moduł wodny liczy osobno:

  soil         bilans wody w glebie (FAO-56, pogoda + nawodnienia)    water_balance.py
  plants       stan roślin z Sentinel-2/1 (silnik anomalii, alarmy)    water_stress.py
  topsoil      wilgotność wierzchniej warstwy (OPTRAM)                 optram.py
  canopy_temp  temperatura łanu (Landsat, indeks TSI)                  thermal.py
  weather      prognoza i bilans opadów 30 dni na tle wielolecia      weather.py

Każde źródło dostaje status ok / warn / bad / none (brak danych albo wynik
niewiarygodny) i kod komunikatu z parametrami — tekst składa interfejs w
wybranym języku. Ocena ogólna liczy, ile źródeł zgadza się co do niedoboru
wody: jedno źródło to sygnał „do obserwacji”, dwa i więcej — potwierdzenie.

Oś czasu dla wykresu sprowadza wszystko do skali 0–100 „im wyżej, tym więcej
wody”: zapas wody w glebie (%), wilgotność wierzchnia OPTRAM (W·100),
chłodzenie łanu z termiki ((1−TSI)·100), plus NDVI jako tło (zieleń łanu),
opady i nawodnienia (mm) oraz alarmy.

Nic tu nie odpytuje satelitów — tylko bilans pobiera pogodę (z cache).
"""
import logging
from datetime import date, timedelta
from typing import Optional

import db

log = logging.getLogger("water_overview")

HISTORY_DAYS = 120
PLANTS_MAX_AGE_DAYS = 20
ALARM_RECENT_DAYS = 14
TOPSOIL_MAX_AGE_DAYS = 12
THERMAL_MAX_AGE_DAYS = 16
MIN_CANOPY_NDVI = 0.35
THERMAL_MIN_NDVI = 0.3
TOPSOIL_DRY_W = 0.30
TSI_WARN = 0.25
TSI_BAD = 0.5
DRY_MONTH_PCT = 10
WET_FORECAST_MM = 15
BALANCE_RISK_TYPE = "water_balance_deficit"


def _age(d_iso: Optional[str], today: date) -> Optional[int]:
    if not d_iso:
        return None
    return (today - date.fromisoformat(str(d_iso)[:10])).days


def _src(key, status, code, as_of=None, **params):
    return {"key": key, "status": status, "code": code, "as_of": as_of, "params": params}


# ---------------------------------------------------------------------------
# Oceny źródeł (czyste funkcje — dane wejściowe już pobrane)
# ---------------------------------------------------------------------------

def assess_soil(bal: Optional[dict]) -> dict:
    if not bal or bal.get("status") in (None, "no_data"):
        return _src("soil", "none", "soil_no_data")
    p = {"used": round(bal.get("depletion_mm") or 0), "raw": round(bal.get("raw_mm") or 0),
         "taw": round(bal.get("taw_mm") or 0),
         "left_pct": round(100 * (1 - (bal.get("depletion_frac") or 0)))}
    if bal["status"] == "stress":
        return _src("soil", "bad", "soil_stress", bal.get("as_of"), **p)
    if bal.get("days_to_stress") is not None:
        return _src("soil", "warn", "soil_soon", bal.get("as_of"), days=bal["days_to_stress"],
                    date=bal.get("suggested_irrigation_date"), **p)
    if bal["status"] == "approaching":
        return _src("soil", "warn", "soil_approach", bal.get("as_of"), **p)
    return _src("soil", "ok", "soil_ok", bal.get("as_of"), days=bal.get("forecast_horizon_days") or 0, **p)


def assess_plants(obs: list, alarms: list, today: date) -> dict:
    sat_alarms = [a for a in alarms if a.get("risk_type") != BALANCE_RISK_TYPE
                  and a.get("triage_status") != "rejected"
                  and (_age(a.get("obs_date"), today) or 999) <= ALARM_RECENT_DAYS]
    if sat_alarms:
        a = max(sat_alarms, key=lambda x: x.get("obs_date") or "")
        status = "bad" if a.get("triage_status") == "confirmed" else "warn"
        return _src("plants", status, "plants_alarm", a.get("obs_date"), risk=a.get("risk_type"),
                    triage=a.get("triage_status"))
    with_ndvi = [o for o in obs if o.get("ndvi_mean") is not None]
    if not with_ndvi or (_age(with_ndvi[-1]["obs_date"], today) or 999) > PLANTS_MAX_AGE_DAYS:
        return _src("plants", "none", "plants_no_data",
                    with_ndvi[-1]["obs_date"] if with_ndvi else None)
    last = with_ndvi[-1]
    if last["ndvi_mean"] < MIN_CANOPY_NDVI:
        return _src("plants", "none", "plants_low_cover", last["obs_date"], ndvi=round(last["ndvi_mean"], 2))
    return _src("plants", "ok", "plants_ok", last["obs_date"], ndvi=round(last["ndvi_mean"], 2))


def assess_topsoil(opt: Optional[dict], today: date) -> dict:
    import optram
    if not opt or not opt.get("calibrated"):
        return _src("topsoil", "none", "topsoil_off")
    cur = opt.get("current")
    if not cur or (_age(cur["obs_date"], today) or 999) > TOPSOIL_MAX_AGE_DAYS:
        return _src("topsoil", "none", "topsoil_old", cur["obs_date"] if cur else None)
    era = opt.get("era5") or {}
    if (opt.get("params") or {}).get("quality") == "poor" or (
            era.get("n", 0) >= optram.ERA5_MIN_N and era.get("r") is not None and era["r"] < optram.ERA5_MIN_R):
        return _src("topsoil", "none", "topsoil_unreliable", cur["obs_date"], w=round(cur["w"], 2),
                    r=era.get("r"))
    p = {"w": round(cur["w"], 2), "pct": round(100 * cur["w"])}
    if cur["w"] < TOPSOIL_DRY_W:
        return _src("topsoil", "warn", "topsoil_dry", cur["obs_date"], **p)
    return _src("topsoil", "ok", "topsoil_moist", cur["obs_date"], **p)


def assess_canopy_temp(series: list, today: date) -> dict:
    if not series:
        return _src("canopy_temp", "none", "temp_off")
    last = series[-1]
    if (_age(last["scene_date"], today) or 999) > THERMAL_MAX_AGE_DAYS:
        return _src("canopy_temp", "none", "temp_old", last["scene_date"])
    if last.get("ndvi") is None or last["ndvi"] < THERMAL_MIN_NDVI or last.get("tsi") is None:
        return _src("canopy_temp", "none", "temp_low_cover", last["scene_date"])
    p = {"tsi": round(last["tsi"], 2), "dt": None if last.get("t_cold") is None
         else round(last["t_field"] - last["t_cold"], 1)}
    if last["tsi"] >= TSI_BAD:
        return _src("canopy_temp", "bad", "temp_hot", last["scene_date"], **p)
    if last["tsi"] >= TSI_WARN:
        return _src("canopy_temp", "warn", "temp_warm", last["scene_date"], **p)
    return _src("canopy_temp", "ok", "temp_cool", last["scene_date"], **p)


def assess_weather(bal: Optional[dict]) -> dict:
    fc = (bal or {}).get("forecast")
    if not fc:
        return _src("weather", "none", "weather_no_data")
    p = {"days": fc.get("days"), "rain": round(fc.get("precip_mm") or 0),
         "et0": round(fc.get("et0_mm") or 0),
         "tmax": None if fc.get("tmax_c") is None else round(fc["tmax_c"])}
    pct = (bal or {}).get("cwb_30d_pct")
    if fc.get("dry_hot"):
        return _src("weather", "warn", "weather_dry_hot", **p)
    if pct is not None and pct <= DRY_MONTH_PCT:
        return _src("weather", "warn", "weather_dry_month", pct=round(pct), **p)
    if p["rain"] >= WET_FORECAST_MM:
        return _src("weather", "ok", "weather_rain", **p)
    return _src("weather", "ok", "weather_normal", **p)


TOPSOIL_WET_W = 0.6


def conflicts(sources: list) -> list:
    """Sprzeczności między źródłami — najcenniejsza wskazówka dla doradcy.
    Test na żywo (pole „uzbek 2”, 11.2026): bilans „stres, zużyte 68 mm”,
    a OPTRAM W = 0,96 — model nie wie o wodzie spoza dziennika."""
    by = {s["key"]: s for s in sources}
    out = []
    soil, top = by["soil"], by["topsoil"]
    if soil["status"] in ("bad", "warn") and top["status"] == "ok" and \
            (top["params"].get("w") or 0) >= TOPSOIL_WET_W:
        out.append({"code": "conflict_unlogged_water", "w": top["params"]["w"]})
    if soil["status"] in ("bad", "warn") and by["canopy_temp"]["status"] == "ok":
        # Łan chłodny jak najlepiej nawodniona roślinność = rośliny mają wodę.
        out.append({"code": "conflict_unlogged_water_temp"})
    if soil["status"] == "ok" and any(by[k]["status"] == "bad" for k in ("plants", "canopy_temp")):
        out.append({"code": "conflict_not_water"})
    return out


def overall(sources: list, field: dict, bal: Optional[dict]) -> dict:
    core_src = [s for s in sources if s["key"] != "weather"]
    bad = [s["key"] for s in core_src if s["status"] == "bad"]
    warn = [s["key"] for s in sources if s["status"] == "warn"]
    known = [s for s in core_src if s["status"] != "none"]
    # Pogoda to prognoza, nie sygnał z pola — sama nie zmienia oceny
    # (audyt 11.2026), tylko dokłada wskazówkę w „Co teraz”.
    core_warn = [k for k in warn if k != "weather"]
    if bad and len(bad) + len(core_warn) >= 2:
        level, code = "bad", "overall_confirmed"
    elif bad:
        level, code = "bad", "overall_possible"
    elif core_warn:
        level, code = "warn", "overall_watch"
    elif known:
        level, code = "ok", "overall_good"
    else:
        level, code = "none", "overall_no_data"
    conf = conflicts(sources)
    action = None
    irrigated = field.get("irrigation_mode") == "irrigation"
    if any(c["code"].startswith("conflict_unlogged_water") for c in conf) and bad in ([], ["soil"]):
        # Jedynym „złym” źródłem jest model, a satelita widzi mokrą glebę —
        # nie alarmujemy, tylko prosimy o uzupełnienie dziennika.
        level, code = "warn", "overall_conflict"
        return {"level": level, "code": code, "bad": bad, "warn": warn, "n_known": len(known),
                "n_sources": len(core_src), "conflicts": conf,
                "action": {"code": "action_log_water"}}
    if irrigated and bal and bal.get("suggested_irrigation_date") and bal.get("suggested_dose_mm") is not None:
        action = {"code": "action_irrigate", "date": bal["suggested_irrigation_date"],
                  "mm": round(bal["suggested_dose_mm"]), "m3": round(bal.get("suggested_dose_m3_ha") or 0)}
    elif irrigated and level == "bad":
        action = {"code": "action_check_irrigation"}
    elif not irrigated and level == "bad":
        action = {"code": "action_rainfed"}
    elif level == "ok" and "weather" in warn:
        action = {"code": "action_weather_watch"}
    elif level in ("ok", "warn"):
        action = {"code": "action_none" if level == "ok" else "action_watch"}
    return {"level": level, "code": code, "bad": bad, "warn": warn,
            "n_known": len(known), "n_sources": len(core_src), "conflicts": conf, "action": action}


# ---------------------------------------------------------------------------
# Oś czasu (0–100, w górę = więcej wody)
# ---------------------------------------------------------------------------

def chart_start(today: date, balance: Optional[dict]) -> date:
    """Wykres ma pokazać cały liczony bilans (od 1 marca), nie tylko ostatnie
    120 dni. Bez daty sezonu zostaje dotychczasowe okno."""
    window = today - timedelta(days=HISTORY_DAYS)
    raw = ((balance or {}).get("summary") or {}).get("season_start")
    if not raw:
        return window
    try:
        start = date.fromisoformat(str(raw)[:10])
    except ValueError:
        return window
    return min(start, window)


NDMI_LO, NDMI_HI = -0.2, 0.6
CIRE_MIN_PEAK = 0.5    # poniżej: brak łanu w sezonie (goła gleba) — krzywa chlorofilu nic nie mówi


# ---------------------------------------------------------------------------
# Rzeczywiste parowanie (ETa) z termiki Landsat — podejście w stylu SSEBop
# (Senay i in. 2013): ułamek parowania ETf = 1 − TSI (0 = gorące jak sucha
# gleba, 1 = chłodne jak najlepiej uwodniona roślinność w okolicy),
# ETa = ETf · Kc_max · ET0. Między przelotami ETf interpolowane liniowo (przerwy
# do ETA_MAX_GAP_DAYS), dalej brak wartości. Porównanie z ETc z bilansu FAO-56
# (ten sam ET0) mówi, czy pole naprawdę parowało tyle, ile zakłada model.
# Niekalibrowane; na polach < thermal.SMALL_FIELD_HA przybliżone (piksel 100 m).
# ---------------------------------------------------------------------------

ETA_MAX_GAP_DAYS = 32
ETA_LOW_RATIO = 0.75     # termika < 75% modelu: pole paruje mniej, niż zakłada bilans
ETA_HIGH_RATIO = 1.25    # termika > 125% modelu: więcej wody, niż zna bilans
ETA_MIN_DAYS = 10


def _etf_at(scenes: list, d: date) -> Optional[float]:
    for i, (sd, f) in enumerate(scenes):
        if sd == d:
            return f
        if sd > d:
            if i == 0:
                return None
            pd, pf = scenes[i - 1]
            if (sd - pd).days > ETA_MAX_GAP_DAYS:
                return None
            return pf + (f - pf) * (d - pd).days / (sd - pd).days
    return None


def thermal_eta(balance: Optional[dict], thermal_series: list, kc_max: float,
                small_field: bool = False) -> Optional[dict]:
    if not balance or not thermal_series:
        return None
    scenes = sorted((date.fromisoformat(str(r["scene_date"])[:10]), max(0.0, min(1.0, 1.0 - r["tsi"])))
                    for r in thermal_series
                    if r.get("tsi") is not None and (r.get("ndvi") or 0) >= THERMAL_MIN_NDVI)
    if not scenes:
        return None
    scene_dates = {sd for sd, _ in scenes}
    days, points = [], []
    for d in balance.get("days") or []:
        if d.get("is_forecast") or d.get("et0_mm") is None or d.get("etc_mm") is None:
            continue
        dd = date.fromisoformat(str(d["date"])[:10])
        etf = _etf_at(scenes, dd)
        if etf is None:
            continue
        eta = etf * kc_max * d["et0_mm"]
        days.append((dd, eta, d["etc_mm"]))
        if dd in scene_dates:
            points.append({"date": dd.isoformat(), "eta_mm": round(eta, 1), "model_mm": round(d["etc_mm"], 1),
                           "etf": round(etf, 2)})
    if len(days) < ETA_MIN_DAYS:
        return {"points": points, "season": None, "approx": small_field}
    eta_sum = sum(x[1] for x in days)
    model_sum = sum(x[2] for x in days)
    ratio = eta_sum / model_sum if model_sum > 0 else None
    verdict = None if ratio is None else ("lower" if ratio < ETA_LOW_RATIO else
                                         ("higher" if ratio > ETA_HIGH_RATIO else "agree"))
    return {"points": points, "approx": small_field, "kc_max": kc_max,
            "season": {"from": days[0][0].isoformat(), "to": days[-1][0].isoformat(), "days": len(days),
                       "eta_mm": round(eta_sum), "model_mm": round(model_sum),
                       "pct": None if ratio is None else round(100 * ratio), "verdict": verdict}}


def _soil_points(balance: Optional[dict], start: date) -> dict:
    if not balance:
        return {"soil": [], "rain": [], "irrigation": [], "soil_checks": [], "threshold_pct": None}
    s = balance["summary"]
    taw, raw = s.get("taw_mm"), s.get("raw_mm")
    out = {"soil": [], "rain": [], "irrigation": [], "soil_checks": [],
           "threshold_pct": None if not taw else round(100 * (1 - raw / taw), 1),
           "anchored_from": s.get("anchored_from"), "season_start": s.get("season_start"),
           "initial_source": (s.get("initial_state") or {}).get("source")}
    s0 = start.isoformat()
    for d in balance["days"]:
        if d["date"] < s0 or not taw:
            continue
        out["soil"].append({"date": d["date"], "v": round(100 * (1 - d["depletion_mm"] / taw), 1),
                            "forecast": bool(d["is_forecast"]), "est": bool(d.get("estimated"))})
        if d.get("soil_check_pct") is not None:
            out["soil_checks"].append({"date": d["date"], "v": d["soil_check_pct"]})
        if d.get("p_eff_mm"):
            out["rain"].append({"date": d["date"], "mm": d["p_eff_mm"], "forecast": bool(d["is_forecast"])})
        if d.get("irrigation_mm"):
            out["irrigation"].append({"date": d["date"], "mm": d["irrigation_mm"]})
    return out


def build_timeline(balance: Optional[dict], obs: list, opt: Optional[dict], thermal_series: list,
                   alarms: list, start: date, topsoil_reliable: bool, events: list) -> dict:
    s0 = start.isoformat()
    tl = _soil_points(balance, start)
    # Nawodnienia z dziennika bez dawki też pokazujemy (znacznik bez mm).
    logged = {p["date"] for p in tl["irrigation"]}
    for e in events:
        d = str(e.get("event_date"))[:10]
        if d >= s0 and d not in logged:
            tl["irrigation"].append({"date": d, "mm": e.get("amount_mm")})
    tl["greenness"] = [{"date": o["obs_date"], "v": round(100 * o["ndvi_mean"], 1)}
                       for o in obs if o.get("ndvi_mean") is not None and o["obs_date"] >= s0]
    # EVI2 — zieleń bez nasycenia przy zwartym łanie (NDVI dochodzi do ~0,85 i stoi).
    tl["greenness_evi2"] = [{"date": o["obs_date"], "v": round(100 * o["evi2_mean"], 1)}
                            for o in obs if o.get("evi2_mean") is not None and o["obs_date"] >= s0]
    # Wilgotność liści (NDMI, w bazie ndwi_mean) na skali 0–100: −0,2 (goła,
    # sucha gleba) -> 0, 0,6 (zwarty, nawodniony łan) -> 100. Para do chlorofilu.
    tl["leaf_water"] = [{"date": o["obs_date"], "v": round(max(0.0, min(100.0, 100 * (o["ndwi_mean"] - NDMI_LO) / (NDMI_HI - NDMI_LO))), 1),
                         "ndmi": round(o["ndwi_mean"], 3)}
                        for o in obs if o.get("ndwi_mean") is not None and o["obs_date"] >= s0]
    # Chlorofil (CIred-edge) jako % szczytu sezonu na tym polu — liczy się
    # przebieg: spada NDMI, a chlorofil stoi -> woda; spada chlorofil, a NDMI
    # stoi -> raczej azot, choroba, zasolenie. Bez progów (niekalibrowane).
    ci = [(o["obs_date"], o["cire_mean"]) for o in obs
          if o.get("cire_mean") is not None and o["obs_date"] >= s0]
    # szczyt liczony z całego sezonu, nawet gdy wykres pokazuje krótsze okno
    season0 = min(s0, str((balance or {}).get("summary", {}).get("season_start") or s0)[:10])
    peak = max((o["cire_mean"] for o in obs if o.get("cire_mean") is not None and o["obs_date"] >= season0),
               default=0)
    tl["chlorophyll"] = [{"date": d_, "v": round(100 * max(v, 0) / peak, 1), "cire": round(v, 2)}
                         for d_, v in ci] if peak >= CIRE_MIN_PEAK else []
    tl["topsoil"] = [{"date": r["obs_date"], "v": round(100 * r["w"], 1)}
                     for r in ((opt or {}).get("series") or []) if r["obs_date"] >= s0]
    tl["topsoil_reliable"] = topsoil_reliable
    tl["canopy_cooling"] = [{"date": r["scene_date"], "v": round(100 * (1 - r["tsi"]), 1),
                             "t_field": r.get("t_field"), "t_cold": r.get("t_cold"), "t_hot": r.get("t_hot")}
                            for r in thermal_series
                            if r["scene_date"] >= s0 and r.get("tsi") is not None
                            and (r.get("ndvi") or 0) >= THERMAL_MIN_NDVI]
    tl["alarms"] = [{"date": a["obs_date"], "risk": a.get("risk_type"), "triage": a.get("triage_status")}
                    for a in alarms if a.get("obs_date", "") >= s0 and a.get("triage_status") != "rejected"]
    tl["start"] = s0
    return tl


# ---------------------------------------------------------------------------
# Całość
# ---------------------------------------------------------------------------

def build_overview(field: dict, today: Optional[date] = None, with_balance: bool = True) -> dict:
    import optram
    today = today or date.today()
    obs = db.get_water_observations(field["id"])
    alarms = db.get_water_alarms(water_field_id=field["id"])
    th_series = db.get_thermal_series(field["id"])
    events = db.get_irrigation_events(field["id"])
    opt = optram.get_field_optram(field["id"])
    balance, balance_error = None, None
    if with_balance and field.get("geometry_wkt"):
        try:
            import water_balance
            balance = water_balance.compute_field_balance(field, today=today, quick_soil=True)
        except Exception as e:
            log.warning("Pełny obraz: bilans pola %s niedostępny: %s", field.get("id"), e)
            balance_error = str(e)[:200]
    bal = balance["summary"] if balance else None
    sources = [assess_soil(bal), assess_plants(obs, alarms, today), assess_topsoil(opt, today),
               assess_canopy_temp(th_series, today), assess_weather(bal)]
    eta = None
    try:
        import thermal
        import water_balance
        area = thermal.field_area_ha(field)
        eta = thermal_eta(balance, th_series, water_balance.crop_params(field.get("crop_type"))[0],
                          small_field=area is not None and area < thermal.SMALL_FIELD_HA)
    except Exception as e:
        log.warning("Pełny obraz: ETa z termiki pola %s: %s", field.get("id"), e)
    irr_check = None
    try:
        import irrigation_detect
        irr_check = irrigation_detect.irrigation_visibility(
            events, obs, (opt or {}).get("series") or [], th_series, balance,
            start=date.fromisoformat(chart_start(today, balance).isoformat()))
    except Exception as e:
        log.warning("Pełny obraz: wykrywanie nawodnień pola %s: %s", field.get("id"), e)
    if field.get("irrigation_mode") == "irrigation" and sources[0]["status"] != "none":
        # Bilans pola nawadnianego bez wpisów w dzienniku zakłada brak
        # nawadniania — trzeba to powiedzieć wprost (w Uzbekistanie rolnicy
        # rzadko prowadzą dziennik, a model pokazuje wtedy „suszę”).
        window_start = (today - timedelta(days=92)).isoformat()
        if not any(str(e.get("event_date"))[:10] >= window_start for e in events) \
                and not recent_soil_check(field["id"], today):
            sources[0]["params"]["no_irrigation_log"] = True
    topsoil = sources[2]
    return {
        "field": {"id": field["id"], "parcel_id": field.get("parcel_id"),
                  "irrigation_mode": field.get("irrigation_mode"), "crop_type": field.get("crop_type")},
        "as_of": today.isoformat(),
        "overall": overall(sources, field, bal),
        "sources": sources,
        "timeline": build_timeline(balance, obs, opt, th_series, alarms,
                                   chart_start(today, balance),
                                   topsoil["status"] != "none" or topsoil["code"] == "topsoil_old", events),
        "balance_error": balance_error,
        "eta": eta,
        "irrigation_check": irr_check,
        "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
    }


# ---------------------------------------------------------------------------
# „Przygotuj / odśwież wszystko” — plan kroków z szacunkiem kosztu (PU) i
# wykonanie pojedynczego kroku (interfejs woła kroki po kolei i pokazuje
# postęp). Szacunki PU zmierzone w teście na żywo 11.2026.
# ---------------------------------------------------------------------------

PREP_BACKFILL_DAYS = 120
PREP_MIN_RECENT_OBS = 6
PREP_STALE_DAYS = 15            # ostatnia obserwacja starsza -> uzupełnij lukę (np. serwer był wyłączony)
PREP_OPTRAM_MAX_AGE_DAYS = 180
PREP_OPTRAM_MONTHS = 12
PREP_THERMAL_MONTHS = 6
PREP_THERMAL_COVER_DAYS = 20    # termika „sięga początku sezonu”, gdy pierwsza scena jest najpóźniej tyle dni po nim
EST_PU = {"backfill": 3.0, "optram": 12.5, "thermal": 2.5, "check_now": 2.0,
          "optram_refresh": 1.7, "thermal_refresh": 2.5, "soil": 0.0}
# Kolejność kroków: gleba przed „najnowszymi danymi”, żeby bilans w ostatnim
# kroku liczył już z pobraną pojemnością wodną.
PREP_STEPS = ("backfill", "optram", "thermal", "soil", "check_now")


def backfill_start(field: dict, today: date) -> Optional[date]:
    """Od kiedy uzupełnić historię, żeby bilans miał zieleń łanu od początku
    sezonu; None — historia (obserwacje + NDVI z OPTRAM) już go pokrywa."""
    import water_balance
    s0 = water_balance.season_start(today)
    pts = water_balance.ndvi_points_for_field(field["id"])
    if pts and (pts[0][0] - s0).days <= water_balance.NDVI_HOLD_BACK_DAYS:
        return None
    return min(s0, today - timedelta(days=PREP_BACKFILL_DAYS))


def backfill_window(field: dict, today: date) -> tuple:
    """(start | None, reason). Uzupełnia: brak historii, lukę od początku
    sezonu albo lukę od ostatniej obserwacji (gdy serwer/scheduler nie
    działał). Istniejące daty i tak są pomijane (INSERT OR IGNORE)."""
    obs = db.get_water_observations(field["id"])
    recent = [o for o in obs if (_age(o["obs_date"], today) or 999) <= 90]
    if len(recent) < PREP_MIN_RECENT_OBS:
        return (backfill_start(field, today) or today - timedelta(days=PREP_BACKFILL_DAYS)), "few_obs"
    s = backfill_start(field, today)
    if s is not None:
        return s, "season_gap"
    last = max(o["obs_date"] for o in obs)
    if (_age(last, today) or 0) > PREP_STALE_DAYS:
        return date.fromisoformat(str(last)[:10]), "stale"
    missing = missing_indicators_start(field, obs, today)
    if missing:
        return missing, "new_indicators"
    return None, "has_history"


def _indicators_marker(field_id: int) -> str:
    return f"indicators_fill:{field_id}"


def missing_indicators_start(field: dict, obs: list, today: date) -> Optional[date]:
    """Pierwsza obserwacja sezonu bez EVI2/red-edge (zapisana, zanim moduł je
    liczył). Uzupełniamy raz na pole — gdyby scena już nie wracała z
    Sentinel Hub, nie płacimy za to przy każdym kliknięciu."""
    import water_balance
    if db.get_setting(_indicators_marker(field["id"])):
        return None
    s0 = water_balance.season_start(today).isoformat()
    dates = [o["obs_date"] for o in obs if o["obs_date"] >= s0
             and o.get("ndvi_mean") is not None and o.get("cire_mean") is None]
    return date.fromisoformat(min(dates)[:10]) if dates else None


def thermal_window(field: dict, today: date) -> tuple:
    """(start | None, mode): mode "full" — pobierz od początku sezonu (albo 6
    mies., co wcześniej), "refresh" — termika pokrywa sezon, dociągnij nowe sceny."""
    import water_balance
    s0 = min(water_balance.season_start(today), today - timedelta(days=int(PREP_THERMAL_MONTHS * 30.4)))
    series = db.get_thermal_series(field["id"])
    if series and (date.fromisoformat(series[0]["scene_date"][:10]) - s0).days <= PREP_THERMAL_COVER_DAYS:
        return None, "refresh"
    return s0, "full"


def prepare_plan(field: dict, today: Optional[date] = None) -> dict:
    """Plan „Przygotuj / odśwież wszystko”: każde kliknięcie to pełne
    odświeżenie wszystkich źródeł. Krok jest pomijany tylko wtedy, gdy nie ma
    nic do zrobienia — i zawsze z podanym powodem."""
    import soil_water
    import thermal
    today = today or date.today()
    if not field.get("geometry_wkt"):
        return {"steps": [{"step": "check_now", "run": False, "reason": "no_geometry", "est_pu": 0}],
                "total_pu": 0}
    steps = []
    n_recent = len([o for o in db.get_water_observations(field["id"])
                    if (_age(o["obs_date"], today) or 999) <= 90])
    bf_start, bf_reason = backfill_window(field, today)
    bf_days = (today - bf_start).days if bf_start else 0
    steps.append({"step": "backfill", "run": bf_start is not None, "reason": bf_reason, "n_obs": n_recent,
                  "from": bf_start.isoformat() if bf_start else None,
                  "est_pu": round(EST_PU["backfill"] * max(bf_days, 16) / PREP_BACKFILL_DAYS, 1) if bf_start else 0})

    p = db.get_optram_params(field["id"])
    age = _age((p or {}).get("created_at"), today) if p else None
    if p is None or (age is not None and age > PREP_OPTRAM_MAX_AGE_DAYS):
        steps.append({"step": "optram", "run": True, "mode": "calibrate",
                      "reason": "not_calibrated" if p is None else "old_calibration", "est_pu": EST_PU["optram"]})
    elif p.get("quality") == "poor":
        steps.append({"step": "optram", "run": False, "mode": "refresh", "reason": "poor_calibration", "est_pu": 0})
    else:
        steps.append({"step": "optram", "run": True, "mode": "refresh", "reason": "calibrated",
                      "est_pu": EST_PU["optram_refresh"]})

    area = thermal.field_area_ha(field)
    th_start, th_mode = thermal_window(field, today)
    th_pu = EST_PU["thermal"] if th_mode == "refresh" else \
        round(EST_PU["thermal"] * max(1.0, (today - th_start).days / (PREP_THERMAL_MONTHS * 30.4)), 1)
    steps.append({"step": "thermal", "run": True, "mode": th_mode,
                  "reason": "small_field" if (area is not None and area < thermal.SMALL_FIELD_HA) else
                  ("has_data" if th_mode == "refresh" else "no_data"),
                  "from": th_start.isoformat() if th_start else None, "area_ha": area, "est_pu": th_pu})

    soil = soil_water.field_taw(field, fetch=False)
    if soil["source"] in ("manual", "texture_field", "soilgrids"):
        steps.append({"step": "soil", "run": False, "reason": "soil_" + soil["source"],
                      "taw_mm": soil["taw_mm"], "est_pu": 0})
    else:
        steps.append({"step": "soil", "run": True, "reason": "soil_default", "est_pu": 0})

    steps.append({"step": "check_now", "run": True, "reason": "always", "est_pu": EST_PU["check_now"]})
    total = round(sum(s["est_pu"] for s in steps if s["run"]), 1)
    return {"steps": steps, "total_pu": total}


def run_prepare_step(field: dict, step: str, today: Optional[date] = None) -> dict:
    """Wykonuje jeden krok. Nie rzuca — błąd wraca jako ok=False."""
    today = today or date.today()
    try:
        if step == "backfill":
            import water_replay
            start, _ = backfill_window(field, today)
            start = start or today - timedelta(days=PREP_STALE_DAYS)
            missing = missing_indicators_start(field, db.get_water_observations(field["id"]), today)
            r = water_replay.backfill_field(field, start, today)
            if missing is None or start <= missing:
                # okno objęło wszystkie obserwacje bez EVI2/red-edge — więcej nie próbujemy
                db.set_setting(_indicators_marker(field["id"]), today.isoformat())
            return {"step": step, "ok": True, "stored": r["stored"], "clean": r["clean"],
                    "updated": r.get("updated", 0)}
        if step == "optram":
            import optram
            p = db.get_optram_params(field["id"])
            age = _age((p or {}).get("created_at"), today) if p else None
            if p and p.get("quality") != "poor" and (age is None or age <= PREP_OPTRAM_MAX_AGE_DAYS):
                n = optram.update_recent(field, force=True)
                return {"step": step, "ok": True, "mode": "refresh", "n_new": n}
            r = optram.calibrate_field(field, today - timedelta(days=int(PREP_OPTRAM_MONTHS * 30.4)), today)
            return {"step": step, "ok": True, "mode": "calibrate", "quality": r["params"]["quality"],
                    "n_series": len(r["series"]), "era5_r": r["era5"]["r"]}
        if step == "thermal":
            import thermal
            start, mode = thermal_window(field, today)
            if mode == "refresh":
                n = thermal.update_recent(field, force=True)
            else:
                n = len(thermal.fetch_field(field, start, today))
            return {"step": step, "ok": True, "mode": mode, "n_scenes": n}
        if step == "soil":
            import soil_water
            row = db.get_field_soil(field["id"]) or {}
            if not row.get("sg_profile_json"):
                # świeża próba, nawet jeśli poprzednia nie wyszła przed chwilą
                db.save_field_soil(field["id"], sg_error=None, sg_error_at=None)
            s = soil_water.field_taw(field, fetch=True)
            if s["source"] == "default":
                err = (db.get_field_soil(field["id"]) or {}).get("sg_error") or "SoilGrids niedostępne"
                return {"step": step, "ok": False, "error": err, "taw_mm": s["taw_mm"]}
            return {"step": step, "ok": True, "taw_mm": s["taw_mm"], "source": s["source"],
                    "texture": s.get("texture")}
        if step == "check_now":
            import water_stress
            v = water_stress.check_field_now(field)
            return {"step": step, "ok": True, "fresh": bool(v.get("fresh_observation_fetched")),
                    "risk_flag": bool(v.get("risk_flag"))}
        return {"step": step, "ok": False, "error": "Nieznany krok."}
    except Exception as e:
        log.exception("Przygotowanie pola %s, krok %s", field.get("id"), step)
        return {"step": step, "ok": False, "error": str(e)[:300]}


# ---------------------------------------------------------------------------
# Czy bilansowi wolno wierzyć? (audyt 11.2026)
# Bilans pola NAWADNIANEGO bez wpisów w dzienniku zakłada, że pola nikt nie
# podlewał — i zawsze „widzi” wyczerpany zapas wody. Na polach w Uzbekistanie
# dawało to stały alarm bilansowy w kolejce doradcy co 7 dni, czerwone
# „Nawodnij teraz” w panelu rolnika i — przy włączonych auto-sugestiach — SMS
# z zaleceniem podlewania mokrego pola. Wtedy bilans zostaje kontekstem, ale
# nie wyzwala alarmów ani zaleceń dla rolnika.
# ---------------------------------------------------------------------------

BALANCE_LOG_WINDOW_DAYS = 92


SOIL_CHECK_TRUST_DAYS = 10


def recent_soil_check(field_id: int, today: date) -> bool:
    cutoff = (today - timedelta(days=SOIL_CHECK_TRUST_DAYS)).isoformat()
    return any(cutoff <= c["check_date"] <= today.isoformat() for c in db.get_soil_checks(field_id))


def balance_trust(field: dict, today: Optional[date] = None) -> dict:
    """{"trusted": bool, "reason": None | "no_irrigation_log" | "satellite_disagrees"}"""
    import optram
    today = today or date.today()
    if field.get("irrigation_mode") != "irrigation":
        return {"trusted": True, "reason": None}
    if recent_soil_check(field["id"], today):
        # Świeża ocena wilgotności w polu ustawia bilans — ważniejsza niż brak
        # dziennika czy sygnał satelity (przez kilka dni od oceny).
        return {"trusted": True, "reason": None}
    start = (today - timedelta(days=BALANCE_LOG_WINDOW_DAYS)).isoformat()
    if not any(str(e.get("event_date"))[:10] >= start for e in db.get_irrigation_events(field["id"])):
        return {"trusted": False, "reason": "no_irrigation_log"}
    top = assess_topsoil(optram.get_field_optram(field["id"]), today)
    temp = assess_canopy_temp(db.get_thermal_series(field["id"]), today)
    if (top["status"] == "ok" and (top["params"].get("w") or 0) >= TOPSOIL_WET_W) or temp["status"] == "ok":
        return {"trusted": False, "reason": "satellite_disagrees"}
    return {"trusted": True, "reason": None}
