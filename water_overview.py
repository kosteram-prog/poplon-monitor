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
            balance = water_balance.compute_field_balance(field, today=today)
        except Exception as e:
            log.warning("Pełny obraz: bilans pola %s niedostępny: %s", field.get("id"), e)
            balance_error = str(e)[:200]
    bal = balance["summary"] if balance else None
    sources = [assess_soil(bal), assess_plants(obs, alarms, today), assess_topsoil(opt, today),
               assess_canopy_temp(th_series, today), assess_weather(bal)]
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
                                   today - timedelta(days=HISTORY_DAYS),
                                   topsoil["status"] != "none" or topsoil["code"] == "topsoil_old", events),
        "balance_error": balance_error,
        "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
    }


# ---------------------------------------------------------------------------
# „Przygotuj / odśwież wszystko” — plan kroków z szacunkiem kosztu (PU) i
# wykonanie pojedynczego kroku (interfejs woła kroki po kolei i pokazuje
# postęp). Szacunki PU zmierzone w teście na żywo 11.2026.
# ---------------------------------------------------------------------------

PREP_BACKFILL_DAYS = 120
PREP_MIN_RECENT_OBS = 6
PREP_OPTRAM_MAX_AGE_DAYS = 180
PREP_OPTRAM_MONTHS = 12
PREP_THERMAL_MONTHS = 6
PREP_THERMAL_MIN_HA = 2.0
EST_PU = {"backfill": 3.0, "optram": 12.5, "thermal": 2.5, "check_now": 2.0,
          "optram_refresh": 1.7, "thermal_refresh": 2.5}


def backfill_start(field: dict, today: date) -> Optional[date]:
    """Od kiedy uzupełnić historię, żeby bilans miał zieleń łanu od początku
    sezonu; None — historia (obserwacje + NDVI z OPTRAM) już go pokrywa."""
    import water_balance
    s0 = water_balance.season_start(today)
    pts = water_balance.ndvi_points_for_field(field["id"])
    if pts and (pts[0][0] - s0).days <= water_balance.NDVI_HOLD_BACK_DAYS:
        return None
    return min(s0, today - timedelta(days=PREP_BACKFILL_DAYS))


def prepare_plan(field: dict, today: Optional[date] = None) -> dict:
    import thermal
    today = today or date.today()
    steps = []
    if not field.get("geometry_wkt"):
        return {"steps": [{"step": "check_now", "run": False, "reason": "no_geometry", "est_pu": 0}],
                "total_pu": 0}
    recent = [o for o in db.get_water_observations(field["id"])
              if (_age(o["obs_date"], today) or 999) <= 90]
    bf_start = backfill_start(field, today)
    few = len(recent) < PREP_MIN_RECENT_OBS
    reason = "few_obs" if few else ("season_gap" if bf_start else "has_history")
    bf_days = (today - (bf_start or today - timedelta(days=PREP_BACKFILL_DAYS))).days
    steps.append({"step": "backfill", "run": few or bf_start is not None, "reason": reason,
                  "n_obs": len(recent), "est_pu": round(EST_PU["backfill"] * bf_days / PREP_BACKFILL_DAYS, 1)})
    p = db.get_optram_params(field["id"])
    age = _age((p or {}).get("created_at"), today) if p else None
    need_opt = p is None or (age is not None and age > PREP_OPTRAM_MAX_AGE_DAYS)
    steps.append({"step": "optram", "run": need_opt,
                  "reason": "not_calibrated" if p is None else ("old_calibration" if need_opt else "calibrated"),
                  "est_pu": EST_PU["optram"]})
    area = thermal.field_area_ha(field)
    has_th = bool(db.get_thermal_series(field["id"]))
    if area is not None and area < PREP_THERMAL_MIN_HA:
        steps.append({"step": "thermal", "run": False, "reason": "small_field", "area_ha": area, "est_pu": 0})
    else:
        steps.append({"step": "thermal", "run": not has_th, "reason": "has_data" if has_th else "no_data",
                      "area_ha": area, "est_pu": EST_PU["thermal"]})
    est_check = EST_PU["check_now"] + (EST_PU["optram_refresh"] if (p or need_opt) else 0) + \
        (EST_PU["thermal_refresh"] if (has_th or steps[-1]["run"]) else 0)
    steps.append({"step": "check_now", "run": True, "reason": "always", "est_pu": round(est_check, 1)})
    total = round(sum(s["est_pu"] for s in steps if s["run"]), 1)
    return {"steps": steps, "total_pu": total}


def run_prepare_step(field: dict, step: str, today: Optional[date] = None) -> dict:
    """Wykonuje jeden krok. Nie rzuca — błąd wraca jako ok=False."""
    today = today or date.today()
    try:
        if step == "backfill":
            import water_replay
            start = backfill_start(field, today) or today - timedelta(days=PREP_BACKFILL_DAYS)
            r = water_replay.backfill_field(field, start, today)
            return {"step": step, "ok": True, "stored": r["stored"], "clean": r["clean"]}
        if step == "optram":
            import optram
            r = optram.calibrate_field(field, today - timedelta(days=int(PREP_OPTRAM_MONTHS * 30.4)), today)
            return {"step": step, "ok": True, "quality": r["params"]["quality"], "n_series": len(r["series"]),
                    "era5_r": r["era5"]["r"]}
        if step == "thermal":
            import thermal
            rows = thermal.fetch_field(field, today - timedelta(days=int(PREP_THERMAL_MONTHS * 30.4)), today)
            return {"step": step, "ok": True, "n_scenes": len(rows)}
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
