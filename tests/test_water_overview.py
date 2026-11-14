"""„Pełny obraz pola” (water_overview.py)."""
import json
from datetime import date, timedelta

import pytest

import water_overview as wo

TODAY = date(2026, 7, 20)
WKT = "POLYGON((64.40 40.20, 64.41 40.20, 64.41 40.21, 64.40 40.21, 64.40 40.20))"


def d(n):
    return (TODAY - timedelta(days=n)).isoformat()


def bal(**kw):
    b = {"status": "ok", "as_of": TODAY.isoformat(), "depletion_mm": 20, "depletion_frac": 0.2, "raw_mm": 50,
         "taw_mm": 100, "days_to_stress": None, "suggested_irrigation_date": None, "suggested_dose_mm": None,
         "forecast_horizon_days": 9, "forecast": {"days": 7, "precip_mm": 2, "et0_mm": 40, "tmax_c": 36,
                                                 "dry_hot": False}, "cwb_30d_pct": 40}
    b.update(kw)
    return b


# --- źródła -----------------------------------------------------------------------

def test_soil():
    assert wo.assess_soil(None)["status"] == "none"
    assert wo.assess_soil(bal())["code"] == "soil_ok"
    s = wo.assess_soil(bal(days_to_stress=2, suggested_irrigation_date=d(-2)))
    assert s["status"] == "warn" and s["params"]["days"] == 2
    s = wo.assess_soil(bal(status="stress", depletion_mm=70, depletion_frac=0.7))
    assert s["status"] == "bad" and s["params"]["left_pct"] == 30


def test_plants():
    obs = [{"obs_date": d(3), "ndvi_mean": 0.7}]
    assert wo.assess_plants(obs, [], TODAY)["code"] == "plants_ok"
    assert wo.assess_plants([{"obs_date": d(3), "ndvi_mean": 0.2}], [], TODAY)["code"] == "plants_low_cover"
    assert wo.assess_plants([{"obs_date": d(40), "ndvi_mean": 0.7}], [], TODAY)["code"] == "plants_no_data"
    al = [{"obs_date": d(2), "risk_type": "water_stress_risk", "triage_status": "pending"}]
    assert wo.assess_plants(obs, al, TODAY)["status"] == "warn"
    al[0]["triage_status"] = "confirmed"
    assert wo.assess_plants(obs, al, TODAY)["status"] == "bad"
    al[0]["triage_status"] = "rejected"
    assert wo.assess_plants(obs, al, TODAY)["status"] == "ok"
    bal_alarm = [{"obs_date": d(2), "risk_type": "water_balance_deficit", "triage_status": "confirmed"}]
    assert wo.assess_plants(obs, bal_alarm, TODAY)["status"] == "ok"   # alarm z modelu ≠ satelita


def test_topsoil():
    assert wo.assess_topsoil({"calibrated": False}, TODAY)["code"] == "topsoil_off"
    opt = {"calibrated": True, "params": {"quality": "good"}, "era5": {"n": 20, "r": 0.5},
           "current": {"obs_date": d(3), "w": 0.22}}
    assert wo.assess_topsoil(opt, TODAY)["code"] == "topsoil_dry"
    opt["era5"]["r"] = -0.4
    assert wo.assess_topsoil(opt, TODAY)["code"] == "topsoil_unreliable"
    opt["era5"]["r"] = 0.5; opt["current"]["obs_date"] = d(30)
    assert wo.assess_topsoil(opt, TODAY)["code"] == "topsoil_old"


def test_canopy_temp():
    row = {"scene_date": d(5), "tsi": 0.6, "ndvi": 0.7, "t_field": 36.0, "t_cold": 30.0}
    assert wo.assess_canopy_temp([row], TODAY)["status"] == "bad"
    assert wo.assess_canopy_temp([{**row, "tsi": 0.3}], TODAY)["code"] == "temp_warm"
    assert wo.assess_canopy_temp([{**row, "tsi": 0.1}], TODAY)["params"]["dt"] == 6.0
    assert wo.assess_canopy_temp([{**row, "ndvi": 0.2}], TODAY)["code"] == "temp_low_cover"
    assert wo.assess_canopy_temp([{**row, "scene_date": d(30)}], TODAY)["code"] == "temp_old"


def test_weather():
    assert wo.assess_weather(bal(forecast={"days": 7, "precip_mm": 0, "et0_mm": 50, "tmax_c": 40,
                                           "dry_hot": True}))["code"] == "weather_dry_hot"
    assert wo.assess_weather(bal(cwb_30d_pct=5))["code"] == "weather_dry_month"
    assert wo.assess_weather(bal(forecast={"days": 7, "precip_mm": 25, "et0_mm": 20}))["code"] == "weather_rain"


# --- ocena ogólna ------------------------------------------------------------------

def _src(soil="ok", plants="ok", top="none", temp="none", w=0.5):
    S = wo._src
    return [S("soil", soil, "x"), S("plants", plants, "x"), S("topsoil", top, "x", w=w),
            S("canopy_temp", temp, "x"), S("weather", "ok", "x")]


def test_overall_poziomy():
    irr = {"irrigation_mode": "irrigation"}
    assert wo.overall(_src(), irr, bal())["code"] == "overall_good"
    assert wo.overall(_src(plants="warn"), irr, bal())["level"] == "warn"
    assert wo.overall(_src(temp="bad"), irr, bal())["code"] == "overall_possible"
    o = wo.overall(_src(soil="bad", temp="warn"), irr,
                   bal(status="stress", suggested_irrigation_date=TODAY.isoformat(), suggested_dose_mm=55,
                       suggested_dose_m3_ha=550))
    assert o["code"] == "overall_confirmed" and o["action"] == {"code": "action_irrigate", "date": TODAY.isoformat(),
                                                                "mm": 55, "m3": 550}
    assert wo.overall(_src(soil="none", plants="none"), irr, None)["code"] == "overall_no_data"
    assert wo.overall(_src(temp="bad"), {"irrigation_mode": "rainfed"}, bal())["action"]["code"] == "action_rainfed"


def test_sprzecznosc_model_sucho_satelita_mokro():
    o = wo.overall(_src(soil="bad", top="ok", w=0.96), {"irrigation_mode": "irrigation"},
                   bal(status="stress", suggested_irrigation_date=TODAY.isoformat(), suggested_dose_mm=68))
    assert o["code"] == "overall_conflict" and o["level"] == "warn"
    assert o["action"]["code"] == "action_log_water" and o["conflicts"][0]["code"] == "conflict_unlogged_water"


def test_sprzecznosc_woda_ok_rosliny_zle():
    o = wo.overall(_src(soil="ok", temp="bad"), {"irrigation_mode": "irrigation"}, bal())
    assert {"code": "conflict_not_water"} in o["conflicts"]


# --- całość z bazy -----------------------------------------------------------------

def test_build_overview_z_bazy(isolated_db, monkeypatch):
    import water_balance
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-9", "irrigation", WKT)
    days = [{"date": d(k), "depletion_mm": 10 + k * 0.5, "p_eff_mm": 5 if k == 10 else 0,
             "irrigation_mm": 40 if k == 30 else 0, "is_forecast": False} for k in range(150, -1, -1)]
    days += [{"date": d(-k), "depletion_mm": 12 + k, "p_eff_mm": 0, "irrigation_mm": 0, "is_forecast": True}
             for k in range(1, 8)]
    monkeypatch.setattr(water_balance, "compute_field_balance",
                        lambda f, today=None: {"summary": bal(), "days": days})
    isolated_db.add_water_observation(fid, d(3), ndvi_mean=0.72, ndwi_mean=0.3)
    isolated_db.upsert_thermal(fid, [{"scene_date": d(5), "t_field": 31, "t_cold": 30, "t_hot": 50,
                                      "tsi": 0.05, "ndvi": 0.7}])
    isolated_db.add_irrigation_event(fid, d(12), "kroplowe")          # bez dawki — znacznik
    aid = isolated_db.add_water_alarm(fid, d(4), "water_stress_risk", "m", "nk")
    isolated_db.set_water_alarm_triage(aid, "rejected")
    o = wo.build_overview(isolated_db.get_water_field_by_id(fid), today=TODAY)
    assert o["overall"]["code"] == "overall_good"
    tl = o["timeline"]
    assert tl["threshold_pct"] == 50.0 and tl["soil"][0]["date"] >= tl["start"]
    assert any(p["forecast"] for p in tl["soil"]) and tl["canopy_cooling"] == [{"date": d(5), "v": 95.0, "t_field": 31.0, "t_cold": 30.0, "t_hot": 50.0}]
    assert {p["date"] for p in tl["irrigation"]} == {d(30), d(12)} and tl["alarms"] == []
    assert tl["greenness"] == [{"date": d(3), "v": 72.0}]


def test_api_overview(app_client, isolated_db, monkeypatch):
    import water_balance
    fid = isolated_db.add_water_field("UZ-9", "irrigation", WKT)

    def boom(f, today=None):
        raise RuntimeError("Open-Meteo 429")
    monkeypatch.setattr(water_balance, "compute_field_balance", boom)
    r = app_client.get(f"/api/water/field/{fid}/overview")
    assert r.status_code == 200
    j = r.json()
    assert j["balance_error"] and j["sources"][0]["code"] == "soil_no_data"
    assert j["overall"]["code"] == "overall_no_data"


def test_sprzecznosc_z_termika_i_brak_dziennika(isolated_db, monkeypatch):
    import water_balance
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-10", "irrigation", WKT)
    monkeypatch.setattr(water_balance, "compute_field_balance",
                        lambda f, today=None: {"summary": bal(status="stress", depletion_mm=80, depletion_frac=0.8),
                                               "days": []})
    isolated_db.upsert_thermal(fid, [{"scene_date": d(3), "t_field": 30, "t_cold": 29.5, "t_hot": 50,
                                      "tsi": 0.05, "ndvi": 0.7}])
    o = wo.build_overview(isolated_db.get_water_field_by_id(fid), today=TODAY)
    assert o["sources"][0]["params"]["no_irrigation_log"] is True
    assert o["overall"]["code"] == "overall_conflict"
    assert {"code": "conflict_unlogged_water_temp"} in o["overall"]["conflicts"]
    isolated_db.add_irrigation_event(fid, d(5), "zalewowe", amount_mm=80)
    o = wo.build_overview(isolated_db.get_water_field_by_id(fid), today=TODAY)
    assert "no_irrigation_log" not in o["sources"][0]["params"]


# --- „Przygotuj / odśwież wszystko” --------------------------------------------------

BIG_WKT = "POLYGON((64.40 40.20, 64.43 40.20, 64.43 40.22, 64.40 40.22, 64.40 40.20))"   # ~570 ha
SMALL_WKT = "POLYGON((64.40 40.20, 64.401 40.20, 64.401 40.201, 64.40 40.201, 64.40 40.20))"


def test_plan_nowe_pole_wszystko(isolated_db):
    isolated_db.init_db()
    f = isolated_db.get_water_field_by_id(isolated_db.add_water_field("N", "irrigation", BIG_WKT))
    plan = wo.prepare_plan(f, today=TODAY)
    run = {s["step"]: s["run"] for s in plan["steps"]}
    assert run == {"backfill": True, "optram": True, "thermal": True, "check_now": True}
    # historia od 1 marca: 141 dni zamiast 120 -> 3 PU · 141/120
    assert plan["total_pu"] == pytest.approx(3.5 + 12.5 + 2.5 + 2 + 1.7 + 2.5, abs=0.1)


def test_plan_pole_gotowe_tylko_sprawdzenie(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("N", "irrigation", BIG_WKT)
    for k in range(8):
        isolated_db.add_water_observation(fid, d(5 * k), ndvi_mean=0.6, ndwi_mean=0.2)
    isolated_db.add_water_observation(fid, "2026-03-05", ndvi_mean=0.3, ndwi_mean=0.1)   # początek sezonu
    isolated_db.save_optram_params(fid, {"i_d": 0, "s_d": 1, "i_w": 2, "s_w": 5, "quality": "good"})
    isolated_db.upsert_thermal(fid, [{"scene_date": d(3), "t_field": 30, "tsi": 0.1, "ndvi": 0.6}])
    plan = wo.prepare_plan(isolated_db.get_water_field_by_id(fid), today=TODAY)
    assert [s["step"] for s in plan["steps"] if s["run"]] == ["check_now"]
    assert plan["total_pu"] == pytest.approx(2 + 1.7 + 2.5)


def test_plan_male_pole_bez_termiki(isolated_db):
    isolated_db.init_db()
    f = isolated_db.get_water_field_by_id(isolated_db.add_water_field("S", "irrigation", SMALL_WKT))
    th = next(s for s in wo.prepare_plan(f)["steps"] if s["step"] == "thermal")
    assert th["run"] is False and th["reason"] == "small_field"


def test_kroki_wywoluja_moduly_i_lapia_bledy(isolated_db, monkeypatch):
    import optram, thermal, water_replay, water_stress
    isolated_db.init_db()
    f = isolated_db.get_water_field_by_id(isolated_db.add_water_field("N", "irrigation", BIG_WKT))
    monkeypatch.setattr(water_replay, "backfill_field", lambda fl, a, b: {"stored": 9, "clean": 11})
    monkeypatch.setattr(optram, "calibrate_field", lambda fl, a, b: {"params": {"quality": "good"},
                                                                    "series": [1, 2], "era5": {"r": 0.4}})
    monkeypatch.setattr(thermal, "fetch_field", lambda fl, a, b: [1, 2, 3])
    monkeypatch.setattr(water_stress, "check_field_now", lambda fl: {"fresh_observation_fetched": True})
    assert wo.run_prepare_step(f, "backfill")["stored"] == 9
    assert wo.run_prepare_step(f, "optram")["quality"] == "good"
    assert wo.run_prepare_step(f, "thermal")["n_scenes"] == 3
    assert wo.run_prepare_step(f, "check_now")["fresh"] is True

    def boom(*a):
        raise RuntimeError("Sentinel Hub 429")
    monkeypatch.setattr(thermal, "fetch_field", boom)
    r = wo.run_prepare_step(f, "thermal")
    assert r["ok"] is False and "429" in r["error"]


def test_api_prepare(app_client, isolated_db, monkeypatch):
    import water_stress
    fid = isolated_db.add_water_field("N", "irrigation", BIG_WKT)
    assert app_client.get(f"/api/water/field/{fid}/prepare_plan").json()["total_pu"] > 0
    monkeypatch.setattr(water_stress, "check_field_now", lambda fl: {"fresh_observation_fetched": False})
    r = app_client.post(f"/api/water/field/{fid}/prepare_step", json={"step": "check_now"})
    assert r.status_code == 200 and r.json()["ok"]
    assert app_client.post(f"/api/water/field/{fid}/prepare_step", json={"step": "rm -rf"}).status_code == 422



def test_sama_pogoda_nie_zmienia_oceny():
    src = _src()
    src[4] = wo._src("weather", "warn", "weather_dry_hot")
    o = wo.overall(src, {"irrigation_mode": "irrigation"}, bal())
    assert o["level"] == "ok" and o["action"]["code"] == "action_weather_watch"
