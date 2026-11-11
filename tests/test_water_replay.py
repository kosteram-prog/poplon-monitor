"""Testy uzupełniania historii i powtórki sezonu (water_replay.py) oraz
alarmu z modelu bilansu wodnego (water_stress.balance_alarm_decision,
scheduler.check_all_water_fields). Sieć zamockowana."""
import json
import os
from datetime import date, timedelta

import pytest

import water_replay as wr
import water_stress as ws

SAMPLE_WKT = "POLYGON((17.0 51.1, 17.004 51.1, 17.004 51.103, 17.0 51.103, 17.0 51.1))"
T0 = date(2026, 6, 1)


def _optical(n=12, step=3, drop_at=None):
    """Sceny co `step` dni; od indeksu drop_at NDMI spada przy stabilnym NDVI."""
    out = []
    for i in range(n):
        ndmi = 0.30 + (0.01 if i % 2 else -0.01)
        if drop_at is not None and i >= drop_at:
            ndmi = 0.18
        out.append({"date": T0 + timedelta(days=i * step), "ndvi": 0.80, "ndwi": ndmi,
                    "valid_frac": 0.95 if i != 3 else 0.30,   # scena 3 zachmurzona
                    "ndwi_p10": ndmi - 0.04, "ndwi_p50": ndmi, "ndwi_low_frac": 0.03})
    return out


def _sources(drop_at=None):
    opt = _optical(drop_at=drop_at)
    radar = [{"date": o["date"] + timedelta(days=1), "vv_db": -11.0, "vh_db": -17.0, "rvi": 0.5}
             for o in opt]
    orbits = {r["date"]: 22 for r in radar}
    weather = [{"date": T0 - timedelta(days=40) + timedelta(days=i), "precip_mm": 1.0, "et0_mm": 3.0}
               for i in range(120)]
    return {"optical": opt, "radar": radar, "orbits": orbits, "weather": weather,
            "dry_dates": set()}


@pytest.fixture
def field(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT, region="R", crop_type="pszenica")
    return isolated_db.get_water_field_by_id(fid)


def test_backfill_zapisuje_tylko_czyste_sceny_i_jest_idempotentny(field, isolated_db, monkeypatch):
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources())
    st = wr.backfill_field(field, T0, T0 + timedelta(days=40))
    assert st["scenes"] == 12 and st["clean"] == 11 and st["stored"] == 11
    obs = isolated_db.get_water_observations(field["id"])
    assert len(obs) == 11
    assert obs[0]["s1_rel_orbit"] == 22 and obs[0]["cwb_14d_mm"] == -28.0
    assert wr.backfill_field(field, T0, T0 + timedelta(days=40))["stored"] == 0


def test_replay_wykrywa_spadek_bez_zapisu_do_kolejki(field, isolated_db, monkeypatch):
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources(drop_at=9))
    wr.backfill_field(field, T0, T0 + timedelta(days=40))
    res = wr.replay_field(field)
    flagged = [r for r in res["rows"] if r["risk_flag"]]
    assert flagged and flagged[0]["obs_date"] == (T0 + timedelta(days=27)).isoformat()
    # karencja: kolejne dni spadku nie generują nowego alarmu tego samego typu
    assert res["summary"]["alarms_total"] == 1
    assert res["alarms"][0]["risk_type"] == "water_stress_risk"
    assert isolated_db.get_water_alarms() == []


def test_replay_brak_alarmow_gdy_stabilnie(field, monkeypatch):
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources())
    wr.backfill_field(field, T0, T0 + timedelta(days=40))
    res = wr.replay_field(field)
    assert res["summary"]["alarms_total"] == 0
    assert res["summary"]["observations"] == 11


def test_replay_z_bilansem_generuje_alarm_bilansu_gdy_satelita_milczy(field, monkeypatch):
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources())
    wr.backfill_field(field, T0, T0 + timedelta(days=40))
    dry = [{"date": T0 - timedelta(days=1) + timedelta(days=i), "precip_mm": 0.0, "et0_mm": 5.0}
           for i in range(60)]
    res = wr.replay_field(field, weather_series=dry)
    types = {a["risk_type"] for a in res["alarms"]}
    assert ws.BALANCE_RISK_TYPE in types
    assert res["summary"]["stress_days_in_balance"] > 0
    assert any(r["balance_status"] == "stress" for r in res["rows"])


# --- decyzja o alarmie z bilansu -----------------------------------------

def test_balance_alarm_decision():
    f = {"parcel_id": "UZB-1", "irrigation_mode": "irrigation"}
    bal = {"status": "approaching", "days_to_stress": 2, "suggested_irrigation_date": "2026-07-03",
           "suggested_dose_mm": 60, "suggested_dose_m3_ha": 600, "depletion_mm": 50, "raw_mm": 58}
    d = ws.balance_alarm_decision(f, {"risk_flag": False, "water_balance": bal})
    assert d["risk_type"] == ws.BALANCE_RISK_TYPE and "600 m³/ha" in d["message_pl"]
    assert ws.balance_alarm_decision(f, {"risk_flag": True, "water_balance": bal}) is None
    far = dict(bal, days_to_stress=8)
    assert ws.balance_alarm_decision(f, {"risk_flag": False, "water_balance": far}) is None
    rain = {"parcel_id": "PL-1", "irrigation_mode": "rainfed"}
    assert "m³/ha" not in ws.balance_alarm_decision(rain, {"risk_flag": False, "water_balance": bal})["message_pl"]


def test_scheduler_tworzy_alarm_bilansu_gdy_satelita_milczy(field, isolated_db, monkeypatch):
    monkeypatch.setattr(ws, "fetch_and_store_observation", lambda f: None)
    monkeypatch.setattr(ws, "build_verdict_for_field", lambda f: {
        "risk_flag": False, "risk_type": None, "message_pl": "ok", "obs_date": "2026-07-01",
        "calibration_status": "niekalibrowane"})
    import water_balance
    monkeypatch.setattr(water_balance, "compute_field_balance", lambda f: {"summary": {
        "status": "stress", "days_to_stress": 0, "depletion_mm": 70, "raw_mm": 55,
        "suggested_irrigation_date": "2026-07-01", "suggested_dose_mm": 70, "suggested_dose_m3_ha": 700}})
    import scheduler
    scheduler.check_all_water_fields()
    scheduler.check_all_water_fields()   # drugi przebieg: karencja, bez duplikatu
    alarms = isolated_db.get_water_alarms(field["id"])
    assert len(alarms) == 1 and alarms[0]["risk_type"] == ws.BALANCE_RISK_TYPE


def test_alarm_bilansu_nie_wchodzi_do_kalibracji(isolated_db, monkeypatch):
    comps = json.dumps({"own_ndwi": {"z_score": -3.0}})
    alarms = [{"triage_status": "confirmed", "risk_type": ws.BALANCE_RISK_TYPE,
               "components_json": comps}] * 20
    monkeypatch.setattr(isolated_db, "get_water_alarms", lambda **kw: alarms)
    assert ws.compute_suggested_water_threshold(min_samples=5) is None


# --- wczytywanie pól i CLI ------------------------------------------------

def test_load_fields_geojson_z_reprojekcja_2180(tmp_path):
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2180"}},
          "features": [{"type": "Feature",
                        "properties": {"id": "LENS-07", "crop_type": "pszenica", "soil_taw_mm": "120"},
                        "geometry": {"type": "Polygon", "coordinates": [[
                            [358000, 360000], [358200, 360000], [358200, 360200],
                            [358000, 360200], [358000, 360000]]]}}]}
    p = tmp_path / "pola.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    fields = wr.load_fields(str(p))
    assert fields[0]["parcel_id"] == "LENS-07" and fields[0]["soil_taw_mm"] == 120.0
    from shapely import wkt
    c = wkt.loads(fields[0]["geometry_wkt"]).centroid
    assert 14 < c.x < 24 and 49 < c.y < 55            # przeliczone do WGS84 (Polska)


def test_load_fields_csv(tmp_path):
    p = tmp_path / "pola.csv"
    p.write_text("parcel_id,irrigation_mode,region,wkt\nUZB-1,irygacyjny,Samarkand,"
                 "\"POLYGON((66.9 39.6, 66.91 39.6, 66.91 39.61, 66.9 39.61, 66.9 39.6))\"\n",
                 encoding="utf-8")
    f = wr.load_fields(str(p))[0]
    assert f["irrigation_mode"] == "irrigation" and f["region"] == "Samarkand"


def test_cli_end_to_end(tmp_path, monkeypatch):
    p = tmp_path / "pola.csv"
    p.write_text(f"parcel_id,region,wkt\nA,R,\"{SAMPLE_WKT}\"\nB,R,\"{SAMPLE_WKT}\"\n", encoding="utf-8")
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources(drop_at=9 if f["parcel_id"] == "A" else None))
    import weather
    monkeypatch.setattr(weather, "fetch_weather_series", lambda g, a, b: [])
    out = tmp_path / "wyniki"
    rc = wr.run_cli([str(p), "--od", "2026-06-01", "--do", "2026-07-15",
                     "--baza", str(tmp_path / "r.db"), "--wyniki", str(out)])
    assert rc == 0
    alarmy = (out / "alarmy.csv").read_text(encoding="utf-8-sig")
    assert "water_stress_risk" in alarmy and alarmy.count("\n") == 2   # nagłówek + 1 alarm (pole A)
    assert "Razem alarmów: 1 na 2 pól." in (out / "podsumowanie.txt").read_text(encoding="utf-8")


def test_cli_odmawia_pracy_na_produkcyjnej_bazie(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "pola.csv"
    p.write_text(f"parcel_id,wkt\nA,\"{SAMPLE_WKT}\"\n", encoding="utf-8")
    assert wr.run_cli([str(p), "--od", "2026-06-01", "--do", "2026-06-30", "--baza", "monitoring.db"]) == 2


# --- API --------------------------------------------------------------------

def test_api_backfill_i_replay(app_client, monkeypatch):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "API-1", "irrigation_mode": "rainfed", "geometry_wkt": SAMPLE_WKT})
    fid = r.json()["id"]
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources(drop_at=9))
    r = app_client.post(f"/api/water/field/{fid}/backfill", json={"days": 60})
    assert r.status_code == 200 and r.json()["stored"] == 11
    r = app_client.get(f"/api/water/field/{fid}/replay")
    assert r.status_code == 200 and r.json()["summary"]["alarms_total"] == 1
    assert app_client.post(f"/api/water/field/{fid}/backfill", json={"days": 3}).status_code == 422
