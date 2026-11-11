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


@pytest.fixture
def irr_field(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZB-1", "irrigation", SAMPLE_WKT, region="R", crop_type="bawełna")
    return isolated_db.get_water_field_by_id(fid)


def test_replay_z_bilansem_generuje_alarm_bilansu_gdy_satelita_milczy(irr_field, monkeypatch):
    field = irr_field
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources())
    wr.backfill_field(field, T0, T0 + timedelta(days=40))
    dry = [{"date": T0 - timedelta(days=1) + timedelta(days=i), "precip_mm": 0.0, "et0_mm": 5.0}
           for i in range(60)]
    res = wr.replay_field(field, weather_series=dry)
    types = {a["risk_type"] for a in res["alarms"]}
    assert ws.BALANCE_RISK_TYPE in types
    assert res["summary"]["stress_days_in_balance"] > 0
    assert res["stress_periods"] and res["stress_periods"][0]["from"] <= res["stress_periods"][0]["to"]
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
    # pole deszczowe: bilans to kontekst, bez alarmu (nie ma czego podlać)
    rain = {"parcel_id": "PL-1", "irrigation_mode": "rainfed"}
    assert ws.balance_alarm_decision(rain, {"risk_flag": False, "water_balance": bal}) is None


def test_scheduler_tworzy_alarm_bilansu_gdy_satelita_milczy(irr_field, isolated_db, monkeypatch):
    field = irr_field
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


# --- oceny powtórki i kalibracja progu z obu rodzajów błędów -----------------

def _rating(z, rating, rt="water_stress_risk", **m):
    return {"water_field_id": 1, "obs_date": "2026-07-01", "risk_type": rt, "rating": rating,
            "metrics_json": json.dumps({"z_ndmi": z, **m})}


def test_kalibracja_za_malo_ocen():
    r = wr.suggest_threshold_from_ratings([_rating(-2.0, "confirmed"), _rating(-1.6, "rejected")])
    assert r["status"] == "not_enough_data" and r["positives"] == 1


def test_kalibracja_zaostrza_gdy_falszywe_alarmy_tuz_pod_progiem():
    ratings = [_rating(z, "confirmed") for z in (-3.5, -3.0, -2.8)] + \
              [_rating(z, "rejected") for z in (-1.6, -1.7, -1.9)]
    r = wr.suggest_threshold_from_ratings(ratings)
    assert r["status"] == "ok" and r["suggested_threshold"] <= -2.0
    assert r["at_suggested"]["fp"] == 0 and r["at_suggested"]["tp"] == 3
    assert r["at_current"]["fp"] == 3


def test_kalibracja_luzuje_gdy_przeoczone_problemy_powyzej_progu():
    ratings = [_rating(z, "confirmed") for z in (-2.5, -2.2)] + [_rating(-1.2, "missed"), _rating(-1.1, "missed", rt="")] + \
              [_rating(z, "rejected") for z in (-0.3, 0.2, 0.5)]
    r = wr.suggest_threshold_from_ratings(ratings)
    assert r["suggested_threshold"] > ws.Z_SCORE_RISK_THRESHOLD      # łagodniej niż -1.5
    assert r["at_suggested"]["fn"] == 0 and r["at_current"]["fn"] == 2


def test_kalibracja_wskazuje_przeoczenia_zablokowane_inna_regula():
    ratings = [_rating(-3.0, "missed", rt="", canopy_present=False)] + \
              [_rating(z, "confirmed") for z in (-2.5, -2.2)] + [_rating(z, "rejected") for z in (-0.5, 0.1, 0.4)]
    r = wr.suggest_threshold_from_ratings(ratings)
    assert len(r["blocked_by_other_rules"]) == 1 and r["blocked_by_other_rules"][0]["canopy_present"] is False


def test_kalibracja_pomija_alarmy_rownomiernosci_i_bilansu():
    ratings = [_rating(-3.0, "rejected", rt="uneven_moisture_risk")] * 5 + \
              [_rating(-3.0, "rejected", rt=ws.BALANCE_RISK_TYPE)] * 5
    assert wr.suggest_threshold_from_ratings(ratings)["negatives"] == 0


def test_rate_replay_zapisuje_metryki_dnia(field, isolated_db, monkeypatch):
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources(drop_at=9))
    wr.backfill_field(field, T0, T0 + timedelta(days=40))
    day = (T0 + timedelta(days=27)).isoformat()
    row = wr.rate_replay(field, day, "confirmed", risk_type="water_stress_risk")
    m = json.loads(row["metrics_json"])
    assert row["rating"] == "confirmed" and m["z_ndmi"] < -1.5 and m["risk_flag"] is True
    # ponowna ocena tej samej pozycji nadpisuje
    assert wr.rate_replay(field, day, "rejected", risk_type="water_stress_risk")["rating"] == "rejected"
    assert len(isolated_db.get_replay_ratings(field["id"])) == 1
    # przeoczony problem: najbliższa obserwacja w ±3 dni
    missed = wr.rate_replay(field, (T0 + timedelta(days=4)).isoformat(), "missed")
    assert missed["obs_date"] == (T0 + timedelta(days=6)).isoformat() or missed["obs_date"] == (T0 + timedelta(days=3)).isoformat()
    with pytest.raises(ValueError):
        wr.rate_replay(field, "2025-01-01", "missed")


def test_api_backfill_zakres_dat_i_oceny(app_client, monkeypatch):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "API-2", "irrigation_mode": "rainfed", "geometry_wkt": SAMPLE_WKT})
    fid = r.json()["id"]
    seen = {}

    def fake_sources(f, a, b):
        seen["range"] = (a, b)
        return _sources(drop_at=9)
    monkeypatch.setattr(ws, "fetch_sources", fake_sources)
    r = app_client.post(f"/api/water/field/{fid}/backfill", json={"date_from": "2026-04-01", "date_to": "2026-09-30"})
    assert r.status_code == 200 and seen["range"] == (date(2026, 4, 1), date(2026, 9, 30))
    assert r.json()["date_from"] == "2026-04-01"
    assert app_client.post(f"/api/water/field/{fid}/backfill", json={"date_from": "2025-01-01", "date_to": "2026-09-30"}).status_code == 422
    assert app_client.post(f"/api/water/field/{fid}/backfill", json={"date_from": "2026-09-30", "date_to": "2026-04-01"}).status_code == 422
    day = (T0 + timedelta(days=27)).isoformat()
    r = app_client.post(f"/api/water/field/{fid}/replay_rating", json={"obs_date": day, "rating": "confirmed",
                                                                       "risk_type": "water_stress_risk"})
    assert r.status_code == 200
    rid = r.json()["id"]
    assert app_client.get("/api/water/replay_ratings", params={"water_field_id": fid}).json()[0]["id"] == rid
    assert app_client.get("/api/water/calibration/suggest_from_replay").json()["status"] == "not_enough_data"
    assert app_client.post(f"/api/water/field/{fid}/replay_rating", json={"obs_date": day, "rating": "zle"}).status_code == 422
    assert app_client.delete(f"/api/water/replay_rating/{rid}").status_code == 200



def test_przeoczenie_w_dniu_alarmu_odrzucone_i_nie_nadpisuje_oceny(field, isolated_db, monkeypatch):
    monkeypatch.setattr(ws, "fetch_sources", lambda f, a, b: _sources(drop_at=9))
    wr.backfill_field(field, T0, T0 + timedelta(days=40))
    day = (T0 + timedelta(days=27)).isoformat()
    wr.rate_replay(field, day, "confirmed", risk_type="water_stress_risk")
    with pytest.raises(ValueError, match="trafny"):
        wr.rate_replay(field, day, "missed")
    ok_day = (T0 + timedelta(days=12)).isoformat()
    m = wr.rate_replay(field, ok_day, "missed")
    assert m["risk_type"] == ""
    ratings = {(r["obs_date"], r["rating"]) for r in isolated_db.get_replay_ratings(field["id"])}
    assert (day, "confirmed") in ratings and (ok_day, "missed") in ratings
