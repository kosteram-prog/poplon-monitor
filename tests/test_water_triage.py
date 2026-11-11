"""Testy Kroku 3 planu modułu teledetekcji i wody: kolejka triage
(water_alarms), water_stress.record_alarm_if_risky i scheduler.
check_all_water_fields. Wzorem test_db.py / test_api.py / test_scheduler.py."""
from datetime import date, timedelta

import pytest

import water_stress as ws

SAMPLE_WKT = "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))"


def _risky_verdict(obs_date=None):
    return {
        "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
        "risk_flag": True,
        "risk_type": "water_stress_risk",
        "message_pl": "Pole X wykazuje deficyt wilgoci.",
        "components": {"own_ndwi": {"current": 0.05}, "own_vv": None, "cohort_ndwi": None},
        "obs_date": obs_date or date.today().isoformat(),
    }


# --- warstwa bazy: water_alarms --------------------------------------------

def test_add_water_alarm_i_odczyt(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    aid = isolated_db.add_water_alarm(fid, "2026-06-01", "water_stress_risk",
                                      "test message", "niekalibrowane",
                                      ndwi_value=0.05, components_json="{}")
    alarm = isolated_db.get_water_alarm_by_id(aid)
    assert alarm["risk_type"] == "water_stress_risk"
    assert alarm["triage_status"] == "pending"


def test_get_water_alarms_filtruje_po_statusie(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    aid = isolated_db.add_water_alarm(fid, "2026-06-01", "water_stress_risk", "m", "niekalibrowane")
    assert len(isolated_db.get_water_alarms(triage_status="pending")) == 1
    isolated_db.set_water_alarm_triage(aid, "confirmed", triage_note="ok", triage_by="Marek")
    assert isolated_db.get_water_alarms(triage_status="pending") == []
    confirmed = isolated_db.get_water_alarms(triage_status="confirmed")
    assert len(confirmed) == 1
    assert confirmed[0]["triage_by"] == "Marek"
    assert confirmed[0]["triaged_at"] is not None


def test_set_water_alarm_triage_odrzuca_nieznany_status(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    aid = isolated_db.add_water_alarm(fid, "2026-06-01", "water_stress_risk", "m", "niekalibrowane")
    with pytest.raises(ValueError):
        isolated_db.set_water_alarm_triage(aid, "zaakceptowane_domyslnie")


def test_water_alarm_exists_recent(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    assert isolated_db.water_alarm_exists_recent(fid, "water_stress_risk") is False
    isolated_db.add_water_alarm(fid, "2026-06-01", "water_stress_risk", "m", "niekalibrowane")
    assert isolated_db.water_alarm_exists_recent(fid, "water_stress_risk") is True
    # inny risk_type -> nie liczy sie jako "ten sam" alarm
    assert isolated_db.water_alarm_exists_recent(fid, "irrigation_performance_anomaly") is False


# --- water_stress.record_alarm_if_risky ------------------------------------

def test_record_alarm_if_risky_nic_nie_zapisuje_gdy_brak_ryzyka(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    field = isolated_db.get_water_field_by_id(fid)
    verdict = {"risk_flag": False, "risk_type": None}
    assert ws.record_alarm_if_risky(field, verdict) is None
    assert isolated_db.get_water_alarms(fid) == []


def test_record_alarm_if_risky_zapisuje_nowy_alarm(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    field = isolated_db.get_water_field_by_id(fid)
    alarm_id = ws.record_alarm_if_risky(field, _risky_verdict())
    assert alarm_id is not None
    alarms = isolated_db.get_water_alarms(fid)
    assert len(alarms) == 1
    assert alarms[0]["risk_type"] == "water_stress_risk"
    assert alarms[0]["ndwi_value"] == 0.05


def test_record_alarm_if_risky_nie_duplikuje_swiezego_alarmu(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    field = isolated_db.get_water_field_by_id(fid)
    first = ws.record_alarm_if_risky(field, _risky_verdict())
    second = ws.record_alarm_if_risky(field, _risky_verdict())
    assert first is not None
    assert second is None
    assert len(isolated_db.get_water_alarms(fid)) == 1


# --- scheduler.check_all_water_fields (zamockowana siec) -------------------

def test_check_all_water_fields_tworzy_alarm_dla_ryzykownego_pola(isolated_db, monkeypatch):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)

    import water_stress
    monkeypatch.setattr(water_stress, "fetch_and_store_observation", lambda field: {"obs_date": "x"})
    monkeypatch.setattr(water_stress, "build_verdict_for_field", lambda field: _risky_verdict())

    import scheduler
    scheduler.check_all_water_fields()

    alarms = isolated_db.get_water_alarms(fid)
    assert len(alarms) == 1
    assert alarms[0]["triage_status"] == "pending"


def test_check_all_water_fields_pomija_pole_gdy_fetch_sie_wywala(isolated_db, monkeypatch):
    isolated_db.init_db()
    isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)

    import water_stress

    def _boom(field):
        raise RuntimeError("Sentinel Hub timeout")
    monkeypatch.setattr(water_stress, "fetch_and_store_observation", _boom)

    import scheduler
    scheduler.check_all_water_fields()  # nie ma rzucic wyjatku na zewnatrz

    assert isolated_db.get_water_alarms() == []


# --- warstwa API: triage_queue / triage --------------------------------

def test_api_triage_queue_i_triage_flow(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "AAA", "irrigation_mode": "rainfed", "geometry_wkt": SAMPLE_WKT,
    })
    field_id = r.json()["id"]

    import db as dbmod
    alarm_id = dbmod.add_water_alarm(field_id, "2026-06-01", "water_stress_risk",
                                     "test", "niekalibrowane / wymaga walidacji terenowej")

    r2 = app_client.get("/api/water/triage_queue")
    assert r2.status_code == 200
    assert len(r2.json()) == 1
    assert r2.json()[0]["id"] == alarm_id

    r3 = app_client.post(f"/api/water/alarm/{alarm_id}/triage", json={
        "triage_status": "confirmed", "triage_note": "potwierdzone w terenie", "triage_by": "Marek",
    })
    assert r3.status_code == 200

    r4 = app_client.get("/api/water/triage_queue")
    assert r4.json() == []  # domyslnie tylko pending

    r5 = app_client.get("/api/water/triage_queue?status=confirmed")
    assert len(r5.json()) == 1
    assert r5.json()[0]["triage_by"] == "Marek"


def test_api_triage_odrzuca_zly_status(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "AAA", "irrigation_mode": "rainfed", "geometry_wkt": SAMPLE_WKT,
    })
    field_id = r.json()["id"]
    import db as dbmod
    alarm_id = dbmod.add_water_alarm(field_id, "2026-06-01", "water_stress_risk", "t", "niekalibrowane")

    r2 = app_client.post(f"/api/water/alarm/{alarm_id}/triage", json={"triage_status": "cos_innego"})
    assert r2.status_code == 422


def test_api_triage_dla_nieistniejacego_alarmu_zwraca_404(app_client):
    r = app_client.post("/api/water/alarm/999/triage", json={"triage_status": "confirmed"})
    assert r.status_code == 404
    r2 = app_client.get("/api/water/alarm/999")
    assert r2.status_code == 404
