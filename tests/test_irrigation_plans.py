"""Testy planowania nawadniania (sekcja 5.2 planu, "faza 2") — kalendarz
planowanych nawodnień per pole, oznaczanie "wykonano"/"pominięto".
Wzorem test_irrigation_log.py: warstwa bazy (isolated_db) i warstwa API
(app_client)."""
import pytest


def _mk_field(db, mode="irrigation"):
    return db.add_water_field(
        "A", mode,
        "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))")


# --- warstwa bazy ------------------------------------------------------

def test_add_irrigation_plan_odrzuca_nieznana_metode(isolated_db):
    isolated_db.init_db()
    fid = _mk_field(isolated_db)
    with pytest.raises(ValueError):
        isolated_db.add_irrigation_plan(fid, "2026-10-01", method="sprinkler_x")


def test_add_irrigation_plan_metoda_jest_opcjonalna(isolated_db):
    isolated_db.init_db()
    fid = _mk_field(isolated_db)
    plan_id = isolated_db.add_irrigation_plan(fid, "2026-10-01")
    plan = isolated_db.get_irrigation_plan_by_id(plan_id)
    assert plan["method"] is None
    assert plan["status"] == "planned"


def test_get_irrigation_plans_filtruje_po_polu_i_statusie(isolated_db):
    isolated_db.init_db()
    fid1 = _mk_field(isolated_db)
    fid2 = _mk_field(isolated_db)
    p1 = isolated_db.add_irrigation_plan(fid1, "2026-10-05", method="kroplowe")
    isolated_db.add_irrigation_plan(fid1, "2026-10-01", method="zalewowe")
    isolated_db.add_irrigation_plan(fid2, "2026-10-03")
    isolated_db.set_irrigation_plan_status(p1, "done")

    all_f1 = isolated_db.get_irrigation_plans(water_field_id=fid1)
    assert len(all_f1) == 2
    # ORDER BY planned_date ASC
    assert all_f1[0]["planned_date"] == "2026-10-01"

    planned_only = isolated_db.get_irrigation_plans(water_field_id=fid1, status="planned")
    assert len(planned_only) == 1
    assert planned_only[0]["planned_date"] == "2026-10-01"

    assert len(isolated_db.get_irrigation_plans(status="planned")) == 2


def test_set_irrigation_plan_status_odrzuca_nieznany_status(isolated_db):
    isolated_db.init_db()
    fid = _mk_field(isolated_db)
    plan_id = isolated_db.add_irrigation_plan(fid, "2026-10-01")
    with pytest.raises(ValueError):
        isolated_db.set_irrigation_plan_status(plan_id, "anulowane")


def test_set_irrigation_plan_status_zapisuje_event_id_i_completed_at(isolated_db):
    isolated_db.init_db()
    fid = _mk_field(isolated_db)
    plan_id = isolated_db.add_irrigation_plan(fid, "2026-10-01", method="kroplowe")
    event_id = isolated_db.add_irrigation_event(fid, "2026-10-01", "kroplowe")
    isolated_db.set_irrigation_plan_status(plan_id, "done", irrigation_event_id=event_id)
    plan = isolated_db.get_irrigation_plan_by_id(plan_id)
    assert plan["status"] == "done"
    assert plan["irrigation_event_id"] == event_id
    assert plan["completed_at"] is not None


def test_set_irrigation_plan_status_moze_zapisac_notatke(isolated_db):
    isolated_db.init_db()
    fid = _mk_field(isolated_db)
    plan_id = isolated_db.add_irrigation_plan(fid, "2026-10-01")
    isolated_db.set_irrigation_plan_status(plan_id, "skipped", notes="deszcz wystarczyl")
    plan = isolated_db.get_irrigation_plan_by_id(plan_id)
    assert plan["status"] == "skipped"
    assert plan["notes"] == "deszcz wystarczyl"


def test_delete_irrigation_plan(isolated_db):
    isolated_db.init_db()
    fid = _mk_field(isolated_db)
    plan_id = isolated_db.add_irrigation_plan(fid, "2026-10-01")
    isolated_db.delete_irrigation_plan(plan_id)
    assert isolated_db.get_irrigation_plan_by_id(plan_id) is None


# --- warstwa API -------------------------------------------------------

def _add_field_api(app_client, parcel_id="AAA"):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": parcel_id, "irrigation_mode": "irrigation",
        "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    assert r.status_code == 200
    return r.json()["id"]


def test_api_plan_add_i_list(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan",
                         json={"planned_date": "2026-10-05", "method": "kroplowe"})
    assert r.status_code == 200
    plan_id = r.json()["id"]

    r2 = app_client.get(f"/api/water/field/{fid}/plan")
    assert r2.status_code == 200
    plans = r2.json()
    assert len(plans) == 1
    assert plans[0]["id"] == plan_id
    assert plans[0]["status"] == "planned"


def test_api_plan_add_odrzuca_zla_date(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan", json={"planned_date": "nie-data"})
    assert r.status_code == 422


def test_api_plan_add_404_dla_nieistniejacego_pola(app_client):
    r = app_client.post("/api/water/field/999/plan", json={"planned_date": "2026-10-05"})
    assert r.status_code == 404


def test_api_plan_upcoming_dokleja_parcel_id(app_client):
    fid = _add_field_api(app_client, parcel_id="UZB-Samarkand-12")
    app_client.post(f"/api/water/field/{fid}/plan", json={"planned_date": "2026-10-05"})

    r = app_client.get("/api/water/plan/upcoming")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["parcel_id"] == "UZB-Samarkand-12"


def test_api_plan_complete_loguje_zdarzenie_domyslnie(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan",
                         json={"planned_date": "2026-10-05", "method": "kroplowe"})
    plan_id = r.json()["id"]

    r2 = app_client.post(f"/api/water/plan/{plan_id}/complete",
                          json={"amount_or_duration": 4.0, "amount_unit": "mm"})
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "done"
    assert body["irrigation_event_id"] is not None

    events = app_client.get(f"/api/water/field/{fid}/irrigation").json()
    assert len(events) == 1
    assert events[0]["method"] == "kroplowe"
    assert events[0]["source"] == "plan"

    plans = app_client.get(f"/api/water/field/{fid}/plan").json()
    assert plans[0]["status"] == "done"
    assert plans[0]["irrigation_event_id"] == body["irrigation_event_id"]


def test_api_plan_complete_bez_logowania_zdarzenia(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan",
                         json={"planned_date": "2026-10-05", "method": "kroplowe"})
    plan_id = r.json()["id"]

    r2 = app_client.post(f"/api/water/plan/{plan_id}/complete", json={"log_event": False})
    assert r2.status_code == 200
    assert r2.json()["irrigation_event_id"] is None
    assert app_client.get(f"/api/water/field/{fid}/irrigation").json() == []


def test_api_plan_complete_bez_metody_wymaga_jej_w_requeście(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan", json={"planned_date": "2026-10-05"})
    plan_id = r.json()["id"]

    r2 = app_client.post(f"/api/water/plan/{plan_id}/complete", json={})
    assert r2.status_code == 400

    r3 = app_client.post(f"/api/water/plan/{plan_id}/complete", json={"method": "zalewowe"})
    assert r3.status_code == 200


def test_api_plan_skip(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan", json={"planned_date": "2026-10-05"})
    plan_id = r.json()["id"]

    r2 = app_client.post(f"/api/water/plan/{plan_id}/skip", json={"notes": "deszcz wystarczyl"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "skipped"

    plans = app_client.get(f"/api/water/field/{fid}/plan").json()
    assert plans[0]["status"] == "skipped"
    assert plans[0]["notes"] == "deszcz wystarczyl"

    events = app_client.get(f"/api/water/field/{fid}/irrigation").json()
    assert events == []  # skip nigdy nie loguje realnego zdarzenia


def test_api_plan_delete(app_client):
    fid = _add_field_api(app_client)
    r = app_client.post(f"/api/water/field/{fid}/plan", json={"planned_date": "2026-10-05"})
    plan_id = r.json()["id"]

    r2 = app_client.delete(f"/api/water/plan/{plan_id}")
    assert r2.status_code == 200
    assert app_client.get(f"/api/water/field/{fid}/plan").json() == []


def test_api_plan_complete_i_skip_404_dla_nieistniejacego_planu(app_client):
    assert app_client.post("/api/water/plan/999/complete", json={}).status_code == 404
    assert app_client.post("/api/water/plan/999/skip", json={}).status_code == 404
    assert app_client.delete("/api/water/plan/999").status_code == 404
