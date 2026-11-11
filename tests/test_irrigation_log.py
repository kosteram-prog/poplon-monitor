"""Testy modułu log-u nawodnienia (water_monitored_fields / irrigation_events)
— warstwa bazy (isolated_db) i warstwa API (app_client). Wzorem
test_db.py / test_api.py."""
import pytest


# --- warstwa bazy ------------------------------------------------------

def test_add_water_field_odrzuca_nieznany_irrigation_mode(isolated_db):
    isolated_db.init_db()
    with pytest.raises(ValueError):
        isolated_db.add_water_field("141201_1.0001.6509", "sterowane_zaworem", "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))")


def test_add_water_field_odrzuca_nieznane_kolumny(isolated_db):
    isolated_db.init_db()
    with pytest.raises(ValueError):
        isolated_db.add_water_field(
            "141201_1.0001.6509", "rainfed",
            "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
            evil="DROP TABLE water_monitored_fields")


def test_add_water_field_akceptuje_znane_pola(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field(
        "141201_1.0001.6509", "irrigation", "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))", region="Samarkand", farmer_name="Test")
    field = isolated_db.get_water_field_by_id(fid)
    assert field["irrigation_mode"] == "irrigation"
    assert field["region"] == "Samarkand"
    assert field["status"] == "active"


def test_get_water_fields_filtruje_po_statusie(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))")
    isolated_db.update_water_field_status(fid, "completed")
    assert isolated_db.get_water_fields(status="active") == []
    assert len(isolated_db.get_water_fields(status="completed")) == 1


def test_add_irrigation_event_odrzuca_nieznana_metode(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "irrigation", "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))")
    with pytest.raises(ValueError):
        isolated_db.add_irrigation_event(fid, "2026-06-01", "sprinkler_x")


def test_add_irrigation_event_i_odczyt_historii(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "irrigation", "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))")
    isolated_db.add_irrigation_event(fid, "2026-06-05", "kroplowe", amount_or_duration=3.5, amount_unit="mm")
    isolated_db.add_irrigation_event(fid, "2026-06-01", "zalewowe", amount_or_duration=12)
    events = isolated_db.get_irrigation_events(fid)
    assert len(events) == 2
    # ORDER BY event_date ASC
    assert events[0]["event_date"] == "2026-06-01"
    assert events[1]["method"] == "kroplowe"


# --- warstwa API ---------------------------------------------------------

def test_api_water_field_add_i_list(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "141201_1.0001.6509", "irrigation_mode": "rainfed",
        "region": "Lower Silesia", "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    assert r.status_code == 200
    field_id = r.json()["id"]

    r2 = app_client.get("/api/water/field/list")
    assert r2.status_code == 200
    fields = r2.json()
    assert len(fields) == 1
    assert fields[0]["id"] == field_id
    assert fields[0]["recent_irrigation_events"] == []


def test_api_water_field_add_odrzuca_zly_irrigation_mode(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "141201_1.0001.6509", "irrigation_mode": "auto_sterowanie", "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    assert r.status_code == 422


def test_api_irrigation_event_add_i_get(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "AAA", "irrigation_mode": "irrigation", "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    field_id = r.json()["id"]

    r2 = app_client.post(f"/api/water/field/{field_id}/irrigation", json={
        "event_date": "2026-06-01", "method": "kroplowe",
        "amount_or_duration": 4.0, "amount_unit": "mm",
    })
    assert r2.status_code == 200

    r3 = app_client.get(f"/api/water/field/{field_id}/irrigation")
    assert r3.status_code == 200
    assert len(r3.json()) == 1
    assert r3.json()[0]["method"] == "kroplowe"


def test_api_irrigation_event_odrzuca_ujemna_ilosc(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "AAA", "irrigation_mode": "irrigation", "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    field_id = r.json()["id"]
    r2 = app_client.post(f"/api/water/field/{field_id}/irrigation", json={
        "event_date": "2026-06-01", "method": "kroplowe", "amount_or_duration": -5,
    })
    assert r2.status_code == 422


def test_api_irrigation_event_odrzuca_date_z_przyszlosci(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "AAA", "irrigation_mode": "irrigation", "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    field_id = r.json()["id"]
    r2 = app_client.post(f"/api/water/field/{field_id}/irrigation", json={
        "event_date": "2099-01-01", "method": "kroplowe",
    })
    assert r2.status_code == 422


def test_api_irrigation_event_dla_nieistniejacego_pola_zwraca_404(app_client):
    r = app_client.post("/api/water/field/999/irrigation", json={
        "event_date": "2026-06-01", "method": "kroplowe",
    })
    assert r.status_code == 404


def test_api_water_field_delete_dezaktywuje(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "AAA", "irrigation_mode": "rainfed", "geometry_wkt": "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))",
    })
    field_id = r.json()["id"]
    r2 = app_client.delete(f"/api/water/field/{field_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "deactivated"

    r3 = app_client.get("/api/water/field/list?status=active")
    assert r3.json() == []


def test_api_water_field_patch_zmienia_uprawe(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "UZB-PATCH-1", "irrigation_mode": "irrigation",
        "geometry_wkt": "POLYGON((66.9 39.6, 66.901 39.6, 66.901 39.601, 66.9 39.601, 66.9 39.6))",
        "region": "Samarkand", "crop_type": "bawełna"})
    assert r.status_code == 200
    fid = r.json()["id"]
    r = app_client.patch(f"/api/water/field/{fid}", json={"crop_type": "pszenica"})
    assert r.status_code == 200
    assert r.json()["crop_type"] == "pszenica"
    assert r.json()["region"] == "Samarkand"  # nieprzesłane pole bez zmian
    assert app_client.patch("/api/water/field/99999", json={"crop_type": "x"}).status_code == 404
