"""Testy geometrii pól modułu wodnego: ULDK po numerze / punkcie, import plików."""
import json
from datetime import date

import pytest
from shapely.geometry import box

import core
import water_geometry as wg

SQ1 = box(17.0, 51.1, 17.002, 51.101)
SQ2 = box(17.002, 51.1, 17.004, 51.101)   # przylega do SQ1


@pytest.fixture
def fake_uldk(monkeypatch):
    geoms = {"141201_1.0001.6509": SQ1, "141201_1.0001.6510": SQ2}

    def fetch(pid):
        if pid not in geoms:
            raise core.AppError(f"Nie znaleziono działki o ID „{pid}”.")
        return geoms[pid], {"gmina": "Testowa"}
    monkeypatch.setattr(core, "fetch_parcel_geometry", fetch)
    monkeypatch.setattr(core, "fetch_parcel_at_point",
                        lambda lon, lat: (SQ1, {"parcel_id": "141201_1.0001.6509", "gmina": "Testowa"}))
    return geoms


def test_rozpoznaje_polskie_numery_dzialek():
    assert wg.is_polish_parcel_ids("141201_1.0001.6509")
    assert wg.is_polish_parcel_ids("141201_1.0001.6509, 026401_1.0001.AR_12.123/4")
    assert not wg.is_polish_parcel_ids("UZB-Samarkand-12")
    assert not wg.is_polish_parcel_ids("141201_1.0001.6509, pole-2")
    assert not wg.is_polish_parcel_ids("")


def test_kilka_dzialek_daje_sume_obrysow(fake_uldk):
    r = wg.geometry_from_parcel_ids("141201_1.0001.6509; 141201_1.0001.6510")
    from shapely import wkt
    g = wkt.loads(r["geometry_wkt"])
    assert g.geom_type == "Polygon"            # przylegające -> jeden poligon
    assert abs(g.area - (SQ1.area + SQ2.area)) < 1e-12
    assert [p["parcel_id"] for p in r["parcels"]] == ["141201_1.0001.6509", "141201_1.0001.6510"]


def test_api_dodaje_pole_bez_wkt_po_numerze_dzialki(app_client, fake_uldk, isolated_db):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "141201_1.0001.6509", "irrigation_mode": "rainfed"})
    assert r.status_code == 200 and r.json()["geometry_source"] == "uldk"
    f = isolated_db.get_water_field_by_id(r.json()["id"])
    assert f["geometry_wkt"].startswith("POLYGON")


def test_api_bez_wkt_i_bez_numeru_dzialki_odrzuca(app_client, fake_uldk):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "UZB-Samarkand-12", "irrigation_mode": "irrigation"})
    assert r.status_code == 422 and "obrys na mapie" in r.json()["detail"]


def test_api_nieistniejaca_dzialka(app_client, fake_uldk):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "141201_1.0001.9999", "irrigation_mode": "rainfed"})
    assert r.status_code == 422 and "Nie znaleziono" in r.json()["detail"]


def test_api_parcel_lookup_i_parcel_at(app_client, fake_uldk):
    r = app_client.get("/api/water/parcel_lookup", params={"ids": "141201_1.0001.6509"})
    assert r.status_code == 200 and r.json()["area_ha"] > 0
    assert app_client.get("/api/water/parcel_lookup", params={"ids": "abc"}).status_code == 422
    r = app_client.get("/api/water/parcel_at", params={"lat": 51.1005, "lon": 17.001})
    assert r.status_code == 200 and r.json()["parcel_id"] == "141201_1.0001.6509"


def test_api_import_geojson_i_csv(app_client, fake_uldk, isolated_db):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": "UZB-1", "irrigation_mode": "irygacyjny", "crop_type": "bawełna"},
         "geometry": {"type": "Polygon", "coordinates": [[[66.9, 39.6], [66.91, 39.6], [66.91, 39.61], [66.9, 39.61], [66.9, 39.6]]]}},
        {"type": "Feature", "properties": {"id": "UZB-2"}, "geometry": None},          # brak geometrii, nie-PL
    ]}
    r = app_client.post("/api/water/field/import", json={
        "filename": "pola.geojson", "content": json.dumps(gj), "region": "Samarkand"})
    body = r.json()
    assert [a["parcel_id"] for a in body["added"]] == ["UZB-1"]
    assert body["errors"][0]["parcel_id"] == "UZB-2"
    f = isolated_db.get_water_field_by_id(body["added"][0]["id"])
    assert f["irrigation_mode"] == "irrigation" and f["region"] == "Samarkand" and f["crop_type"] == "bawełna"

    csv_text = "parcel_id;crop_type;soil_taw_mm;wkt\n141201_1.0001.6509;pszenica;120,5;\nUZB-1;;;\n"
    r = app_client.post("/api/water/field/import", json={"filename": "pola.csv", "content": csv_text})
    body = r.json()
    assert [a["parcel_id"] for a in body["added"]] == ["141201_1.0001.6509"]
    assert body["skipped"][0]["parcel_id"] == "UZB-1"          # już w module
    f = isolated_db.get_water_field_by_id(body["added"][0]["id"])
    assert f["soil_taw_mm"] == 120.5 and f["irrigation_mode"] == "rainfed"


def test_api_import_zly_plik(app_client):
    r = app_client.post("/api/water/field/import", json={"filename": "x.geojson", "content": "{nie json"})
    assert r.status_code == 422


def test_fetch_parcel_at_point_parsuje_uldk(monkeypatch):
    from unittest.mock import MagicMock
    text = "0\nSRID=4326;POLYGON((17 51,17.001 51,17.001 51.001,17 51.001,17 51))|026401_1.0001.2664|dolnośląskie|Wrocław|Wrocław (miasto)\n"
    monkeypatch.setattr(core.requests, "get", lambda url, params=None, timeout=None:
                        MagicMock(status_code=200, text=text, raise_for_status=lambda: None))
    g, meta = core.fetch_parcel_at_point(17.0005, 51.0005)
    assert meta["parcel_id"] == "026401_1.0001.2664" and g.area > 0
    monkeypatch.setattr(core.requests, "get", lambda url, params=None, timeout=None:
                        MagicMock(status_code=200, text="-1 brak wyników", raise_for_status=lambda: None))
    with pytest.raises(core.AppError):
        core.fetch_parcel_at_point(60.0, 40.0)


# --- Fields of The World (gotowe obrysy poza Polską) ---------------------------

def _wkb(g):
    from shapely import wkb
    return wkb.dumps(g)


@pytest.fixture
def fake_ftw(monkeypatch):
    f2024 = box(66.699, 39.619, 66.702, 39.622)
    f2025 = box(66.6995, 39.6195, 66.7015, 39.6215)
    other = box(66.71, 39.61, 66.72, 39.62)
    calls = []

    def query(url, lon, lat):
        calls.append(url)
        if "Uzbekistan" not in url:
            return []
        return [("1", 2024, 97.3, _wkb(f2024)), ("2", 2025, 98.3, _wkb(f2025)), ("3", 2025, 99.0, _wkb(other))]
    monkeypatch.setattr(wg, "_ftw_query_point", query)
    return calls


def test_ftw_wybiera_najnowszy_rocznik_zawierajacy_punkt(fake_ftw):
    r = wg.ftw_field_at(66.70, 39.62)
    assert r["source"] == "ftw" and r["year"] == 2025 and r["confidence"] == 98
    assert r["ftw_id"] == "UZ-2025-2" and r["area_ha"] > 0
    assert "Fields of The World" in r["attribution"]


def test_ftw_kolejnosc_krajow_i_brak_pola(fake_ftw):
    # punkt w Dolinie Fergańskiej leży w bboxach UZ, TJ i KG — UZ pierwszy
    assert wg.ftw_candidate_countries(71.6, 40.45)[:3] == ["UZ", "TJ", "KG"]
    assert wg.ftw_field_at(66.80, 39.70) is None          # żaden kandydat nie zawiera punktu
    assert wg.ftw_candidate_countries(10.0, 45.0) == []


def test_ftw_url_kraju():
    assert wg.ftw_url("UZ").endswith("admin:country_code=UZ/Uzbekistan.parquet")


def test_api_field_at_auto_pl_uldk_poza_pl_ftw(app_client, fake_uldk, fake_ftw):
    r = app_client.get("/api/water/field_at", params={"lat": 51.1005, "lon": 17.001})
    assert r.status_code == 200 and r.json()["source"] == "uldk"
    r = app_client.get("/api/water/field_at", params={"lat": 39.62, "lon": 66.70})
    assert r.status_code == 200 and r.json()["source"] == "ftw" and r.json()["year"] == 2025
    r = app_client.get("/api/water/field_at", params={"lat": 51.1005, "lon": 17.001, "source": "ftw"})
    assert r.status_code == 404                                # fake FTW nie ma pól w PL
    assert "Poland" in fake_ftw[-1]
    r = app_client.get("/api/water/field_at", params={"lat": 45.0, "lon": 10.0})
    assert r.status_code == 404 and "narysuj" in r.json()["detail"]


def test_api_field_at_ftw_niedostepne(app_client, monkeypatch):
    def boom(url, lon, lat):
        raise IOError("HTTP 503")
    monkeypatch.setattr(wg, "_ftw_query_point", boom)
    r = app_client.get("/api/water/field_at", params={"lat": 39.61, "lon": 66.71})
    assert r.status_code == 502 and "FTW" in r.json()["detail"]


def test_api_naprawia_niepoprawny_obrys(app_client, isolated_db):
    bowtie = "POLYGON((66.70 39.62, 66.71 39.63, 66.71 39.62, 66.70 39.63, 66.70 39.62))"   # ósemka
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "UZB-8", "irrigation_mode": "irrigation", "geometry_wkt": bowtie})
    assert r.status_code == 200
    from shapely import wkt
    g = wkt.loads(isolated_db.get_water_field_by_id(r.json()["id"])["geometry_wkt"])
    assert g.is_valid and g.area > 0
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "UZB-9", "irrigation_mode": "irrigation", "geometry_wkt": "POINT(66.7 39.6)"})
    assert r.status_code == 422


def test_ftw_usuwa_male_dziury():
    from shapely.geometry import Polygon
    outer = [(0, 0), (200, 0), (200, 200), (0, 200)]
    small = [(10, 10), (20, 10), (20, 20), (10, 20)]          # 100 m²
    big = [(100, 100), (150, 100), (150, 150), (100, 150)]    # 2500 m²
    g = wg._drop_small_holes(Polygon(outer, [small, big]), wg.FTW_MIN_HOLE_M2)
    assert len(g.interiors) == 1 and Polygon(g.interiors[0]).area == 2500


# --- pola rozrzucone po kraju / osobne pola / duplikaty -----------------------

@pytest.fixture
def far_uldk(monkeypatch):
    geoms = {"040604_5.0014.48": box(18.85, 53.37, 18.86, 53.376),     # Kujawy
             "101807_5.0013.47": box(18.126, 51.346, 18.134, 51.351),   # ~225 km dalej
             "141201_1.0001.6509": SQ1, "141201_1.0001.6510": SQ2}
    monkeypatch.setattr(core, "fetch_parcel_geometry", lambda pid: (geoms[pid], {}))
    return geoms


def test_extent_km():
    assert wg.extent_km(SQ1) < 0.2
    assert wg.extent_km(box(17.1, 50.9, 18.86, 53.38)) > 200


def test_api_odrzuca_pole_z_dzialek_rozrzuconych_po_kraju(app_client, far_uldk):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "040604_5.0014.48, 101807_5.0013.47", "irrigation_mode": "rainfed"})
    assert r.status_code == 422
    assert "osobne pole" in r.json()["detail"] and "km" in r.json()["detail"]


def test_api_kazda_dzialka_jako_osobne_pole(app_client, far_uldk, isolated_db):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "040604_5.0014.48, 101807_5.0013.47", "irrigation_mode": "rainfed",
        "split_parcels": True, "crop_type": "pszenica"})
    assert r.status_code == 200
    assert [a["parcel_id"] for a in r.json()["added"]] == ["040604_5.0014.48", "101807_5.0013.47"]
    fields = isolated_db.get_water_fields(status="active")
    assert len(fields) == 2 and all(f["crop_type"] == "pszenica" for f in fields)
    # ponownie: już istniejące są pomijane
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "040604_5.0014.48, 101807_5.0013.47", "irrigation_mode": "rainfed", "split_parcels": True})
    assert r.json()["added"] == [] and len(r.json()["skipped_existing"]) == 2


def test_api_sasiednie_dzialki_nadal_jako_jedno_pole(app_client, far_uldk):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "141201_1.0001.6509, 141201_1.0001.6510", "irrigation_mode": "rainfed"})
    assert r.status_code == 200 and r.json()["geometry_source"] == "uldk"


def test_api_nie_dodaje_duplikatu(app_client, far_uldk):
    body = {"parcel_id": "141201_1.0001.6509", "irrigation_mode": "rainfed"}
    assert app_client.post("/api/water/field/add", json=body).status_code == 200
    r = app_client.post("/api/water/field/add", json=body)
    assert r.status_code == 409 and "już w module" in r.json()["detail"]


def test_sprawdzenie_istniejacego_rozleglego_pola_daje_czytelny_blad(isolated_db, monkeypatch):
    import water_stress as ws
    isolated_db.init_db()
    wide = "MULTIPOLYGON(((17.1 50.9,17.11 50.9,17.11 50.91,17.1 50.9)),((18.85 53.37,18.86 53.37,18.86 53.38,18.85 53.37)))"
    fid = isolated_db.add_water_field("stare-pole", "rainfed", wide)
    with pytest.raises(RuntimeError, match="osobno"):
        ws.fetch_sources(isolated_db.get_water_field_by_id(fid), date(2026, 8, 1), date(2026, 9, 1))
