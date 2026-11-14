"""Testy analizy wieloletniej stref plonu + nawożenia (yield_multi.py,
yield_multi_api.py, yield_report.py) — bez sieci: Process API podmienione
na sztuczne GeoTIFF-y (jak w test_yield_mvi)."""
import io
import json
import zipfile
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

import yield_multi as ymu
import yield_mvi as ym

SQ_WKT = "POLYGON((17.000 51.100, 17.004 51.100, 17.004 51.1027, 17.000 51.1027, 17.000 51.100))"


def _fake_arr(h=30, w=28, n=8, gradient=True, flip=False):
    x = np.linspace(0, 1, w)[None, :].repeat(h, 0)
    if flip:
        x = x[:, ::-1]
    evi2 = 0.35 + 0.35 * x if gradient else np.full((h, w), 0.5)
    ndmi = 0.15 + 0.20 * x if gradient else np.full((h, w), 0.3)
    nsc = np.full((h, w), n, dtype="float32")
    return np.stack([evi2, ndmi, evi2 * ndmi, nsc]).astype("float32")


def _tiff_bytes(arr, bounds=(17.0, 51.1, 17.004, 51.1027)):
    import rasterio
    from rasterio.transform import from_bounds
    h, w = arr.shape[1:]
    tr = from_bounds(*bounds, w, h)
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", height=h, width=w, count=4, dtype="float32", crs="EPSG:4326",
                      transform=tr) as ds:
            ds.write(arr)
        return mem.read()


def _inside(shape=(30, 28)):
    return np.ones(shape, dtype=bool)


# --------------------------- łączenie lat ---------------------------

def test_year_rel_srednia_1():
    r = ymu.year_rel(_fake_arr(), _inside())
    assert abs(float(np.nanmean(r["rel"][r["valid"]])) - 1.0) < 1e-6
    assert r["cv_pct"] > 0 and r["median_scenes"] == 8


def test_year_rel_odrzuca_zle_lata():
    with pytest.raises(ValueError, match="czystymi scenami"):
        ymu.year_rel(_fake_arr(n=2), _inside())
    bare = _fake_arr(gradient=False)
    bare[2][:] = 0.005
    with pytest.raises(ValueError, match="łanu"):
        ymu.year_rel(bare, _inside())


def test_combine_years_usrednia_i_wymaga_polowy_lat():
    r1 = ymu.year_rel(_fake_arr(), _inside())
    r2 = ymu.year_rel(_fake_arr(), _inside())
    r3 = ymu.year_rel(_fake_arr(flip=True), _inside())     # odwrócony gradient
    comb = ymu.combine_years([r1, r2, r3])
    assert abs(float(np.nanmean(comb["rel"][comb["valid"]])) - 1.0) < 1e-6
    # 2× gradient W->E + 1× odwrotny => wschód nadal lepszy, ale słabiej
    only12 = ymu.combine_years([r1, r2])
    east_12 = float(np.nanmean(only12["rel"][:, -3:]))
    east_123 = float(np.nanmean(comb["rel"][:, -3:]))
    assert east_123 < east_12
    # piksel bez danych w >= połowie lat wypada
    r4 = ymu.year_rel(_fake_arr(), _inside())
    r4["valid"] = r4["valid"].copy(); r4["valid"][:, :10] = False
    r5 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in r4.items()}
    comb2 = ymu.combine_years([r4, r5, r1])
    assert not comb2["valid"][:, :10].any() and comb2["valid"][:, 15:].all()


def test_zones_from_rel_gradient():
    r = ymu.year_rel(_fake_arr(), _inside())
    zones = ymu.zones_from_rel(r["rel"], r["valid"])
    assert zones[:, 0].max() <= zones[:, -1].min()
    zs = ymu.zone_stats_multi(r["rel"], zones, r["valid"], np.full(r["rel"].shape, np.nan))
    assert sum(z["share_pct"] for z in zs) == pytest.approx(100, abs=0.5)


# --------------------------- dawki nawozu ---------------------------

def _zones_stats():
    r = ymu.year_rel(_fake_arr(), _inside())
    zones = ymu.zones_from_rel(r["rel"], r["valid"])
    return ymu.zone_stats_multi(r["rel"], zones, r["valid"], np.full(r["rel"].shape, np.nan))


def test_dawki_potencjal_zachowuja_srednia():
    zs = _zones_stats()
    f = ymu.fertilizer_doses(zs, 300.0, "potencjal", rounding=1)
    assert abs(f["effective_avg_kg_ha"] - 300.0) <= 1.0           # masa zachowana (± zaokrąglenie)
    doses = {z["zone"]: z["dose_kg_ha"] for z in f["zones"]}
    ks = sorted(doses)
    assert doses[ks[0]] < doses[ks[-1]]                            # mocna strefa dostaje więcej
    assert f["total_kg"] == pytest.approx(
        sum(z["dose_kg_ha"] * z["area_ha"] for z in f["zones"]), abs=len(f["zones"]))


def test_dawki_wyrownawcza_odwraca_kierunek():
    zs = _zones_stats()
    f = ymu.fertilizer_doses(zs, 200.0, "wyrownawcza", rounding=1)
    doses = {z["zone"]: z["dose_kg_ha"] for z in f["zones"]}
    ks = sorted(doses)
    assert doses[ks[0]] > doses[ks[-1]]                            # słaba strefa dostaje więcej
    assert abs(f["effective_avg_kg_ha"] - 200.0) <= 1.0


def test_dawki_zaokraglenie_widelki_i_walidacja():
    zs = _zones_stats()
    f = ymu.fertilizer_doses(zs, 300.0, "potencjal", rounding=10, dose_min=280, dose_max=310)
    for z in f["zones"]:
        assert z["dose_kg_ha"] % 10 == 0 and 280 <= z["dose_kg_ha"] <= 310
    with pytest.raises(ValueError, match="[Ss]trategi"):
        ymu.fertilizer_doses(zs, 300.0, "inna")
    with pytest.raises(ValueError, match="dodatnia"):
        ymu.fertilizer_doses(zs, 0, "potencjal")


# --------------------------- próby glebowe ---------------------------

def test_parse_soil_csv_srednik_i_przecinek_dziesietny():
    csv = ("Nr;Szerokość;Długość;pH;P2O5;K2O;Mg;Próchnica\n"
           "1;51,101;17,001;6,2;12,5;18,0;5,1;2,10\n"
           "2;51,102;17,003;5,8;8,0;14,5;4,2;1,85\n"
           "3;;;6,0;10,0;;;\n").encode("utf-8")
    out = ymu.parse_soil_csv(csv)
    s = out["samples"]
    assert len(s) == 3 and s[0]["id"] == "1" and s[0]["ph"] == 6.2 and s[0]["p2o5"] == 12.5
    assert s[0]["lat"] == 51.101 and s[2]["lat"] is None and s[2]["k2o"] is None


def test_parse_soil_csv_zamienione_wspolrzedne_i_bledy():
    csv = ("lat,lon,pH\n17.001,51.101,6.5\n").encode("utf-8")
    out = ymu.parse_soil_csv(csv)
    assert out["samples"][0]["lat"] == 51.101 and out["warnings"]
    with pytest.raises(ValueError, match="kolumn"):
        ymu.parse_soil_csv(b"a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="nagłówek|wiersz"):
        ymu.parse_soil_csv(b"pH\n")


def test_soil_zone_summary():
    samples = [{"zone": 1, "ph": 6.0, "p2o5": 10, "k2o": None, "mg": 4, "humus": 2},
               {"zone": 1, "ph": 7.0, "p2o5": 20, "k2o": 15, "mg": 6, "humus": None},
               {"zone": None, "ph": 5.0, "p2o5": 1, "k2o": 1, "mg": 1, "humus": 1}]
    zs = ymu.soil_zone_summary(samples)
    assert len(zs) == 1 and zs[0]["zone"] == 1 and zs[0]["n"] == 2
    assert zs[0]["ph"] == 6.5 and zs[0]["k2o"] == 15


# --------------------------- plik siewnika ---------------------------

def test_prescription_shapefile_zip(tmp_path):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon",
          "coordinates": [[[17.0, 51.1], [17.002, 51.1], [17.002, 51.102], [17.0, 51.102], [17.0, 51.1]]]},
         "properties": {"zone": 2, "label": "90–97%", "rate_kg_ha": 280.0}},
        {"type": "Feature", "geometry": {"type": "Polygon",
          "coordinates": [[[17.002, 51.1], [17.004, 51.1], [17.004, 51.102], [17.002, 51.102], [17.002, 51.1]]]},
         "properties": {"zone": 4, "label": "103–110%", "rate_kg_ha": 320.0}}]}
    out = str(tmp_path / "dawki_shp.zip")
    ymu.prescription_shapefile_zip(gj, out)
    with zipfile.ZipFile(out) as zf:
        names = {n.split(".")[-1] for n in zf.namelist()}
        assert names == {"shp", "shx", "dbf", "prj"}
        zf.extractall(tmp_path / "shp")
    import shapefile
    r = shapefile.Reader(str(tmp_path / "shp" / "dawki_shp"))
    recs = [dict(zip([f[0] for f in r.fields[1:]], rec)) for rec in r.records()]
    assert {rec["RATE"] for rec in recs} == {280.0, 320.0}
    assert len(r.shapes()) == 2


# --------------------------- pełny przebieg + API ---------------------------

@pytest.fixture
def fake_process_api(monkeypatch):
    import core
    import sentinel1
    monkeypatch.setattr(core, "cdse_token", lambda: "t")
    calls = []

    def post(url, payload, headers, timeout=None, refresh_token_fn=None):
        calls.append(payload)
        yr = payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:4]
        # 2023: za mało scen (rok "zachmurzony") -> pominięty
        arr = _fake_arr(n=2) if yr == "2023" else _fake_arr()
        return MagicMock(status_code=200, content=_tiff_bytes(arr))
    monkeypatch.setattr(sentinel1, "_post_with_retry", post)
    return calls


@pytest.fixture
def inline_threads(monkeypatch):
    import yield_multi_api as yma

    class T:
        def __init__(self, target=None, args=(), daemon=None):
            self._t, self._a = target, args
        def start(self):
            self._t(*self._a)
    monkeypatch.setattr(yma, "Thread", T)


YEARS = [{"year": 2022, "crop": "kukurydza", "yield_t_ha": 9.5},
         {"year": 2023, "crop": "pszenica", "yield_t_ha": 7.0},
         {"year": 2024, "crop": "rzepak", "yield_t_ha": 3.6},
         {"year": 2025, "crop": "kukurydza", "yield_t_ha": 10.1}]


def test_build_analysis_pomija_zly_rok(tmp_path, monkeypatch, fake_process_api):
    from shapely import wkt
    monkeypatch.setattr(ymu, "ANALYSES_DIR", str(tmp_path))
    stats = ymu.build_analysis(wkt.loads(SQ_WKT), YEARS, 11)
    assert stats["years_ok"] == 3
    skipped = [y for y in stats["years"] if y["status"] == "error"]
    assert len(skipped) == 1 and skipped[0]["year"] == 2023
    d = tmp_path / "11"
    for f in ("strefy.geojson", "strefy.tif", "nakladka.png", "rel_sredni.tif",
              "rok_2022.png", "rel_2022.tif", "rok_2025.png"):
        assert (d / f).exists(), f
    assert not (d / "rok_2023.png").exists()
    assert stats["zones"] and len(stats["bounds"]) == 4 and stats["area_ha"] > 0


def test_build_analysis_za_malo_lat(tmp_path, monkeypatch, fake_process_api):
    from shapely import wkt
    monkeypatch.setattr(ymu, "ANALYSES_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="co najmniej 2"):
        ymu.build_analysis(wkt.loads(SQ_WKT), [{"year": 2023, "crop": "pszenica", "yield_t_ha": 7.0}], 12)


def test_api_analiza_pelny_cykl(app_client, tmp_path, monkeypatch, fake_process_api, inline_threads):
    monkeypatch.setattr(ymu, "ANALYSES_DIR", str(tmp_path))
    r = app_client.post("/api/yield/analysis", json={
        "label": "LENs-07 wielolecie", "geometry_wkt": SQ_WKT, "years": YEARS,
        "fertilizer": {"fert_name": "NPK 6-20-30", "avg_dose_kg_ha": 300, "strategy": "potencjal"}})
    assert r.status_code == 200, r.text
    aid = r.json()["id"]
    lst = app_client.get("/api/yield/analysis").json()
    assert lst[0]["id"] == aid and lst[0]["status"] == "done"
    a = app_client.get(f"/api/yield/analysis/{aid}").json()
    assert a["stats"]["years_ok"] == 3 and a["zones_geojson"]["features"]
    fert = a["stats"]["fertilizer"]
    assert fert["fert_name"] == "NPK 6-20-30" and abs(fert["effective_avg_kg_ha"] - 300) <= 5

    # przeliczenie dawek inną strategią
    r = app_client.post(f"/api/yield/analysis/{aid}/fertilizer",
                        json={"avg_dose_kg_ha": 200, "strategy": "wyrownawcza", "rounding_kg": 1})
    assert r.status_code == 200
    f2 = r.json()["fertilizer"]
    d2 = {z["zone"]: z["dose_kg_ha"] for z in f2["zones"]}
    ks = sorted(d2)
    assert d2[ks[0]] > d2[ks[-1]]

    # próby glebowe (środek pola -> przypisane do strefy)
    csv = ("nr;lat;lon;pH;P2O5;K2O;Mg\n"
           "1;51,1012;17,0012;6,1;11;16;5\n"
           "2;51,1015;17,0035;6,5;14;19;6\n"
           "3;;;5,9;9;13;4\n").encode("utf-8")
    r = app_client.post(f"/api/yield/analysis/{aid}/soil",
                        files={"file": ("proby.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["n_samples"] == 3 and r.json()["zone_summary"]
    a = app_client.get(f"/api/yield/analysis/{aid}").json()
    assert a["soil"]["samples"][0]["zone"] in (1, 2, 3, 4, 5)
    assert a["soil"]["samples"][2]["zone"] is None

    # eksporty
    r = app_client.get(f"/api/yield/analysis/{aid}/file/dawki_shp.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert app_client.get(f"/api/yield/analysis/{aid}/file/dawki.geojson").status_code == 200
    assert app_client.get(f"/api/yield/analysis/{aid}/file/nakladka.png").status_code == 200
    assert app_client.get(f"/api/yield/analysis/{aid}/file/rok_2022.png").status_code == 200
    assert app_client.get(f"/api/yield/analysis/{aid}/file/zle.txt").status_code == 404

    # raport PDF
    r = app_client.get(f"/api/yield/analysis/{aid}/report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"

    # usunięcie
    assert app_client.delete(f"/api/yield/analysis/{aid}").status_code == 200
    assert not (tmp_path / str(aid)).exists()
    assert app_client.get(f"/api/yield/analysis/{aid}").status_code == 404


def test_api_analiza_walidacja(app_client):
    base = {"geometry_wkt": SQ_WKT}
    r = app_client.post("/api/yield/analysis", json=base | {
        "years": [{"year": 2024, "crop": "kukurydza", "yield_t_ha": 9}]})
    assert r.status_code == 422                                     # < 2 lata
    r = app_client.post("/api/yield/analysis", json=base | {
        "years": [{"year": 2024, "crop": "kukurydza", "yield_t_ha": 9},
                  {"year": 2024, "crop": "pszenica", "yield_t_ha": 7}]})
    assert r.status_code == 422                                     # zdublowany rok
    r = app_client.post("/api/yield/analysis", json={
        "parcel_id": "pole-bez-obrysu",
        "years": [{"year": 2023, "crop": "kukurydza", "yield_t_ha": 9},
                  {"year": 2024, "crop": "pszenica", "yield_t_ha": 7}]})
    assert r.status_code == 422                                     # brak geometrii
