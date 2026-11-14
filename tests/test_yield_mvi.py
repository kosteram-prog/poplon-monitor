"""Testy map stref plonu MVI (yield_mvi.py, yield_api.py) — bez sieci:
odpowiedź Process API podmieniona na sztuczny GeoTIFF."""
import json
import os
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

import yield_mvi as ym

SQ_WKT = "POLYGON((17.000 51.100, 17.004 51.100, 17.004 51.1027, 17.000 51.1027, 17.000 51.100))"


def _fake_arr(h=30, w=28, n=8, gradient=True):
    """EVI2 i NDMI rosną z zachodu na wschód (lepsza część pola na wschodzie)."""
    x = np.linspace(0, 1, w)[None, :].repeat(h, 0)
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


def test_disaggregate_zachowuje_srednia_i_tworzy_strefy():
    res = ym.disaggregate(_fake_arr(), 9.5)
    s = res["stats"]
    assert abs(s["map_mean_yield_t_ha"] - 9.5) < 1e-3          # zachowanie masy
    assert s["p10_t_ha"] < 9.5 < s["p90_t_ha"]
    assert sum(z["share_pct"] for z in s["zones"]) == pytest.approx(100, abs=0.5)
    lo = s["zones"][0]; hi = s["zones"][-1]
    assert lo["mean_yield_t_ha"] < 9.5 * 0.9 and hi["mean_yield_t_ha"] > 9.5 * 1.1
    assert s["high_variability"] == (s["cv_pct"] > ym.HIGH_CV_PCT)
    # gorsza strefa po zachodniej stronie
    assert res["zones"][:, 0].max() < res["zones"][:, -1].min()


def test_disaggregate_jednorodne_pole_jedna_strefa():
    res = ym.disaggregate(_fake_arr(gradient=False), 6.0)
    zs = {z["zone"]: z["share_pct"] for z in res["stats"]["zones"]}
    assert zs[3] == 100.0 and res["stats"]["cv_pct"] == 0.0


def test_disaggregate_za_malo_scen_i_brak_lanu():
    arr = _fake_arr(n=2)
    with pytest.raises(ValueError, match="czystymi scenami"):
        ym.disaggregate(arr, 5.0)
    bare = _fake_arr(gradient=False)
    bare[0][:] = 0.05; bare[2][:] = 0.05 * 0.3
    with pytest.raises(ValueError, match="łanu"):
        ym.disaggregate(bare, 5.0)


def test_disaggregate_maska_pola_z_obrysu():
    arr = _fake_arr()
    inside = np.zeros(arr.shape[1:], dtype=bool)
    inside[5:25, 5:20] = True
    res = ym.disaggregate(arr, 7.0, inside=inside)
    assert res["stats"]["field_pixels"] == 300 and res["stats"]["valid_pixels"] == 300
    assert not res["valid"][0, 0]


def test_okno_sezonu_dla_upraw():
    assert ym.crop_window("Kukurydza", 2026) == (date(2026, 6, 1), date(2026, 9, 15))
    assert ym.crop_window("pszenica", 2025)[1] == date(2025, 6, 30)
    assert ym.crop_window("egzotyczna", 2026) == (date(2026, 6, 1), date(2026, 9, 15))


@pytest.fixture
def fake_process_api(monkeypatch):
    import core
    import sentinel1
    monkeypatch.setattr(core, "cdse_token", lambda: "t")
    calls = []

    def post(url, payload, headers, timeout=None, refresh_token_fn=None):
        calls.append((url, payload))
        return MagicMock(status_code=200, content=_tiff_bytes(_fake_arr()))
    monkeypatch.setattr(sentinel1, "_post_with_retry", post)
    return calls


def test_build_yield_map_zapisuje_pliki(tmp_path, monkeypatch, fake_process_api):
    from shapely import wkt
    monkeypatch.setattr(ym, "YIELD_MAPS_DIR", str(tmp_path))
    stats = ym.build_yield_map(wkt.loads(SQ_WKT), 8.0, date(2026, 6, 1), date(2026, 9, 15), 7)
    d = tmp_path / "7"
    assert (d / "plon_mvi.tif").exists() and (d / "nakladka.png").exists() and (d / "strefy.geojson").exists()
    gj = json.loads((d / "strefy.geojson").read_text(encoding="utf-8"))
    assert gj["features"] and {f["properties"]["zone"] for f in gj["features"]} <= {1, 2, 3, 4, 5}
    assert abs(stats["map_mean_yield_t_ha"] - 8.0) < 1e-3 and len(stats["bounds"]) == 4
    url, payload = fake_process_api[0]
    assert url.endswith("/process") and payload["evalscript"] == ym.EVALSCRIPT_MVI
    assert payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"].startswith("2026-06-01")
    import rasterio
    with rasterio.open(d / "plon_mvi.tif") as ds:
        band = ds.read(1)
        assert abs(np.nanmean(band) - 8.0) < 0.05


def test_api_mapa_plonu(app_client, tmp_path, monkeypatch, fake_process_api):
    monkeypatch.setattr(ym, "YIELD_MAPS_DIR", str(tmp_path))
    r = app_client.post("/api/yield/mvi", json={"geometry_wkt": SQ_WKT, "crop_type": "kukurydza",
                                                "season_year": 2026, "reported_yield_t_ha": 10.2,
                                                "label": "LENs-07"})
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    lst = app_client.get("/api/yield/maps").json()
    assert lst[0]["id"] == mid and lst[0]["status"] == "done" and lst[0]["date_from"] == "2026-06-01"
    m = app_client.get(f"/api/yield/map/{mid}").json()
    assert m["zones_geojson"]["type"] == "FeatureCollection" and m["stats"]["reported_yield_t_ha"] == 10.2
    assert app_client.get(f"/api/yield/map/{mid}/overlay.png").headers["content-type"] == "image/png"
    assert app_client.get(f"/api/yield/map/{mid}/plon_mvi.tif").status_code == 200
    assert app_client.get(f"/api/yield/map/{mid}/zle.txt").status_code == 404
    assert app_client.get("/api/yield/crop_window", params={"crop_type": "rzepak", "season_year": 2026}).json()["date_to"] == "2026-06-10"
    assert app_client.delete(f"/api/yield/map/{mid}").status_code == 200
    assert not (tmp_path / str(mid)).exists()


def test_api_mapa_plonu_walidacja(app_client):
    r = app_client.post("/api/yield/mvi", json={"parcel_id": "pole-bez-obrysu", "crop_type": "kukurydza",
                                                "season_year": 2026, "reported_yield_t_ha": 9})
    assert r.status_code == 422
    r = app_client.post("/api/yield/mvi", json={"geometry_wkt": SQ_WKT, "crop_type": "kukurydza", "season_year": 2026,
                                                "reported_yield_t_ha": 9, "date_from": "2026-09-01", "date_to": "2026-06-01"})
    assert r.status_code == 422


def test_pas_brzegowy_pominiety(tmp_path, monkeypatch, fake_process_api):
    """Test na żywo: piksele na granicy pola (droga, miedza) dawały czerwoną
    obwódkę — pas EDGE_BUFFER_M jest pomijany, gdy pole jest dość duże."""
    from shapely import wkt
    monkeypatch.setattr(ym, "YIELD_MAPS_DIR", str(tmp_path))
    g = wkt.loads(SQ_WKT)
    full = ym.build_yield_map(g, 8.0, date(2026, 6, 1), date(2026, 9, 15), 8)
    monkeypatch.setattr(ym, "EDGE_BUFFER_M", 0.0)
    no_buf = ym.build_yield_map(g, 8.0, date(2026, 6, 1), date(2026, 9, 15), 9)
    assert full["field_pixels"] < no_buf["field_pixels"]


def test_nakladka_bez_ostrzezen_przy_nan():
    import warnings
    yld = np.array([[8.0, np.nan], [9.0, 7.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        png = ym.overlay_png(yld, np.isfinite(yld), 8.0)
    assert png[:4] == b"\x89PNG"
