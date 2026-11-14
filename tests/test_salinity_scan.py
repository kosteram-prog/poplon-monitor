"""Podejrzenie zasolenia z wielolecia Sentinel-2 (salinity_scan.py)."""
from datetime import date

import numpy as np
from rasterio.transform import from_origin

import salinity_scan as ss

H, W = 20, 30
TR = from_origin(66.900, 39.050, 0.000116, 0.0000898)      # ~10 m
INSIDE = np.ones((H, W), dtype=bool)
SALTY = (slice(2, 8), slice(2, 10))       # słabo co roku + jasno wiosną
WET = (slice(12, 18), slice(20, 28))      # słabo co roku, wiosną normalnie (np. podmokłość)


def year_arr(peak=0.75, weak=True, bright=True, fallow=False):
    a = np.zeros((4, H, W), dtype="float32")
    a[0] = 0.15 if fallow else peak
    a[1] = 0.10
    a[2], a[3] = 8, 4
    if weak and not fallow:
        a[0][SALTY] = peak * 0.7
        a[0][WET] = peak * 0.7
    if bright:
        a[1][SALTY] = 0.14
    return a


def test_klasy_slaby_i_jasny_vs_tylko_slaby():
    stack = [(y, year_arr()) for y in range(2021, 2027)]
    r = ss.classify(stack, INSIDE)
    assert r["classes"][4, 5] == 3          # słaby łan co roku + jasna gleba wiosną
    assert r["classes"][14, 24] == 2        # tylko słaby łan
    assert r["classes"][0, 0] == 0
    assert r["years_vigor"] == list(range(2021, 2027))


def test_ugor_pominiety_i_za_malo_lat():
    stack = [(2021, year_arr(fallow=True)), (2022, year_arr(fallow=True)), (2023, year_arr()), (2024, year_arr())]
    r = ss.classify(stack, INSIDE)
    assert r["years_skipped"] == [2021, 2022] and r["years_vigor"] == [2023, 2024]
    assert not r["valid"].any() and (r["classes"] == 0).all()     # 2 lata < MIN_VALID_YEARS


def test_jednorazowy_slaby_rok_to_nie_podejrzenie():
    stack = [(y, year_arr(weak=(y == 2023), bright=False)) for y in range(2021, 2027)]
    assert (ss.classify(stack, INSIDE)["classes"] == 0).all()


def test_punkty_pomiarowe_i_kontrolny():
    stack = [(y, year_arr()) for y in range(2021, 2027)]
    r = ss.classify(stack, INSIDE)
    pts = ss.sampling_points(r["classes"], r["valid"], r["rel_mean"], TR)
    suspects = [p for p in pts if p["kind"] == "suspect"]
    assert len(suspects) == 2 and pts[-1]["kind"] == "control" and pts[-1]["n"] == "K"
    # punkt 1 w plamie (obie plamy tej samej wielkości) — w obrębie rastra
    for p in pts:
        assert 66.900 < p["lon"] < 66.900 + W * 0.000116 and 39.050 - H * 0.0000898 < p["lat"] < 39.050


def test_lata_sezonu_bez_biezacego_przed_koncem_okna():
    assert ss.season_years("cotton", date(2026, 9, 1)) == list(range(2020, 2026))
    assert ss.season_years("cotton", date(2026, 11, 14)) == list(range(2021, 2027))


def test_scan_field_zapis_i_api(app_client, isolated_db, monkeypatch):
    import yield_mvi
    isolated_db.init_db()
    wkt = "POLYGON((66.9 39.05, 66.90348 39.05, 66.90348 39.048204, 66.9 39.048204, 66.9 39.05))"
    fid = isolated_db.add_water_field("UZ-S", "irrigation", wkt, crop_type="cotton")
    monkeypatch.setattr(yield_mvi, "field_mask", lambda g, shape, tr: np.ones(shape, dtype=bool))
    calls = []

    def fake(geom, year, crop):
        calls.append(year)
        if year == 2022:
            raise RuntimeError("Sentinel Hub 429")
        return year_arr(), TR
    out = ss.scan_field(isolated_db.get_water_field_by_id(fid), today=date(2026, 11, 14), fetch=fake)
    assert calls == list(range(2021, 2027)) and out["errors"][0]["year"] == 2022
    assert out["status"] == "ok" and out["suspect_pct"] > 0 and out["share_pct"]["3"] > 0
    # sekcja gleby pokazuje wynik i wskazówkę „zmierz EC”
    d = app_client.get(f"/api/water/field/{fid}/soil").json()
    assert d["salinity_scan"]["suspect_pct"] == out["suspect_pct"]
    assert "sat_suspect" in d["salinity"]["hints"] and d["salinity_scan_pu"] == 6.0
    png = app_client.get(f"/api/water/field/{fid}/salinity_scan.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    # po pomiarze EC wskazówka znika
    app_client.put(f"/api/water/field/{fid}/soil", json={"ec_value": 1.0, "ec_method": "ec_1_5"})
    assert "sat_suspect" not in app_client.get(f"/api/water/field/{fid}/soil").json()["salinity"]["hints"]


def test_api_skan_bledy(app_client, isolated_db, monkeypatch):
    fid = isolated_db.add_water_field("UZ-S", "irrigation",
                                      "POLYGON((66.9 39.05, 66.901 39.05, 66.901 39.049, 66.9 39.049, 66.9 39.05))")
    assert app_client.get(f"/api/water/field/{fid}/salinity_scan.png").status_code == 404

    def boom(*a, **k):
        raise RuntimeError("Brak danych Sentinel-2 dla żadnego roku")
    monkeypatch.setattr(ss, "scan_field", boom)
    r = app_client.post(f"/api/water/field/{fid}/salinity_scan")
    assert r.status_code == 502 and "Brak danych" in r.json()["detail"]


def test_brzeg_pola_nie_jest_oceniany():
    """Piksele brzegowe (droga, rów) rok w rok słabsze — nie mogą dawać podejrzenia."""
    inside = np.zeros((H, W), dtype=bool)
    inside[2:18, 2:28] = True
    core = ss.core_mask(inside)
    assert core.sum() == (16 - 2) * (26 - 2) and not core[2, 5] and core[3, 5]
    stack = []
    for y in range(2021, 2027):
        a = year_arr(weak=False, bright=False)
        a[0][2, :] = 0.3                     # słaby pas na brzegu
        stack.append((y, a))
    r = ss.classify(stack, core)
    assert (r["classes"] == 0).all()
    small = np.zeros((H, W), dtype=bool); small[5:9, 5:9] = True      # 16 px — za małe, bez odcinania
    assert (ss.core_mask(small) == small).all()
