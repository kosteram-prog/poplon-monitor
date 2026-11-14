"""OPTRAM — wilgotność wierzchniej warstwy z Sentinel-2 (optram.py)."""
from datetime import date, timedelta

import numpy as np
import pytest

import optram as o

WKT = "POLYGON((69.0 41.0, 69.004 41.0, 69.004 41.003, 69.0 41.003, 69.0 41.0))"
TRUE = dict(i_d=0.6, s_d=1.2, i_w=2.5, s_w=7.0)


def _synthetic(n=60000, seed=1, ndvi_hi=0.85):
    rng = np.random.default_rng(seed)
    nd = rng.uniform(0.05, ndvi_hi, n)
    w = rng.beta(1.2, 1.2, n)
    d = TRUE["i_d"] + TRUE["s_d"] * nd
    wet = TRUE["i_w"] + TRUE["s_w"] * nd
    return nd, d + w * (wet - d) + rng.normal(0, 0.05, n)


def _stack(nd, st, n_scenes=12):
    """Upakuj punkty w tablicę jak z Process API: (2·N, H, W) INT16."""
    per = len(nd) // n_scenes
    side = int(np.ceil(np.sqrt(per)))
    arr = np.full((2 * n_scenes, side, side), o.NODATA, dtype="int16")
    for k in range(n_scenes):
        a = np.full(side * side, o.NODATA, dtype="int32")
        b = np.full(side * side, o.NODATA, dtype="int32")
        a[:per] = np.round(nd[k * per:(k + 1) * per] * 10000)
        b[:per] = np.round(np.minimum(32, st[k * per:(k + 1) * per]) * 1000)
        arr[2 * k] = a.reshape(side, side)
        arr[2 * k + 1] = b.reshape(side, side)
    return arr


# --- matematyka ---------------------------------------------------------------

def test_str_kubelka_munk():
    assert o.str_from_reflectance(0.5) == pytest.approx(0.25)
    assert o.str_from_reflectance(0.1) == pytest.approx(4.05)


def test_fit_odtwarza_krawedzie():
    nd, st = _synthetic()
    p = o.fit_trapezoid(nd, st)
    # kwantyle 2/98% leżą nieco do środka trapezu — tolerancja na to
    assert p["i_d"] == pytest.approx(TRUE["i_d"], abs=0.15)
    assert p["s_d"] == pytest.approx(TRUE["s_d"], abs=0.3)
    assert p["i_w"] == pytest.approx(TRUE["i_w"], abs=0.2)
    assert p["s_w"] == pytest.approx(TRUE["s_w"], abs=0.5)
    assert min(p["r2_d"], p["r2_w"]) > 0.9
    q, warn = o.assess_quality(p, 40)
    assert q == "good" and warn == []


def test_w_srodek_trapezu():
    p = dict(TRUE)
    nd = 0.5
    mid = (TRUE["i_d"] + TRUE["s_d"] * nd + TRUE["i_w"] + TRUE["s_w"] * nd) / 2
    assert o.w_from(nd, mid, p) == pytest.approx(0.5)
    assert o.w_from(nd, 0.0, p) == 0.0 and o.w_from(nd, 99.0, p) == 1.0


def test_fit_odrzuca_waski_zakres_i_malo_pikseli():
    with pytest.raises(ValueError):
        o.fit_trapezoid([0.3] * 100, [1.0] * 100)
    nd, st = _synthetic(n=20000, ndvi_hi=0.2)
    with pytest.raises(ValueError, match="wąski zakres"):
        o.fit_trapezoid(nd, st)


def test_jakosc_krzyzujace_krawedzie():
    p = {"i_d": 1.0, "s_d": 3.0, "i_w": 2.0, "s_w": 1.0, "r2_d": 0.9, "r2_w": 0.9,
         "ndvi_lo": 0.1, "ndvi_hi": 0.8, "n_px": 50000}
    q, warn = o.assess_quality(p, 30)
    assert q == "poor" and any("krzyżują" in w for w in warn)


def test_jakosc_ograniczona_malo_scen():
    nd, st = _synthetic()
    p = o.fit_trapezoid(nd, st)
    q, warn = o.assess_quality(p, 4)
    assert q == "limited" and any("Liczba scen" in w for w in warn)


def test_stack_to_points():
    nd, st = _synthetic(n=12000)
    arr = _stack(nd, st, n_scenes=6)
    arr = np.concatenate([arr, np.full((2,) + arr.shape[1:], o.NODATA, dtype="int16")])  # pusta scena
    x, y, n = o.stack_to_points(arr)
    assert n == 6 and len(x) == 12000
    assert x.min() >= 0.05 - 1e-4 and y.max() <= 32


def test_evalscripty():
    assert "__MAX_SCENES__" not in o.EVALSCRIPT_STACK.replace("__MAX_SCENES__", "60")
    assert "updateOutput" in o.EVALSCRIPT_STACK and "ORBIT" in o.EVALSCRIPT_STACK
    es = o.series_evalscript({"i_d": 0.5, "s_d": 1.25, "i_w": 2.0, "s_w": 6.5})
    assert "const ID = 0.5, SD = 1.25, IW = 2.0, SW = 6.5;" in es and "__" not in es


def test_region_kalibracji():
    from shapely import wkt
    from pyproj import Geod
    reg = o.calibration_region(wkt.loads(WKT), 1500)
    minx, miny, maxx, maxy = reg.bounds
    side = Geod(ellps="WGS84").inv(minx, miny, maxx, miny)[2]
    assert 3000 < side < 3600 and reg.contains(wkt.loads(WKT))


def test_parse_series_filtruje_zachmurzone():
    def item(day, w, nd, n_all, n_bad):
        st = lambda m: {"stats": {"mean": m, "sampleCount": n_all, "noDataCount": n_bad}}
        return {"interval": {"from": f"{day}T00:00:00Z"},
                "outputs": {"w": {"bands": {"B0": st(w)}}, "ndvi": {"bands": {"B0": st(nd)}}}}
    data = {"data": [item("2026-06-01", 4200, 5100, 100, 5), item("2026-06-06", 9000, 3000, 100, 80)]}
    rows = o.parse_series(data, 100)
    assert rows == [{"obs_date": "2026-06-01", "w": 0.42, "ndvi": 0.51, "valid_frac": 0.95}]


def test_pearson_i_percentyl():
    assert o.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert o.pearson([1, 1, 1], [1, 2, 3]) is None
    assert o.percentile_of(5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == pytest.approx(45.0)


# --- kalibracja pola z zapisem (Sentinel Hub zamockowany) ---------------------

@pytest.fixture
def field(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-1", "rainfed", WKT)
    return isolated_db.get_water_field_by_id(fid)


def _mock_sh(monkeypatch, series=None, n_scenes=12):
    nd, st = _synthetic()
    monkeypatch.setattr(o, "fetch_stack", lambda region, a, b: _stack(nd, st, n_scenes))
    today = date.today()
    series = series if series is not None else [
        {"obs_date": (today - timedelta(days=5 * k)).isoformat(), "w": round(0.2 + 0.03 * k, 3),
         "ndvi": 0.4, "valid_frac": 0.9} for k in range(12)][::-1]
    calls = []

    def fake_series(g, a, b, p):
        calls.append((a, b, p["i_d"]))
        return [s for s in series if a.isoformat() <= s["obs_date"] <= b.isoformat()]
    monkeypatch.setattr(o, "fetch_series", fake_series)
    return calls


def test_calibrate_field_zapisuje_i_opisuje(field, isolated_db, monkeypatch):
    _mock_sh(monkeypatch)
    for k in range(12):
        d = (date.today() - timedelta(days=5 * k)).isoformat()
        isolated_db.add_water_observation(field["id"], d, ndvi_mean=0.4, sm_0_7=0.10 + 0.01 * k)
    res = o.calibrate_field(field)
    assert res["calibrated"] and res["params"]["quality"] == "good"
    assert res["params"]["n_scenes"] == 12 and len(res["scatter"]) == o.SCATTER_N
    assert len(res["series"]) == 12
    cur = res["current"]
    assert cur["obs_date"] == date.today().isoformat() and cur["w"] == 0.2 and cur["class"] == "sucho"
    assert res["era5"]["n"] == 12 and res["era5"]["r"] is not None
    msg = o.optram_sentence(field["id"])
    assert "OPTRAM" in msg and "sucho" in msg and "percentyl" in msg


def test_zdanie_puste_gdy_stare_albo_slaba_kalibracja(field, isolated_db, monkeypatch):
    old = (date.today() - timedelta(days=40)).isoformat()
    _mock_sh(monkeypatch, series=[{"obs_date": old, "w": 0.5, "ndvi": 0.3, "valid_frac": 1}])
    o.calibrate_field(field)
    assert o.optram_sentence(field["id"]) == ""
    p = isolated_db.get_optram_params(field["id"])
    isolated_db.save_optram_params(field["id"], {**p, "quality": "poor"})
    isolated_db.upsert_optram_series(field["id"], [{"obs_date": date.today().isoformat(), "w": 0.5, "ndvi": 0.3}])
    assert o.optram_sentence(field["id"]) == ""
    assert o.update_recent(field) == 0


def test_update_recent_dopisuje(field, isolated_db, monkeypatch):
    calls = _mock_sh(monkeypatch)
    o.calibrate_field(field)
    isolated_db.clear_optram_series(field["id"])
    assert o.update_recent(field, days=20) == 0          # tuż po kalibracji — pominięte (bez kosztu PU)
    n = o.update_recent(field, days=20, force=True)
    assert n == 5 and len(isolated_db.get_optram_series(field["id"])) == 5
    assert (calls[-1][1] - calls[-1][0]).days == 20


def test_brak_kalibracji(field):
    assert o.get_field_optram(field["id"]) == {"calibrated": False}
    assert o.update_recent(field) == 0
    v = o.attach_optram(field, {"message_pl": "X."})
    assert v["message_pl"] == "X."


def test_attach_optram_dokleja(field, monkeypatch):
    _mock_sh(monkeypatch)
    o.calibrate_field(field)
    v = o.attach_optram(field, {"message_pl": "Pole OK."})
    assert v["message_pl"].startswith("Pole OK. Wilgotność") and v["optram_sentence"]


def test_kalibracja_za_krotki_okres(field):
    with pytest.raises(ValueError, match="60 dni"):
        o.calibrate_field(field, date.today() - timedelta(days=30), date.today())


# --- API ------------------------------------------------------------------------

def test_api_optram(app_client, isolated_db, monkeypatch):
    fid = isolated_db.add_water_field("UZ-1", "rainfed", WKT)
    assert app_client.get(f"/api/water/field/{fid}/optram").json() == {"calibrated": False}
    assert app_client.post(f"/api/water/field/{fid}/optram/refresh").status_code == 409
    _mock_sh(monkeypatch)
    r = app_client.post(f"/api/water/field/{fid}/optram/calibrate", json={"months": 6})
    assert r.status_code == 200 and r.json()["calibrated"]
    assert r.json()["params"]["date_from"] == (date.today() - timedelta(days=int(6 * 30.4))).isoformat()
    assert app_client.post(f"/api/water/field/{fid}/optram/calibrate", json={"months": 1}).status_code == 422
    assert app_client.post(f"/api/water/field/{fid}/optram/refresh").json()["calibrated"]
    assert app_client.delete(f"/api/water/field/{fid}/optram").json() == {"deleted": True}
    assert app_client.get(f"/api/water/field/{fid}/optram").json() == {"calibrated": False}


def test_api_optram_blad_dopasowania_422(app_client, isolated_db, monkeypatch):
    fid = isolated_db.add_water_field("UZ-1", "rainfed", WKT)
    monkeypatch.setattr(o, "fetch_stack", lambda *a: np.full((2, 5, 5), o.NODATA, dtype="int16"))
    r = app_client.post(f"/api/water/field/{fid}/optram/calibrate", json={})
    assert r.status_code == 422 and "Za mało" in r.json()["detail"]


def test_zdanie_pominiete_przy_niezgodnosci_z_era5(field, isolated_db, monkeypatch):
    """Test na żywo (pole deszczowe w PL) dał r = -0,43 z ERA5-Land — wtedy
    zdanie nie trafia do werdyktu (panel nadal pokazuje dane z ostrzeżeniem)."""
    _mock_sh(monkeypatch)
    for k in range(12):
        d = (date.today() - timedelta(days=5 * k)).isoformat()
        isolated_db.add_water_observation(field["id"], d, ndvi_mean=0.4, sm_0_7=0.30 - 0.01 * k)
    res = o.calibrate_field(field)
    assert res["era5"]["r"] < 0 and o.optram_sentence(field["id"]) == ""


def test_despike_usuwa_pojedynczy_skok_ale_nie_trwala_zmiane():
    base = date(2026, 7, 1)
    rows = [{"obs_date": (base + timedelta(days=3 * k)).isoformat(), "w": 0.2, "ndvi": 0.7} for k in range(10)]
    rows[4] = {**rows[4], "w": 1.0}                         # cień chmury — jeden dzień
    for r in rows[7:]:
        r["w"] = 0.8                                         # deszcz — utrzymuje się
    kept = [r["obs_date"] for r in o.despike(rows)]
    assert rows[4]["obs_date"] not in kept and all(r["obs_date"] in kept for r in rows[7:])


def test_mediana_z_percentyli():
    assert o._median({"mean": 4000, "percentiles": {"50.0": 2500}}) == 2500
    assert o._median({"mean": 4000}) == 4000



def test_update_recent_okno_od_ostatniej_sceny(field, isolated_db, monkeypatch):
    calls = _mock_sh(monkeypatch)
    o.calibrate_field(field)
    isolated_db.set_setting(o._refresh_marker(field["id"]), (date.today() - timedelta(days=3)).isoformat())
    o.update_recent(field)
    a, b, _ = calls[-1]
    # ostatnia scena = dziś, więc okno = SPIKE_WINDOW_DAYS + 2, a nie pełne 30 dni
    assert (b - a).days == o.SPIKE_WINDOW_DAYS + 2
