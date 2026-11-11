"""Termika Landsat (thermal.py)."""
import io
import json
import tarfile
from datetime import date, timedelta

import numpy as np
import pytest

import thermal as th

WKT = "POLYGON((69.0 41.0, 69.006 41.0, 69.006 41.0045, 69.0 41.0045, 69.0 41.0))"   # ~25 ha


def _scene(h=60, w=60, t_field=303.15, field_box=(20, 40, 20, 40), clouds_on_field=False, seed=0):
    """Syntetyczna scena: lewa połowa — gęsta, chłodna roślinność (NDVI 0.75,
    25 °C), prawa — goła gleba (NDVI 0.1, 45 °C), w środku pole."""
    rng = np.random.default_rng(seed)
    lst = np.empty((h, w)); nd = np.empty((h, w))
    lst[:, : w // 2] = 298.15 + rng.normal(0, 0.3, (h, w // 2)); nd[:, : w // 2] = 0.75
    lst[:, w // 2:] = 318.15 + rng.normal(0, 0.3, (h, w - w // 2)); nd[:, w // 2:] = 0.1
    r0, r1, c0, c1 = field_box
    lst[r0:r1, c0:c1] = t_field + rng.normal(0, 0.2, (r1 - r0, c1 - c0)); nd[r0:r1, c0:c1] = 0.65
    if clouds_on_field:
        lst[r0:r1, c0:c1 - 2] = np.nan
    inside = np.zeros((h, w), bool); inside[r0:r1, c0:c1] = True
    return lst, nd, inside


def test_lst_z_bt_emisyjnosc():
    assert th.lst_from_bt(300.0, 0.7) == pytest.approx(300.69, abs=0.02)   # ε=0.99
    assert th.lst_from_bt(300.0, 0.1) > th.lst_from_bt(300.0, 0.7)        # gleba ma niższą ε
    assert "Math.log(emissivity(ndvi))" in th.EVALSCRIPT and "decodeL8C2Qa" in th.EVALSCRIPT


def test_scene_metrics_tsi():
    lst, nd, inside = _scene(t_field=303.15)          # 30 °C: 1/4 drogi od 25 do 45 °C
    m = th.scene_metrics(lst, nd, inside)
    assert m["t_field"] == pytest.approx(30.0, abs=0.2)
    assert m["t_cold"] == pytest.approx(24.5, abs=0.4) and m["t_hot"] == pytest.approx(45.5, abs=0.4)
    assert m["tsi"] == pytest.approx(0.26, abs=0.03)
    assert m["valid_frac"] == 1.0 and m["ndvi"] == pytest.approx(0.65)


def test_scene_metrics_zachmurzone_pole():
    lst, nd, inside = _scene(clouds_on_field=True)
    assert th.scene_metrics(lst, nd, inside) is None


def test_bez_odniesienia_brak_tsi():
    lst, nd, inside = _scene()
    nd[~inside] = 0.4                                   # okolica bez gęstej roślinności i gołej gleby
    m = th.scene_metrics(lst, nd, inside)
    assert m["t_cold"] is None and m["t_hot"] is None and m["tsi"] is None
    assert "brak gęstej roślinności" in th.describe({"scene_date": "2026-07-01", **m})


def test_stack_metrics_daty_i_kolejnosc():
    a1 = _scene(t_field=303.15); a2 = _scene(t_field=313.15); a3 = _scene(clouds_on_field=True)
    arr = np.stack([a1[0], a1[1], a2[0], a2[1], a3[0], a3[1]])
    rows = th.stack_metrics(arr, ["2026-07-09T00:00:00Z", "2026-07-01T00:00:00Z", "2026-07-17T00:00:00Z"], a1[2])
    assert [r["scene_date"] for r in rows] == ["2026-07-01", "2026-07-09"]
    assert rows[0]["tsi"] > 0.7 > rows[1]["tsi"]


def test_describe_klasy_i_nierownomiernosc():
    base = {"scene_date": "2026-07-01", "t_field": 38.0, "t_cold": 26.0, "t_hot": 46.0, "ndvi": 0.6,
            "t_p10": 34.0, "t_p90": 41.0}
    s = th.describe({**base, "tsi": 0.6})
    assert "prawdopodobny stres" in s and "nierówne nawodnienie" in s and "+12.0 K" in s
    s = th.describe({**base, "tsi": 0.1, "t_p90": 35.0, "ndvi": 0.2})
    assert "bez oznak stresu" in s and "głównie glebę" in s and "nierówne" not in s


def _tar(arr, dates):
    import rasterio
    from rasterio.transform import from_origin
    mem = rasterio.MemoryFile()
    with mem.open(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                  dtype="float32", transform=from_origin(69.0, 41.01, 0.0003, 0.0003)) as ds:
        ds.write(arr.astype("float32"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in (("default.tif", mem.read()), ("userdata.json", json.dumps({"dates": dates}).encode())):
            ti = tarfile.TarInfo(name); ti.size = len(data); tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_parse_tar():
    a = _scene()
    arr = np.stack([a[0], a[1]])
    out, transform, dates = th.parse_tar(_tar(arr, ["2026-07-01T00:00:00Z"]))
    assert out.shape == (2, 60, 60) and dates == ["2026-07-01T00:00:00Z"]
    assert transform.a == pytest.approx(0.0003)


# --- pole z zapisem ------------------------------------------------------------

def _fake_stack(today):
    def f(region, a, b):
        from rasterio.transform import from_bounds
        minx, miny, maxx, maxy = region.bounds
        h = w = 120
        tr = from_bounds(minx, miny, maxx, maxy, w, h)
        layers, dates = [], []
        for k, tf in enumerate((303.15, 308.15, 315.15)):
            d = today - timedelta(days=16 - 8 * k)
            if not (a <= d <= b):
                continue
            rng = np.random.default_rng(k)
            lst = np.where(np.arange(w)[None, :] < w // 2, 298.15, 318.15) + rng.normal(0, 0.3, (h, w))
            nd = np.where(np.arange(w)[None, :] < w // 2, 0.75, 0.1) * np.ones((h, w))
            lst[50:70, 50:70] = tf; nd[50:70, 50:70] = 0.6
            layers += [lst, nd]; dates.append(d.isoformat() + "T00:00:00Z")
        return np.stack(layers), tr, dates
    return f


@pytest.fixture
def field(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-1", "irrigation", WKT)
    return isolated_db.get_water_field_by_id(fid)


def test_fetch_field_i_zdanie(field, isolated_db, monkeypatch):
    today = date.today()
    monkeypatch.setattr(th, "fetch_stack", _fake_stack(today))
    rows = th.fetch_field(field, today - timedelta(days=60), today)
    assert len(rows) == 3
    out = th.get_field_thermal(field)
    assert out["has_data"] and out["latest"]["scene_date"] == today.isoformat()
    assert out["latest"]["tsi"] > 0.5 and not out["small_field"]
    v = th.attach_thermal(field, {"message_pl": "Spadek NDMI.", "risk_flag": True})
    assert "Termika Landsat" in v["message_pl"] and "potwierdza" in v["message_pl"]
    assert th.update_recent(field, days=10) == 2


def test_stara_scena_bez_zdania(field, isolated_db):
    old = (date.today() - timedelta(days=40)).isoformat()
    isolated_db.upsert_thermal(field["id"], [{"scene_date": old, "t_field": 30, "tsi": 0.2, "ndvi": 0.6}])
    assert th.thermal_sentence(field) == ("", None)
    assert th.attach_thermal(field, {"message_pl": "X."})["message_pl"] == "X."


def test_update_recent_bez_danych_nic_nie_pobiera(field, monkeypatch):
    monkeypatch.setattr(th, "fetch_stack", lambda *a: (_ for _ in ()).throw(AssertionError("API")))
    assert th.update_recent(field) == 0


def test_male_pole_ostrzezenie(isolated_db):
    isolated_db.init_db()
    small = "POLYGON((69.0 41.0, 69.001 41.0, 69.001 41.001, 69.0 41.001, 69.0 41.0))"
    fid = isolated_db.add_water_field("S", "irrigation", small)
    assert th.get_field_thermal(isolated_db.get_water_field_by_id(fid))["small_field"]


def test_okres_przed_2021(field):
    with pytest.raises(ValueError):
        th.fetch_field(field, date(2019, 1, 1), date(2020, 6, 1))


def test_api(app_client, isolated_db, monkeypatch):
    fid = isolated_db.add_water_field("UZ-1", "irrigation", WKT)
    assert app_client.get(f"/api/water/field/{fid}/thermal").json()["has_data"] is False
    monkeypatch.setattr(th, "fetch_stack", _fake_stack(date.today()))
    r = app_client.post(f"/api/water/field/{fid}/thermal/fetch", json={"months": 2})
    assert r.status_code == 200 and r.json()["added"] == 3 and r.json()["has_data"]
    assert app_client.post(f"/api/water/field/{fid}/thermal/fetch", json={"months": 0}).status_code == 422
    assert app_client.delete(f"/api/water/field/{fid}/thermal").json() == {"deleted": True}
    assert app_client.get(f"/api/water/field/{fid}/thermal").json()["has_data"] is False
