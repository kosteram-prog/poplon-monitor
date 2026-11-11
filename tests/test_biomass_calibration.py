"""CIred-edge i kalibracja biomasy/N poplonu (analytics + biomass_calibration)."""
import math
from datetime import date, timedelta

import pytest

import analytics
import biomass_calibration as bc


def _samples(n=14, form="log", noise=0.0):
    out = []
    for i in range(n):
        x = 0.4 + 3.6 * i / (n - 1)                  # CIre 0.4 .. 4.0
        dm = math.exp(-0.9 + 0.62 * x) if form == "log" else 0.2 + 0.9 * x
        dm *= (1 + noise * (1 if i % 2 else -1))
        out.append({"id": i + 1, "parcel_id": f"P-{i:02d}", "sample_date": "2026-10-20", "cire": x,
                    "dm_t_ha": round(dm, 3), "n_kg_ha": round(25 * x + 5, 1),
                    "ndre": 0.1 + 0.08 * x, "ndvi": min(0.92, 0.3 + 0.2 * x), "evi": 0.2 + 0.12 * x})
    return out


# --- evalscript i odczyt --------------------------------------------------------

def test_evalscript_ma_cire():
    for es in (analytics.EVALSCRIPT_MULTI, analytics.EVALSCRIPT_MULTI_RELAXED):
        assert '{ id: "cire"' in es and "RE3 / (RE1 + 1e-9) - 1.0" in es
        assert "cire:     [Math.round(1000" in es and "cire: [0], dataMask: [0]" in es
    assert analytics.CIRE_SCALE == 1000.0


# --- dopasowanie --------------------------------------------------------------------

def test_log_liniowy_odtwarza_wspolczynniki():
    s = _samples(form="log")
    m = bc.evaluate_form([x["cire"] for x in s], [x["dm_t_ha"] for x in s], "log")
    assert m["a"] == pytest.approx(-0.9, abs=0.01) and m["b"] == pytest.approx(0.62, abs=0.01)
    assert m["rmse_cv"] < 0.02 and m["smear"] == pytest.approx(1.0, abs=1e-3)
    best = bc.best_cire_model([x["cire"] for x in s], [x["dm_t_ha"] for x in s])
    assert best["form"] == "log"


def test_liniowy_wygrywa_dla_liniowych_danych():
    s = _samples(form="linear", noise=0.05)
    best = bc.best_cire_model([x["cire"] for x in s], [x["dm_t_ha"] for x in s])
    assert best["form"] == "linear" and best["r2"] > 0.95


def test_log_odrzucony_przy_zerowej_biomasie():
    assert bc.evaluate_form([0.1, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0], "log") is None


def test_brak_zmiennosci():
    assert bc.evaluate_form([1.0] * 5, [1, 2, 3, 4, 5], "linear") is None


def test_raport_porownanie_i_ostrzezenia():
    rep = bc.fit_report(_samples(n=6))
    assert rep["n_usable"] == 6 and not rep["can_apply_biomass"]
    assert any("Za mało próbek" in w for w in rep["warnings"])
    assert {c["index"] for c in rep["compare"]} == {"NDRE", "NDVI", "EVI2"}
    assert rep["literature"]["n"] == 6 and rep["n_uptake"]["form"] == "linear"
    rep = bc.fit_report(_samples(n=14))
    assert rep["can_apply_biomass"] and rep["can_apply_n"]


def test_waski_zakres_ostrzezenie():
    s = _samples(n=10)
    for x in s:
        x["cire"] = 1.0 + x["cire"] / 10
    assert any("wąski zakres" in w for w in bc.fit_report(s)["warnings"])


# --- estimate_biomass z modelem ------------------------------------------------------

def test_estimate_bez_modelu_jak_dotad():
    res = analytics.estimate_biomass(0.4, 0.2, rvi=0.3, cire=2.0)
    assert res["model_source"] == "literature" and res["cire"] == 2.0
    assert analytics.estimate_biomass(None, None, cire=2.0) is None


def test_estimate_z_modelem_cire():
    analytics.BIOMASS_CIRE_MODEL = {"form": "log", "a": -0.9, "b": 0.62, "smear": 1.0, "n": 20,
                                    "rmse_cv": 0.3, "r2": 0.8, "x_min": 0.4, "x_max": 4.0}
    analytics.N_CIRE_MODEL = {"form": "linear", "a": 5, "b": 25, "smear": 1.0, "n": 12,
                              "rmse_cv": 8.0, "r2": 0.7, "x_min": 0.4, "x_max": 4.0}
    res = analytics.estimate_biomass(None, None, cire=2.0)
    assert res["model_source"] == "field_calibrated"
    assert res["biomass_t_ha"] == pytest.approx(math.exp(-0.9 + 1.24), abs=0.01)
    assert res["biomass_low_t_ha"] == pytest.approx(res["biomass_t_ha"] - 0.3, abs=0.01)
    assert res["n_uptake_kg_ha"] == 55 and res["confidence"] == "medium"
    far = analytics.estimate_biomass(None, None, cire=6.0)
    assert far["cire_extrapolated"] and far["confidence"] == "low"


# --- API + baza + przeliczenie zapisanych obserwacji -----------------------------------

def test_api_pelny_cykl(app_client, isolated_db, monkeypatch):
    samples = _samples(n=10)
    it = iter(samples)

    def fake_match(sample):
        s = next(it)
        return {"matched": True, "scene_date": sample["sample_date"], "delta_days": 1,
                "cire": s["cire"], "ndre": s["ndre"], "ndvi": s["ndvi"], "evi": s["evi"], "valid_frac": 0.9}
    monkeypatch.setattr(bc, "match_scene", fake_match)
    for s in samples:
        r = app_client.post("/api/biomass/samples", json={
            "parcel_id": s["parcel_id"], "sample_date": "2026-10-20", "dm_t_ha": s["dm_t_ha"],
            "n_kg_ha": s["n_kg_ha"], "mix": "gorczyca+facelia", "fresh_g": 500, "frame_m2": 0.25,
            "n_frames": 4, "dm_pct": 14})
        assert r.status_code == 200 and r.json()["match"]["matched"]
    assert len(app_client.get("/api/biomass/samples").json()) == 10
    fut = (date.today() + timedelta(days=2)).isoformat()
    assert app_client.post("/api/biomass/samples", json={"parcel_id": "X1", "sample_date": fut,
                                                         "dm_t_ha": 1}).status_code == 422

    # zapisana obserwacja do przeliczenia po zatwierdzeniu
    import sqlite3
    analytics.init_analytics_tables(isolated_db.DB_PATH)
    with sqlite3.connect(isolated_db.DB_PATH) as c:
        c.execute("INSERT INTO multi_observations (parcel_db_id, obs_date, evi, ndre, cire, valid_frac) "
                  "VALUES (1, '2026-10-01', 0.4, 0.2, 2.0, 0.9)")
    fit = app_client.get("/api/biomass/fit").json()
    assert fit["can_apply_biomass"] and fit["active"]["biomass"] is None
    r = app_client.post("/api/biomass/model/apply", json={"target": "biomass"})
    assert r.status_code == 200 and r.json()["model"]["form"] == "log" and r.json()["recomputed_observations"] == 1
    with sqlite3.connect(isolated_db.DB_PATH) as c:
        bm = c.execute("SELECT biomass_t_ha FROM multi_observations").fetchone()[0]
    assert bm == pytest.approx(math.exp(-0.9 + 0.62 * 2.0), abs=0.05)
    assert analytics.BIOMASS_CIRE_MODEL["n"] == 10
    assert app_client.post("/api/biomass/model/apply", json={"target": "n", "form": "linear"}).status_code == 200
    assert app_client.get("/api/biomass/fit").json()["active"]["n_uptake"]["form"] == "linear"
    assert app_client.delete("/api/biomass/model/biomass").json() == {"cleared": "biomass"}
    assert analytics.BIOMASS_CIRE_MODEL is None
    sid = app_client.get("/api/biomass/samples").json()[0]["id"]
    assert app_client.delete(f"/api/biomass/samples/{sid}").json() == {"deleted": True}


def test_apply_za_malo_probek(app_client):
    assert app_client.post("/api/biomass/model/apply", json={"target": "biomass"}).status_code == 409


def test_match_scene_wybiera_najblizsza_czysta(monkeypatch):
    import core
    monkeypatch.setattr(core, "fetch_parcel_geometry", lambda pid: ("G", {}))
    monkeypatch.setattr(core, "inner_buffer", lambda g, m: (g, None, None))
    d0 = date(2026, 10, 20)
    recs = [{"date": d0 - timedelta(days=1), "valid_frac": 0.3, "cire": 9.0},
            {"date": d0 + timedelta(days=3), "valid_frac": 0.9, "cire": 2.5, "ndre": 0.3, "ndvi": 0.8, "evi": 0.5},
            {"date": d0 - timedelta(days=7), "valid_frac": 1.0, "cire": 1.0}]
    monkeypatch.setattr(analytics, "fetch_multi_series_strict", lambda g, a, b: recs)
    m = bc.match_scene({"parcel_id": "P", "sample_date": d0.isoformat()})
    assert m["matched"] and m["cire"] == 2.5 and m["delta_days"] == 3
    monkeypatch.setattr(analytics, "fetch_multi_series_strict", lambda g, a, b: recs[:1])
    assert bc.match_scene({"parcel_id": "P", "sample_date": d0.isoformat()}) == {"matched": False}
