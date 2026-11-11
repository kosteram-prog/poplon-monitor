"""Testy petli zwrotnej / rekalibracji progu anomalii wodnej
(water_stress.compute_suggested_water_threshold i apply_water_threshold).
Wzorem test_analytics.py (kalibracja wag) i test_db.py (kalibracja progow)."""
import json

import pytest

import water_stress as ws

SAMPLE_WKT = "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))"


def _alarm(fid, triage_status, own_ndwi_z=None, own_vv_z=None):
    components = {
        "own_ndwi": {"z_score": own_ndwi_z} if own_ndwi_z is not None else None,
        "own_vv": {"z_score": own_vv_z} if own_vv_z is not None else None,
        "cohort_ndwi": None,
    }
    return {"triage_status": triage_status, "components_json": json.dumps(components)}


# --- _trigger_z_from_alarm --------------------------------------------------

def test_trigger_z_bierze_bardziej_ujemny_z_dwoch_wskaznikow():
    a = _alarm(1, "confirmed", own_ndwi_z=-1.6, own_vv_z=-2.4)
    assert ws._trigger_z_from_alarm(a) == -2.4


def test_trigger_z_none_gdy_brak_components():
    assert ws._trigger_z_from_alarm({"components_json": None}) is None
    assert ws._trigger_z_from_alarm({"components_json": "not json"}) is None
    assert ws._trigger_z_from_alarm({"components_json": json.dumps({})}) is None


# --- compute_suggested_water_threshold -------------------------------------

def test_suggest_none_gdy_za_malo_probek(isolated_db):
    isolated_db.init_db()
    assert ws.compute_suggested_water_threshold(min_samples=10) is None


def test_suggest_zaostrza_prog_gdy_odrzucone_sa_na_granicy(isolated_db, monkeypatch):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)

    # potwierdzone: silny, jednoznaczny sygnal (bardzo ujemne z)
    confirmed_zs = [-2.6, -2.4, -2.5, -2.7, -2.3]
    # odrzucone: ledwo przekroczyly stary prog -1.5 -- klasyczne falsze alarmy
    rejected_zs = [-1.51, -1.55, -1.52, -1.58, -1.53]

    def _mk(zs, status):
        return [_alarm(fid, status, own_ndwi_z=z) for z in zs]

    monkeypatch.setattr(isolated_db, "get_water_alarms",
                        lambda **kw: _mk(confirmed_zs, "confirmed") + _mk(rejected_zs, "rejected"))

    result = ws.compute_suggested_water_threshold(min_samples=5)
    assert result is not None
    # nowy prog powinien byc bardziej ujemny niz stary -1.5 (zaostrzenie),
    # ale mniej ujemny niz najslabszy potwierdzony (-2.3), zeby go nie stracic
    assert result["suggested_threshold"] < -1.5
    assert result["suggested_threshold"] > -2.3
    assert result["confirmed_samples"] == 5
    assert result["rejected_samples"] == 5


def test_suggest_pomija_pending_i_needs_visit(isolated_db, monkeypatch):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    alarms = ([_alarm(fid, "confirmed", own_ndwi_z=-2.5) for _ in range(6)]
             + [_alarm(fid, "pending", own_ndwi_z=-1.6) for _ in range(10)]
             + [_alarm(fid, "needs_visit", own_ndwi_z=-1.6) for _ in range(10)])
    monkeypatch.setattr(isolated_db, "get_water_alarms", lambda **kw: alarms)
    # tylko 6 potwierdzonych, 0 odrzuconych -> ponizej min_samples=10 mimo 26 rekordow lacznie
    assert ws.compute_suggested_water_threshold(min_samples=10) is None


# --- apply_water_threshold / reload_water_threshold -------------------------

def test_apply_water_threshold_zapisuje_i_przeladowuje(isolated_db):
    isolated_db.init_db()
    assert ws.Z_SCORE_RISK_THRESHOLD == ws.DEFAULT_Z_SCORE_RISK_THRESHOLD
    ok = ws.apply_water_threshold(-2.0)
    assert ok is True
    assert ws.Z_SCORE_RISK_THRESHOLD == -2.0
    assert isolated_db.get_setting("water_zscore_risk_threshold") == "-2.00"


def test_reload_water_threshold_bez_zapisanego_ustawienia_uzywa_domyslnego(isolated_db):
    isolated_db.init_db()
    ws.reload_water_threshold()
    assert ws.Z_SCORE_RISK_THRESHOLD == ws.DEFAULT_Z_SCORE_RISK_THRESHOLD


def test_apply_water_threshold_wplywa_na_classify_water_signal(isolated_db):
    from datetime import date, timedelta
    isolated_db.init_db()

    def _mk_obs(days_ago, ndwi):
        return {"obs_date": (date.today() - timedelta(days=days_ago)).isoformat(),
                "ndwi_mean": ndwi, "vv_db": None, "dry_spell": False}

    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    # historia z wieksza naturalna zmiennoscia (stdev ~0.038), zeby spadek do
    # 0.22 dawal umiarkowany z-score (~-2.1) -- na tyle silny, by zlapac go
    # domyslnym progiem -1.5, ale nie na tyle ekstremalny, by zlapal go tez
    # bardzo surowy prog -3.0 uzywany w tescie ponizej.
    obs = [_mk_obs(d, v) for d, v in zip(range(5, 0, -1), [0.25, 0.35, 0.28, 0.32, 0.30])] \
        + [_mk_obs(0, 0.22)]  # spadek dajacy z ~ -2.1

    baseline = ws.classify_water_signal(field, obs, [], [])
    assert baseline["risk_flag"] is True  # przy domyslnym progu -1.5 to jest flagowane

    ws.apply_water_threshold(-3.0)  # bardzo surowy prog
    stricter = ws.classify_water_signal(field, obs, [], [])
    assert stricter["risk_flag"] is False  # ten sam spadek juz NIE jest flagowany


# --- get_water_calibration_status ------------------------------------------

def test_get_water_calibration_status_liczy_po_statusie(isolated_db, monkeypatch):
    isolated_db.init_db()
    alarms = [_alarm(1, "pending"), _alarm(1, "pending"), _alarm(1, "confirmed"), _alarm(1, "rejected")]
    monkeypatch.setattr(isolated_db, "get_water_alarms", lambda **kw: alarms)
    status = ws.get_water_calibration_status()
    assert status["total_alarms"] == 4
    assert status["by_triage_status"] == {"pending": 2, "confirmed": 1, "rejected": 1}
    assert status["current_z_score_threshold"] == ws.Z_SCORE_RISK_THRESHOLD


# --- warstwa API -------------------------------------------------------------

def test_api_calibration_status_i_suggest_not_enough(app_client):
    r = app_client.get("/api/water/calibration/status")
    assert r.status_code == 200
    assert r.json()["total_alarms"] == 0

    r2 = app_client.get("/api/water/calibration/suggest")
    assert r2.status_code == 200
    assert r2.json()["status"] == "not_enough_data"


def test_api_calibration_apply(app_client):
    r = app_client.post("/api/water/calibration/apply", json={"new_threshold": -2.2})
    assert r.status_code == 200
    assert r.json()["applied"]["z_score_risk_threshold"] == -2.2

    r2 = app_client.get("/api/water/calibration/status")
    assert r2.json()["current_z_score_threshold"] == -2.2


def test_api_calibration_apply_odrzuca_prog_poza_zakresem(app_client):
    r = app_client.post("/api/water/calibration/apply", json={"new_threshold": -10.0})
    assert r.status_code == 422
    r2 = app_client.post("/api/water/calibration/apply", json={"new_threshold": 0.0})
    assert r2.status_code == 422
