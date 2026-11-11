"""Testy bilansu wodnego strefy korzeniowej (water_balance.py, FAO-56)."""
from datetime import date, timedelta

import pytest

import water_balance as wb

SAMPLE_WKT = "POLYGON((66.9 39.6, 66.901 39.6, 66.901 39.601, 66.9 39.601, 66.9 39.6))"


def _wx(start, n, precip=0.0, et0=5.0, forecast_from=None):
    out = []
    for i in range(n):
        d = start + timedelta(days=i)
        out.append({"date": d, "precip_mm": precip, "et0_mm": et0,
                    "is_forecast": forecast_from is not None and d > forecast_from})
    return out


def test_crop_params_znane_i_domyslne():
    assert wb.crop_params("Bawełna ") == (1.15, 0.65, True)
    assert wb.crop_params("cotton")[1] == 0.65
    assert wb.crop_params("egzotyczna")[2] is False
    assert wb.crop_params(None)[:2] == (wb.DEFAULT_KCB_MAX, wb.DEFAULT_P)


def test_kcb_z_ndvi_w_granicach():
    assert wb.kcb_from_ndvi(0.10, 1.15) == pytest.approx(wb.KCB_MIN)
    assert wb.kcb_from_ndvi(0.90, 1.15) == pytest.approx(1.15)
    assert wb.KCB_MIN < wb.kcb_from_ndvi(0.5, 1.15) < 1.15


def test_interpolacja_ndvi():
    pts = [(date(2026, 6, 1), 0.4), (date(2026, 6, 11), 0.8)]
    assert wb.interpolate_ndvi(pts, date(2026, 6, 6)) == pytest.approx(0.6)
    assert wb.interpolate_ndvi(pts, date(2026, 5, 1)) == 0.4
    assert wb.interpolate_ndvi(pts, date(2026, 7, 1)) == 0.8
    assert wb.interpolate_ndvi([], date(2026, 7, 1)) is None


def test_bilans_pelny_lan_bez_opadu_zuzywa_kc_razy_et0():
    start = date(2026, 6, 1)
    days = wb.run_balance(_wx(start, 3), [(start, 0.85)], [], taw_mm=100, p=0.5, kcb_max=1.0)
    # pełny łan: Kcb=1.0, Ke=0 -> 5 mm/d
    assert [d["depletion_mm"] for d in days] == [5.0, 10.0, 15.0]


def test_ks_ogranicza_ewapotranspiracje_ponizej_raw():
    start = date(2026, 6, 1)
    days = wb.run_balance(_wx(start, 30), [(start, 0.85)], [], taw_mm=100, p=0.5, kcb_max=1.0)
    stressed = [d for d in days if d["stress"]]
    assert stressed and stressed[-1]["ks"] < 1.0
    assert days[-1]["depletion_mm"] < 100.0          # nigdy nie wyczerpie całkowicie przy Ks->0
    # dzienne zużycie maleje po przekroczeniu RAW
    assert stressed[-1]["etc_mm"] < 5.0


def test_nawodnienie_z_dawka_i_bez_dawki():
    start = date(2026, 6, 1)
    wx = _wx(start, 10)
    ev = [{"event_date": "2026-06-05", "amount_mm": 20.0}]
    days = wb.run_balance(wx, [(start, 0.85)], ev, taw_mm=200, p=0.5, kcb_max=1.0)
    by = {d["date"]: d for d in days}
    assert by[date(2026, 6, 5)]["depletion_mm"] == 5.0    # 20 + 5 - 20
    ev2 = [{"event_date": "2026-06-05", "amount_mm": None}]
    days2 = wb.run_balance(wx, [(start, 0.85)], ev2, taw_mm=200, p=0.5, kcb_max=1.0)
    assert {d["date"]: d for d in days2}[date(2026, 6, 5)]["depletion_mm"] == 0.0


def test_male_opady_nie_zasilaja_strefy_korzeniowej():
    start = date(2026, 6, 1)
    days = wb.run_balance(_wx(start, 2, precip=1.5, et0=2.0), [(start, 0.85)], [],
                          taw_mm=100, p=0.5, kcb_max=1.0)
    assert days[-1]["depletion_mm"] == 4.0


def test_podsumowanie_dni_do_stresu_i_dawka():
    today = date(2026, 6, 10)
    start = today - timedelta(days=4)
    wx = _wx(start, 15, forecast_from=today)
    days = wb.run_balance(wx, [(start, 0.85)], [], taw_mm=100, p=0.5, kcb_max=1.0)
    s = wb.summarize_balance(days, 100, 0.5, today)
    assert s["status"] == "ok"
    assert s["depletion_mm"] == 25.0               # 5 dni x 5 mm
    assert s["days_to_stress"] == 6                 # 55 mm > RAW 50 mm 6 dni później
    assert s["suggested_irrigation_date"] == "2026-06-16"
    assert s["suggested_dose_mm"] == 55
    assert s["suggested_dose_m3_ha"] == 550
    assert "Sugerowane nawodnienie: 2026-06-16" in wb.describe_balance(s)


def test_compute_field_balance_z_zamockowana_pogoda(isolated_db, monkeypatch):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZB-1", "irrigation", SAMPLE_WKT,
                                      crop_type="bawełna", soil_taw_mm=120.0)
    isolated_db.add_water_observation(fid, (date.today() - timedelta(days=5)).isoformat(),
                                      ndwi_mean=0.3, ndvi_mean=0.7)
    isolated_db.add_irrigation_event(fid, (date.today() - timedelta(days=3)).isoformat(),
                                     "zalewowe", amount_or_duration=800, amount_unit="m3/ha")
    import weather
    today = date.today()
    monkeypatch.setattr(weather, "fetch_recent_and_forecast",
                        lambda geom, past_days, forecast_days: _wx(
                            today - timedelta(days=20), 30, forecast_from=today))
    res = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid))
    s = res["summary"]
    assert s["taw_mm"] == 120.0 and s["p"] == 0.65 and s["taw_assumed"] is False
    assert any(d["irrigation_mm"] == 80.0 for d in res["days"])
    assert s["message_pl"].startswith("Bilans wodny")


def test_irrigation_amount_mm_przeliczenia(isolated_db):
    assert isolated_db.irrigation_amount_mm(30, "mm") == 30.0
    assert isolated_db.irrigation_amount_mm(600, "m³/ha") == 60.0
    assert isolated_db.irrigation_amount_mm(3, "h") is None
    assert isolated_db.irrigation_amount_mm(None, "mm") is None


def test_api_balance_endpoint(app_client, monkeypatch):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "UZB-BAL-1", "irrigation_mode": "irrigation",
        "geometry_wkt": SAMPLE_WKT, "crop_type": "cotton", "soil_taw_mm": 90})
    fid = r.json()["id"]
    import weather
    today = date.today()
    monkeypatch.setattr(weather, "fetch_recent_and_forecast",
                        lambda geom, past_days, forecast_days: _wx(
                            today - timedelta(days=10), 20, forecast_from=today))
    r = app_client.get(f"/api/water/field/{fid}/balance")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["taw_mm"] == 90.0
    assert len(body["days"]) == 20


def test_fetch_recent_and_forecast_parsuje(monkeypatch):
    from unittest.mock import MagicMock
    from shapely.geometry import box
    import weather
    today = date.today()
    payload = {"daily": {"time": [(today - timedelta(days=1)).isoformat(), today.isoformat(),
                                  (today + timedelta(days=1)).isoformat()],
                         "precipitation_sum": [0.0, 3.2, 1.0],
                         "et0_fao_evapotranspiration": [4.1, 3.0, 4.5]}}
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return MagicMock(status_code=200, json=lambda: payload)
    monkeypatch.setattr(weather.requests, "get", fake_get)
    out = weather.fetch_recent_and_forecast(box(66.9, 39.6, 66.91, 39.61), 5, 1)
    assert [o["is_forecast"] for o in out] == [False, False, True]
    assert out[1]["precip_mm"] == 3.2 and out[2]["et0_mm"] == 4.5
    assert captured["past_days"] == 5 and captured["timezone"] == "auto"


def test_attach_water_balance_laczy_zrodla(monkeypatch):
    import water_stress as ws
    monkeypatch.setattr(wb, "compute_field_balance", lambda field: {"summary": {"status": "ok"}})
    v = {"risk_flag": True, "risk_type": "water_stress_risk", "message_pl": "X."}
    ws.attach_water_balance({"id": 1}, v)
    assert "nie wskazuje deficytu" in v["message_pl"]
    monkeypatch.setattr(wb, "compute_field_balance", lambda field: {"summary": {"status": "stress"}})
    v = {"risk_flag": False, "risk_type": None, "message_pl": "Y."}
    ws.attach_water_balance({"id": 1}, v)
    assert "wskazuje już stres" in v["message_pl"]


def test_attach_water_balance_nieblokujace(monkeypatch):
    import water_stress as ws

    def boom(field):
        raise RuntimeError("brak sieci")
    monkeypatch.setattr(wb, "compute_field_balance", boom)
    v = {"risk_flag": False, "message_pl": "Z."}
    assert ws.attach_water_balance({"id": 1}, v) == {"risk_flag": False, "message_pl": "Z."}
