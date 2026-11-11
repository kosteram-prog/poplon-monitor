"""Testy rozszerzeń pogodowych: norma 1991–2020, prognoza, VPD/upał,
wilgotność gleby ERA5-Land, deszczomierz rolnika (bez sieci)."""
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

import weather
import water_balance as wb
import water_stress as ws

SAMPLE_WKT = "POLYGON((17.0 51.1, 17.004 51.1, 17.004 51.103, 17.0 51.103, 17.0 51.1))"


def _clim(daily_value_by_year):
    """Sztuczna klimatologia: każdy dzień roku Y ma wartość f(Y)."""
    out, d = {}, weather.CLIMATE_START
    while d <= weather.CLIMATE_END:
        out[d] = daily_value_by_year(d.year)
        d += timedelta(days=1)
    return out


def test_percentyl_bilansu_wzgledem_normy():
    clim = _clim(lambda y: (y - 2005) * 0.1)        # lata od -1.4 do +1.5 mm/d
    end = date(2026, 7, 15)
    # 30 dni po -1.5 mm/d = -45 mm: suchsze niż każdy rok normy
    assert weather.cwb_percentile(-45.0, end, 30, clim) == 0.0
    assert weather.cwb_percentile(60.0, end, 30, clim) == 100.0
    mid = weather.cwb_percentile(0.0, end, 30, clim)
    assert 40 < mid < 60
    assert weather.cwb_class(5) == "wyjątkowo sucho" and weather.cwb_class(50) == "w normie"
    assert weather.cwb_percentile(-10, end, 30, {}) is None


def test_percentyl_29_lutego():
    clim = _clim(lambda y: 0.0)
    assert weather.cwb_percentile(0.0, date(2028, 2, 29), 30, clim) == 50.0


def test_fetch_climatology_parsuje(monkeypatch):
    payload = {"daily": {"time": ["1991-01-01", "1991-01-02"], "precipitation_sum": [3.0, None],
                         "et0_fao_evapotranspiration": [0.5, 0.4]}}
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return MagicMock(status_code=200, json=lambda: payload)
    monkeypatch.setattr(weather.requests, "get", fake_get)
    clim = weather.fetch_climatology(52.234, 16.489)
    assert clim == {date(1991, 1, 1): 2.5}
    assert captured["latitude"] == 52.2 and captured["start_date"] == "1991-01-01"


def test_prognoza_sucho_i_goraco():
    today = date(2026, 7, 10)
    series = [{"date": today + timedelta(days=i), "precip_mm": 0.0, "et0_mm": 6.0,
               "tmax_c": 33 + i, "vpd_max_kpa": 2.5 + 0.2 * i} for i in range(-2, 9)]
    fc = weather.forecast_summary(series, today)
    assert fc["days"] == 7 and fc["precip_mm"] == 0 and fc["et0_mm"] == 42.0
    assert fc["heat_days"] == 6 and fc["dry_hot"] is True
    rainy = [dict(r, precip_mm=5.0) for r in series]
    assert weather.forecast_summary(rainy, today)["dry_hot"] is False


def test_komunikat_z_norma_i_upalem():
    field = {"parcel_id": "UZB-1", "irrigation_mode": "irrigation"}
    obs = [{"obs_date": (date.today() - timedelta(days=d)).isoformat(), "ndwi_mean": v}
           for d, v in zip(range(5, 0, -1), [0.30, 0.31, 0.29, 0.30, 0.31])]
    obs.append({"obs_date": date.today().isoformat(), "ndwi_mean": 0.12, "cwb_14d_mm": -60.0,
                "cwb_30d_pct": 4.0, "vpd_max_kpa": 4.2, "tmax_c": 39.0})
    r = ws.classify_water_signal(field, obs, [], [])
    assert r["risk_flag"] is True
    assert "wyjątkowo sucho" in r["message_pl"] and "4. percentyl" in r["message_pl"]
    assert "VPD do 4.2 kPa" in r["message_pl"] and "Tmax 39" in r["message_pl"]
    assert r["weather_context"]["cwb_30d_pct"] == 4.0


def test_deszczomierz_zastepuje_opad_modelu():
    wx = [{"date": date(2026, 7, 1), "precip_mm": 0.0, "et0_mm": 5.0},
          {"date": date(2026, 7, 2), "precip_mm": 12.0, "et0_mm": 5.0}]
    n = wb.apply_rain_gauge(wx, [{"reading_date": "2026-07-01", "precip_mm": 18.0},
                                 {"reading_date": "2026-07-01", "precip_mm": 4.0}])
    assert n == 1 and wx[0]["precip_mm"] == 22.0 and wx[0]["precip_source"] == "gauge"
    assert wx[1]["precip_mm"] == 12.0


def test_alarm_bilansu_sucho_i_goraco_dla_irygacji():
    f = {"parcel_id": "UZB-1", "irrigation_mode": "irrigation"}
    bal = {"status": "approaching", "days_to_stress": None, "depletion_mm": 45, "raw_mm": 55,
           "forecast": {"dry_hot": True, "tmax_c": 38.0}}
    d = ws.balance_alarm_decision(f, {"risk_flag": False, "water_balance": bal})
    assert d and "suchy i gorący" in d["message_pl"] and "38" in d["message_pl"]
    bal_ok = dict(bal, forecast={"dry_hot": False})
    assert ws.balance_alarm_decision(f, {"risk_flag": False, "water_balance": bal_ok}) is None


def test_bilans_z_deszczomierzem_prognoza_i_norma(isolated_db, monkeypatch):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZB-1", "irrigation", SAMPLE_WKT, crop_type="bawełna",
                                      soil_taw_mm=120.0)
    today = date.today()
    isolated_db.add_rain_reading(fid, (today - timedelta(days=2)).isoformat(), 30.0)
    wx = [{"date": today - timedelta(days=40) + timedelta(days=i), "precip_mm": 0.0, "et0_mm": 4.0,
           "tmax_c": 36.0, "vpd_max_kpa": 3.5, "is_forecast": (today - timedelta(days=40) + timedelta(days=i)) > today}
          for i in range(48)]
    monkeypatch.setattr(weather, "fetch_recent_and_forecast", lambda g, past_days, forecast_days: wx)
    monkeypatch.setattr(weather, "fetch_climatology", lambda lat, lon: _clim(lambda y: 0.0))
    monkeypatch.setattr(weather, "fetch_weather_series", lambda g, a, b: [
        {"date": today - timedelta(days=5), "sm_0_7": 0.21, "sm_7_28": 0.25, "sm_28_100": 0.30}])
    s = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid))["summary"]
    assert s["rain_gauge_days_used"] == 1
    assert s["forecast"]["dry_hot"] is True and s["forecast"]["heat_days"] >= 1
    assert s["cwb_30d_pct"] == 0.0 and s["cwb_30d_class"] == "wyjątkowo sucho"
    assert s["soil_moisture"]["sm_7_28"] == 0.25
    assert "Open-Meteo" in s["attribution"]
    assert "sucho i gorąco" in s["message_pl"] and "1991–2020" in s["message_pl"]


def test_api_deszczomierz(app_client):
    r = app_client.post("/api/water/field/add", json={
        "parcel_id": "UZB-RAIN", "irrigation_mode": "irrigation", "geometry_wkt": SAMPLE_WKT})
    fid = r.json()["id"]
    r = app_client.post(f"/api/water/field/{fid}/rain", json={"reading_date": "2026-07-01", "precip_mm": 14.5})
    assert r.status_code == 200
    rid = r.json()["id"]
    assert app_client.get(f"/api/water/field/{fid}/rain").json()[0]["precip_mm"] == 14.5
    future = (date.today() + timedelta(days=3)).isoformat()
    assert app_client.post(f"/api/water/field/{fid}/rain", json={"reading_date": future, "precip_mm": 1}).status_code == 422
    assert app_client.post(f"/api/water/field/{fid}/rain", json={"reading_date": "2026-07-02", "precip_mm": -1}).status_code == 422
    assert app_client.delete(f"/api/water/rain/{rid}").status_code == 200
    assert app_client.delete(f"/api/water/rain/{rid}").status_code == 404
