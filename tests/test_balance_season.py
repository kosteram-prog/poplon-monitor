"""Bilans od początku sezonu: stan początkowy z ERA5-Land, odcinek
szacunkowy i punkty kontrolne wilgotności (water_balance, weather)."""
from datetime import date, timedelta

import pytest

import water_balance as wb
import weather

WKT = "POLYGON((66.9 39.6, 66.901 39.6, 66.901 39.601, 66.9 39.601, 66.9 39.6))"
TODAY = date(2026, 7, 20)


def _wx(start, n, precip=0.0, et0=4.0, rain_on=None, sm=(0.25, 0.25, 0.25)):
    out = []
    for i in range(n):
        d = start + timedelta(days=i)
        out.append({"date": d, "precip_mm": (rain_on or {}).get(d, precip), "et0_mm": et0,
                    "is_forecast": False, "sm_0_7": sm[0], "sm_7_28": sm[1], "sm_28_100": sm[2]})
    return out


# --- czyste funkcje -------------------------------------------------------------

@pytest.mark.parametrize("today,exp", [
    (date(2026, 7, 20), date(2026, 3, 1)), (date(2026, 3, 15), date(2025, 3, 1)),
    (date(2026, 3, 31), date(2026, 3, 1)), (date(2027, 1, 10), date(2026, 3, 1))])
def test_season_start(today, exp):
    assert wb.season_start(today) == exp


def test_soil_limits_typ_gleby_z_zimy():
    lim = weather.soil_limits(0.35)
    assert lim["soil_key"] == "medium" and lim["fc"] == 0.347 and lim["wp"] == 0.151
    assert weather.soil_limits(0.25)["soil_key"] == "coarse"
    assert weather.soil_limits(0.40)["soil_key"] == "medium_fine"


def test_peak_theta_p95_i_minimum_dni():
    rows = [{"date": date(2025, 10, 1) + timedelta(days=i), "sm_0_7": v, "sm_7_28": v, "sm_28_100": v}
            for i, v in enumerate([0.25] * 180 + [0.38] * 20)]      # szczyt wiosną, nie zimą
    assert weather.peak_theta(rows) == pytest.approx(0.38)
    assert weather.peak_theta(rows[:60]) is None
    assert weather.peak_theta([{**r, "is_forecast": True} for r in rows]) is None
    assert weather.hydro_year_start(date(2026, 3, 1)) == date(2025, 10, 1)


def test_root_zone_theta_wazona():
    assert weather.root_zone_theta({"sm_0_7": 0.1, "sm_7_28": 0.2, "sm_28_100": 0.3}) == pytest.approx(
        (0.7 + 4.2 + 21.6) / 100)
    assert weather.root_zone_theta({"sm_0_7": 0.1, "sm_7_28": None, "sm_28_100": 0.3}) is None


def test_ndvi_przed_pierwsza_obserwacja_najwyzej_30_dni():
    pts = [(date(2026, 6, 1), 0.7)]
    assert wb.interpolate_ndvi(pts, date(2026, 5, 10), 30) == 0.7
    assert wb.interpolate_ndvi(pts, date(2026, 4, 1), 30) is None
    assert wb.interpolate_ndvi(pts, date(2026, 4, 1)) == 0.7          # zgodność wstecz


def test_run_balance_stan_poczatkowy_i_punkt_kontrolny():
    wx = _wx(date(2026, 3, 1), 10, et0=0.0)
    days = wb.run_balance(wx, [], [], 100.0, 0.5, 1.1, initial_depletion=40.0,
                          soil_checks=[{"check_date": "2026-03-05", "fill_pct": 25}])
    assert days[0]["depletion_mm"] == 40.0
    assert days[4]["depletion_mm"] == 75.0 and days[4]["soil_check_pct"] == 25
    assert days[5]["depletion_mm"] == 75.0 and days[5]["soil_check_pct"] is None


def test_anchor_po_pelnym_uzupelnieniu_profilu():
    s0 = date(2026, 3, 1)
    wx = _wx(s0, 40, rain_on={s0 + timedelta(days=20): 150.0})
    assert wb.anchor_date(wx, [], [], 100.0, 0.5, 1.1, s0, [], 30) == s0 + timedelta(days=20)
    assert wb.anchor_date(_wx(s0, 40), [], [], 100.0, 0.5, 1.1, s0, [], 30) is None
    chk = [{"check_date": "2026-03-10", "fill_pct": 60}]
    assert wb.anchor_date(_wx(s0, 40), [], [], 100.0, 0.5, 1.1, s0, chk, 30) == date(2026, 3, 10)


# --- pełny bilans pola ------------------------------------------------------------

WINTER_SM = (0.36, 0.35, 0.345)     # zimą profil na FC gleby „średniej” (0.347)
SPRING_SM = (0.30, 0.28, 0.26)      # 1 marca: θ = 0.267 -> (0.267−0.151)/(0.347−0.151) ≈ 59%


def _season_rows(a, b, et0=4.0, rain_on=None):
    rows = _wx(a, (b - a).days + 1, et0=et0, rain_on=rain_on)
    for r in rows:
        sm = WINTER_SM if r["date"] < date(2026, 3, 1) else SPRING_SM
        r["sm_0_7"], r["sm_7_28"], r["sm_28_100"] = sm
    return rows


@pytest.fixture
def season_wx(monkeypatch):
    """Archiwum od 1 marca (ostatnie 5 dni bez danych — opóźnienie ERA5) + prognoza."""
    s0 = wb.season_start(TODAY)
    calls = {}

    def arch(geom, a, b):
        calls["arch"] = (a, b)
        rows = _season_rows(a, b, rain_on={s0 + timedelta(days=40): 120.0})
        for r in rows[-5:]:
            r["precip_mm"] = r["et0_mm"] = None
        return rows

    def recent(geom, past_days, forecast_days):
        calls["recent"] = past_days
        start = TODAY - timedelta(days=past_days)
        return [{"date": start + timedelta(days=i), "precip_mm": 0.0, "et0_mm": 5.0,
                 "is_forecast": start + timedelta(days=i) > TODAY}
                for i in range(past_days + forecast_days + 1)]

    monkeypatch.setattr(weather, "fetch_weather_series", arch)
    monkeypatch.setattr(weather, "fetch_recent_and_forecast", recent)
    monkeypatch.setattr(weather, "fetch_climatology", lambda lat, lon: {})
    return {"s0": s0, "calls": calls}


def test_season_weather_skleja_archiwum_i_prognoze(season_wx):
    from shapely import wkt
    rows = weather.fetch_season_weather(wkt.loads(WKT), season_wx["s0"], 10, today=TODAY)
    dates = [r["date"] for r in rows]
    assert dates[0] == season_wx["s0"] and len(dates) == len(set(dates))
    assert dates == sorted(dates) and dates[-1] == TODAY + timedelta(days=10)
    assert all((dates[i + 1] - dates[i]).days == 1 for i in range(len(dates) - 1))
    assert season_wx["calls"]["recent"] == weather.SEASON_FC_PAST_DAYS


def test_bilans_od_poczatku_sezonu_z_era5(isolated_db, season_wx):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT, crop_type="bawełna", soil_taw_mm=100.0)
    res = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid), today=TODAY)
    s, days = res["summary"], res["days"]
    assert s["balance_mode"] == "season" and s["season_start"] == "2026-03-01"
    init = s["initial_state"]
    theta = (7 * 0.30 + 21 * 0.28 + 72 * 0.26) / 100
    assert init["source"] == "era5" and init["soil_key"] == "medium"
    assert init["fill_pct"] == round(100 * (theta - 0.151) / (0.347 - 0.151)) == 59
    assert season_wx["calls"]["arch"][0] == date(2025, 10, 1)       # archiwum od zimy
    assert days[0]["date"] == "2026-03-01" and days[0]["depletion_mm"] < 100 - init["fill_pct"] + 5
    # 120 mm deszczu 40. dnia uzupełnia profil — od tego dnia wynik nie zależy od startu
    assert s["anchored_from"] == "2026-04-10" and s["initial_state_reliable"] is True
    assert days[0]["estimated"] and not days[-1]["estimated"]
    assert not any("Stan początkowy" in a for a in s["assumptions"])


def _low_et(monkeypatch):
    """Mało parowania i zero deszczu: bilanse z pełnym i pustym profilem się nie spotkają."""
    monkeypatch.setattr(weather, "fetch_weather_series",
                        lambda g, a, b: _season_rows(a, b - timedelta(days=5), et0=0.3))
    monkeypatch.setattr(weather, "fetch_recent_and_forecast",
                        lambda g, past_days, forecast_days: _wx(TODAY - timedelta(days=past_days),
                                                                past_days + forecast_days + 1, et0=0.3))


def test_suchy_profil_tez_zakotwicza(isolated_db, season_wx, monkeypatch):
    """Długa susza: profil wysycha bez względu na start — od tego dnia wynik jest pewny."""
    monkeypatch.setattr(weather, "fetch_weather_series",
                        lambda g, a, b: _season_rows(a, b - timedelta(days=5), et0=4.0))
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    s = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid), today=TODAY)["summary"]
    assert s["anchored_from"] is not None and s["status"] == "stress"


def test_bilans_bez_zakotwiczenia_to_szacunek(isolated_db, season_wx, monkeypatch):
    _low_et(monkeypatch)
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    s = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid), today=TODAY)["summary"]
    assert s["anchored_from"] is None and s["initial_state_reliable"] is False
    assert "Szacunek" in s["message_pl"] and "punkt kontrolny" in s["message_pl"]


def test_punkt_kontrolny_zakotwicza_bilans(isolated_db, season_wx, monkeypatch):
    _low_et(monkeypatch)
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT, soil_taw_mm=100.0)
    isolated_db.add_soil_check(fid, "2026-07-01", 62.0, method="feel", level="medium")
    isolated_db.add_soil_check(fid, "2025-07-01", 10.0)                 # poprzedni sezon — pomijany
    res = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid), today=TODAY)
    s = res["summary"]
    day = next(d for d in res["days"] if d["date"] == "2026-07-01")
    assert day["depletion_mm"] == 38.0 and day["soil_check_pct"] == 62
    assert s["anchored_from"] == "2026-07-01" and s["last_soil_check"]["fill_pct"] == 62.0
    assert any("Punkty kontrolne" in a and ": 1 " in a for a in s["assumptions"])


def test_awaryjnie_92_dni_gdy_archiwum_padnie(isolated_db, season_wx, monkeypatch):
    def boom(*a):
        raise RuntimeError("archiwum 500")
    monkeypatch.setattr(weather, "fetch_weather_series", boom)
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    s = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid), today=TODAY)["summary"]
    assert s["balance_mode"] == "recent" and s["initial_state"]["source"] == "assumed_full"
    assert season_wx["calls"]["recent"] == wb.BALANCE_PAST_DAYS


def test_siew_w_sezonie_goła_gleba_przed(isolated_db, season_wx):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT, sowing_date="2026-05-01")
    isolated_db.add_water_observation(fid, "2026-04-20", ndvi_mean=0.8, ndwi_mean=0.2)   # chwast/poprzedni łan
    isolated_db.add_water_observation(fid, "2026-06-01", ndvi_mean=0.6, ndwi_mean=0.2)
    days = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid), today=TODAY)["days"]
    by = {d["date"]: d for d in days}
    assert by["2026-04-20"]["ndvi"] == wb.NDVI_MIN and by["2026-06-01"]["ndvi"] == 0.6


def test_ndvi_z_optram_uzupelnia_poczatek_sezonu(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    isolated_db.add_water_observation(fid, "2026-07-01", ndvi_mean=0.6, ndwi_mean=0.2)
    isolated_db.upsert_optram_series(fid, [{"obs_date": "2026-03-10", "w": 0.5, "ndvi": 0.3, "valid_frac": 1.0},
                                           {"obs_date": "2026-03-20", "w": 0.5, "ndvi": 0.9, "valid_frac": 0.5},
                                           {"obs_date": "2026-07-01", "w": 0.5, "ndvi": 0.1, "valid_frac": 1.0}])
    pts = wb.ndvi_points_for_field(fid)
    assert pts == [(date(2026, 3, 10), 0.3), (date(2026, 7, 1), 0.6)]


# --- API, panel rolnika, oś czasu ---------------------------------------------------

def test_api_punkt_kontrolny(app_client, isolated_db):
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    today = date.today().isoformat()
    r = app_client.post(f"/api/water/field/{fid}/soil_check", json={"check_date": today, "level": "dry"})
    assert r.status_code == 200 and r.json()["fill_pct"] == 37.0
    r = app_client.post(f"/api/water/field/{fid}/soil_check",
                        json={"check_date": today, "fill_pct": 71, "method": "sensor", "depth_cm": 30})
    assert r.json()["fill_pct"] == 71
    assert app_client.post(f"/api/water/field/{fid}/soil_check", json={"check_date": today}).status_code == 422
    future = (date.today() + timedelta(days=2)).isoformat()
    assert app_client.post(f"/api/water/field/{fid}/soil_check",
                           json={"check_date": future, "level": "wet"}).status_code == 422
    rows = app_client.get(f"/api/water/field/{fid}/soil_checks").json()
    assert [r["fill_pct"] for r in rows] == [37.0, 71.0] and rows[1]["method"] == "sensor"
    assert app_client.delete(f"/api/water/soil_check/{rows[0]['id']}").json() == {"status": "deleted"}
    assert app_client.delete(f"/api/water/soil_check/{rows[0]['id']}").status_code == 404


def test_panel_rolnika_ocena_gleby(app_client, isolated_db, monkeypatch):
    import farmer_panel as fp
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT, farmer_phone="+998 90 123 45 67")
    fp.sync_farmers_from_fields()
    tok = isolated_db.get_farmer_by_phone("998901234567")["token"]
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: {"status": "ok"})
    r = app_client.post(f"/api/f/{tok}/field/{fid}/soil_check",
                        json={"check_date": date.today().isoformat(), "level": "moist"})
    assert r.status_code == 200 and r.json()["fill_pct"] == 87.0
    c = isolated_db.get_soil_checks(fid)[-1]
    assert c["source"] == "farmer_panel" and c["level"] == "moist"
    assert app_client.post(f"/api/f/{tok}/field/{fid}/soil_check",
                           json={"check_date": date.today().isoformat(), "level": "soggy"}).status_code == 422
    f0 = app_client.get(f"/api/f/{tok}").json()["fields"][0]
    assert f0["soil_checks"][0]["level"] == "moist"


def test_os_czasu_odcinek_szacunkowy_i_punkty():
    import water_overview as wo
    bal = {"summary": {"taw_mm": 100.0, "raw_mm": 50.0, "anchored_from": "2026-07-02",
                       "season_start": "2026-03-01", "initial_state": {"source": "era5"}},
           "days": [{"date": "2026-07-01", "depletion_mm": 30.0, "is_forecast": False, "estimated": True},
                    {"date": "2026-07-02", "depletion_mm": 38.0, "is_forecast": False, "estimated": False,
                     "soil_check_pct": 62}]}
    tl = wo._soil_points(bal, date(2026, 6, 1))
    assert [p["est"] for p in tl["soil"]] == [True, False]
    assert tl["soil_checks"] == [{"date": "2026-07-02", "v": 62}]
    assert tl["anchored_from"] == "2026-07-02" and tl["initial_source"] == "era5"


def test_plan_uzupelnia_historie_od_poczatku_sezonu(isolated_db):
    import water_overview as wo
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    for k in range(8):
        isolated_db.add_water_observation(fid, (TODAY - timedelta(days=5 * k)).isoformat(),
                                          ndvi_mean=0.6, ndwi_mean=0.2)
    f = isolated_db.get_water_field_by_id(fid)
    assert wo.backfill_start(f, TODAY) == date(2026, 3, 1)
    step = wo.prepare_plan(f, today=TODAY)["steps"][0]
    assert step["run"] and step["reason"] == "season_gap" and step["est_pu"] == pytest.approx(3.5, abs=0.1)
    isolated_db.add_water_observation(fid, "2026-03-15", ndvi_mean=0.2, ndwi_mean=0.1)
    assert wo.backfill_start(isolated_db.get_water_field_by_id(fid), TODAY) is None


def test_swieza_ocena_gleby_przywraca_zaufanie_do_bilansu(isolated_db):
    import water_overview as wo
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ-S", "irrigation", WKT)
    f = isolated_db.get_water_field_by_id(fid)
    assert wo.balance_trust(f, TODAY)["reason"] == "no_irrigation_log"
    isolated_db.add_soil_check(fid, "2026-07-01", 40.0)               # za stara
    assert wo.balance_trust(f, TODAY)["trusted"] is False
    isolated_db.add_soil_check(fid, "2026-07-15", 40.0)
    assert wo.balance_trust(f, TODAY) == {"trusted": True, "reason": None}
