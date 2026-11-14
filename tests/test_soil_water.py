"""Pojemność wodna gleby (soil_water): PTF, SoilGrids, priorytety źródeł."""
import json

import pytest

import soil_water as sw

_REAL_FETCH = sw.fetch_soilgrids          # przed autouse-blokadą z conftest

WKT = "POLYGON((66.9 39.0, 66.902 39.0, 66.902 39.002, 66.9 39.002, 66.9 39.0))"
# profil jak w punkcie demo FAO (Szachrisabz): pył ilasty, ~11% szkieletu
PROFILE = [{"top": t, "bottom": b, "sand": 17.5, "clay": 32.5, "om": om, "coarse": 11.0}
           for t, b, om in ((0, 5, 4.4), (5, 15, 2.3), (15, 30, 1.7), (30, 60, 1.0), (60, 100, 0.7), (100, 200, 0.4))]


def test_saxton_rawls_kolejnosc_gatunkow():
    awc = {k: sw.awc_mm_per_m(s, c) for k, (s, c) in sw.TEXTURE_CLASSES.items()}
    assert 35 < awc["sand"] < 55 and 80 < awc["sandy_loam"] < 110 and 170 < awc["silt_loam"] < 210
    assert awc["sand"] < awc["loamy_sand"] < awc["sandy_loam"] < awc["loam"] < awc["silt_loam"]
    assert sw.awc_mm_per_m(20, 15, coarse_pct=50) == pytest.approx(sw.awc_mm_per_m(20, 15) / 2)


@pytest.mark.parametrize("key", list(sw.TEXTURE_CLASSES))
def test_srodek_klasy_wraca_do_klasy(key):
    s, c = sw.TEXTURE_CLASSES[key]
    assert sw.usda_texture(s, c) == key


def test_profile_awc_strefa_korzeni():
    p12 = sw.profile_awc(PROFILE, 1.2)
    assert p12["depth_cm"] == 120 and 130 <= p12["awc_mm_m"] <= 160
    assert p12["taw_mm"] == pytest.approx(p12["awc_mm_m"] * 1.2, abs=2)
    deep = sw.profile_awc(PROFILE[:5], 1.5)            # brak warstwy 100–200: przedłużamy ostatnią
    assert deep["depth_cm"] == 150
    assert sw.profile_awc([], 1.0) is None


def test_fetch_soilgrids_parsowanie(monkeypatch):
    import requests

    class R:
        status_code = 200

        def json(self):
            mk = lambda name, vals: {"name": name, "depths": [
                {"label": d[0], "values": {"mean": v}} for d, v in zip(sw.SOILGRIDS_DEPTHS, vals)]}
            return {"properties": {"layers": [mk("sand", [175] * 6), mk("clay", [325] * 6),
                                              mk("soc", [253, 130, 98, None, 40, 25]), mk("cfvo", [106] * 5 + [None])]}}
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    prof = _REAL_FETCH(39.05, 66.9)
    assert len(prof) == 6 and prof[0]["sand"] == 17.5 and prof[0]["clay"] == 32.5
    assert prof[0]["om"] == pytest.approx(253 / 100 * 1.724, abs=0.01) and prof[0]["coarse"] == 10.6
    assert prof[3]["om"] == sw.DEFAULT_OM_PCT and prof[5]["coarse"] == 0.0


@pytest.fixture
def field(isolated_db, monkeypatch):
    isolated_db.init_db()
    calls = []
    monkeypatch.setattr(sw, "fetch_soilgrids", lambda lat, lon, **k: calls.append((lat, lon)) or PROFILE)
    fid = isolated_db.add_water_field("UZ", "irrigation", WKT, crop_type="bawełna")
    return isolated_db.get_water_field_by_id(fid), calls


def test_zrodla_priorytet(field, isolated_db):
    f, calls = field
    t = sw.field_taw(f)
    assert t["source"] == "soilgrids" and t["zr_m"] == 1.2 and t["texture"] == "silty_clay_loam"
    assert 155 <= t["taw_mm"] <= 190
    sw.field_taw(f)
    assert len(calls) == 1                               # profil w cache bazy
    isolated_db.save_field_soil(f["id"], texture_class="loamy_sand")
    t = sw.field_taw(f)
    assert t["source"] == "texture_field" and t["taw_mm"] == pytest.approx(60 * 1.2, abs=6)
    isolated_db.save_field_soil(f["id"], root_limit_cm=50)
    t = sw.field_taw(f)
    assert t["zr_m"] == 0.5 and t["zr_limited"] and t["taw_mm"] == pytest.approx(30, abs=3)
    isolated_db.update_water_field(f["id"], soil_taw_mm=140.0)
    t = sw.field_taw(isolated_db.get_water_field_by_id(f["id"]))
    assert t["source"] == "manual" and t["taw_mm"] == 140.0


def test_blad_soilgrids_nie_pyta_co_chwile(isolated_db, monkeypatch):
    isolated_db.init_db()
    n = []

    def boom(lat, lon):
        n.append(1)
        raise RuntimeError("timeout")
    monkeypatch.setattr(sw, "fetch_soilgrids", boom)
    fid = isolated_db.add_water_field("PL", "rainfed", WKT)
    f = isolated_db.get_water_field_by_id(fid)
    assert sw.field_taw(f)["source"] == "default" and sw.field_taw(f)["taw_mm"] == 100.0
    assert len(n) == 1 and "timeout" in isolated_db.get_field_soil(fid)["sg_error"]


def test_api_gleba(app_client, isolated_db, monkeypatch):
    monkeypatch.setattr(sw, "fetch_soilgrids", lambda lat, lon, **k: PROFILE)
    fid = isolated_db.add_water_field("UZ", "irrigation", WKT, crop_type="cotton")
    d = app_client.get(f"/api/water/field/{fid}/soil").json()
    assert d["source"] == "soilgrids" and len(d["classes"]) == 12 and d["attribution"]
    r = app_client.put(f"/api/water/field/{fid}/soil", json={"texture_class": "silt_loam", "root_limit_cm": 80})
    assert r.status_code == 200 and r.json()["source"] == "texture_field" and r.json()["zr_m"] == 0.8
    assert app_client.put(f"/api/water/field/{fid}/soil", json={"texture_class": "mud"}).status_code == 422
    assert app_client.put(f"/api/water/field/{fid}/soil", json={"texture_class": ""}).json()["source"] == "soilgrids"
    assert app_client.post(f"/api/water/field/{fid}/soil/refresh").json()["sg_fetched_at"]


def test_bilans_uzywa_pojemnosci_z_gleby(isolated_db, monkeypatch):
    import water_balance as wb
    import weather
    from datetime import date, timedelta
    monkeypatch.setattr(sw, "fetch_soilgrids", lambda lat, lon, **k: PROFILE)
    t0 = date.today()
    monkeypatch.setattr(weather, "fetch_recent_and_forecast", lambda g, past_days, forecast_days: [
        {"date": t0 - timedelta(days=past_days - i), "precip_mm": 0.0, "et0_mm": 4.0,
         "is_forecast": i > past_days} for i in range(past_days + forecast_days + 1)])
    monkeypatch.setattr(weather, "fetch_climatology", lambda lat, lon: {})
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ", "irrigation", WKT, crop_type="bawełna")
    s = wb.compute_field_balance(isolated_db.get_water_field_by_id(fid))["summary"]
    assert s["soil"]["source"] == "soilgrids" and s["taw_mm"] == s["soil"]["taw_mm"] and not s["taw_assumed"]
    assert not any("Retencja użyteczna" in a for a in s["assumptions"])
    assert s["salinity"]["ks_salt"] == 1.0 and s["salinity"]["ece"] is None


# --- zasolenie i wody gruntowe ------------------------------------------------------

def test_ece_z_pomiaru_i_klasy():
    assert sw.salinity_ece({"salinity_class": "moderate"}, "loam") == (6.0, "class")
    assert sw.salinity_ece({"ec_value": 1.0, "ec_method": "ec_1_5", "salinity_class": "none"}, "silty_clay_loam") == (7.5, "ec_1_5")
    assert sw.salinity_ece({"ec_value": 2.0, "ec_method": "ec_water"}, None) == (3.0, "ec_water")
    assert sw.salinity_ece({"ec_value": 9.0, "ec_method": "ece"}, None) == (9.0, "ec_paste")
    assert sw.salinity_ece({}, "loam") == (None, "none")


def test_fao56_redukcja_bawelna_i_kukurydza():
    cot = sw.field_salinity({"id": 1, "crop_type": "bawełna"}, {"ec_value": 10.0, "ec_method": "ece"})
    # FAO-56 r. 89: 1 − 5,2/(0,85·100)·(10 − 7,7) = 0,859; spadek plonu 5,2·2,3 = 12%
    assert cot["ks_salt"] == pytest.approx(0.859, abs=0.001) and cot["yield_loss_pct"] == 12
    maize = sw.field_salinity({"id": 1, "crop_type": "kukurydza"}, {"salinity_class": "slight"})
    assert maize["ks_salt"] == pytest.approx(1 - 12 / 125 * 1.3, abs=0.001)
    ok = sw.field_salinity({"id": 1, "crop_type": "cotton"}, {"salinity_class": "moderate"})
    assert ok["ks_salt"] == 1.0 and ok["yield_loss_pct"] == 0
    unk = sw.field_salinity({"id": 1}, {"salinity_class": "very_strong"})
    assert not unk["crop_known"] and unk["threshold"] == 4.0 and unk["ks_salt"] == 0.0


def test_wskazowki_rolnik_i_wody_gruntowe():
    h = sw.field_salinity({"id": 1}, {"farmer_salt_crust": "much", "farmer_leaching": "yes", "gw_depth_m": 1.2})["hints"]
    assert h == ["farmer_crust", "farmer_leaching", "gw_shallow"]
    h = sw.field_salinity({"id": 1}, {"farmer_salt_crust": "much", "ec_value": 3, "ec_method": "ece", "gw_depth_m": 3})["hints"]
    assert h == []


def test_bilans_ks_salt_zmniejsza_zuzycie():
    import water_balance as wb
    from datetime import date, timedelta
    wx = [{"date": date(2026, 7, 1) + timedelta(days=i), "precip_mm": 0.0, "et0_mm": 6.0, "is_forecast": False}
          for i in range(10)]
    pts = [(date(2026, 7, 1), 0.8)]
    a = wb.run_balance(wx, pts, [], 150.0, 0.65, 1.15)
    b = wb.run_balance(wx, pts, [], 150.0, 0.65, 1.15, ks_salt=0.8)
    assert b[-1]["depletion_mm"] < a[-1]["depletion_mm"] and b[0]["ks"] == 0.8


def test_api_zasolenie_i_rolnik(app_client, isolated_db, monkeypatch):
    import farmer_panel as fp
    monkeypatch.setattr(sw, "fetch_soilgrids", lambda lat, lon, **k: PROFILE)
    fid = isolated_db.add_water_field("UZ", "irrigation", WKT, crop_type="bawełna", farmer_phone="+998 90 555 44 33")
    r = app_client.put(f"/api/water/field/{fid}/soil", json={"ec_value": 1.4, "ec_method": "ec_1_5", "gw_depth_m": 1.5})
    sa = r.json()["salinity"]
    assert sa["ece_source"] == "ec_1_5" and sa["ece"] == pytest.approx(10.5, abs=0.1) and sa["ks_salt"] < 1
    assert sa["gw_shallow"] and "gw_shallow" in sa["hints"]
    assert app_client.put(f"/api/water/field/{fid}/soil", json={"salinity_class": "salty"}).status_code == 422
    assert app_client.put(f"/api/water/field/{fid}/soil", json={"ec_method": "ec_9"}).status_code == 422
    fp.sync_farmers_from_fields()
    tok = isolated_db.get_farmer_by_phone("998905554433")["token"]
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: {"status": "ok"})
    r = app_client.post(f"/api/f/{tok}/field/{fid}/salt", json={"salt_crust": "spots", "leaching": "yes", "leaching_m3_ha": 3000})
    assert r.status_code == 200
    row = isolated_db.get_field_soil(fid)
    assert row["farmer_salt_crust"] == "spots" and row["farmer_leaching_m3"] == 3000 and row["farmer_report_at"]
    assert app_client.post(f"/api/f/{tok}/field/{fid}/salt", json={"salt_crust": "lots"}).status_code == 422
    assert app_client.get(f"/api/f/{tok}").json()["fields"][0]["salt"]["farmer_leaching"] == "yes"


def test_blad_rownoleglego_pobrania_nie_nadpisuje_profilu(isolated_db, monkeypatch):
    """Dwa równoległe pobrania SoilGrids: jedno zapisało profil, drugie padło
    po czasie — pole ma używać profilu i nie wisieć z komunikatem o błędzie."""
    import soil_water
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ", "irrigation",
                                      "POLYGON((66.909 39.046, 66.910 39.046, 66.910 39.047, 66.909 39.047, 66.909 39.046))",
                                      crop_type="cotton")
    prof = [{"top": 0, "bottom": 200, "sand": 18, "clay": 33, "soc": 7, "cfvo": 12}]

    def slow_fail(lat, lon):
        # w trakcie tego zapytania inne zapisało już profil
        isolated_db.save_field_soil(fid, sg_profile_json=json.dumps(prof), sg_fetched_at="2026-11-14T10:11:52")
        raise RuntimeError("Read timed out")
    monkeypatch.setattr(soil_water, "fetch_soilgrids", slow_fail)
    assert soil_water._soilgrids_profile(isolated_db.get_water_field_by_id(fid), True) == prof
    assert not (isolated_db.get_field_soil(fid) or {}).get("sg_error")


def test_tryb_szybki_jedna_krotka_proba(isolated_db, monkeypatch):
    """Panel nie może czekać ~90 s na SoilGrids: jedna próba, 8 s."""
    import soil_water
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(timeout)
        raise soil_water.requests.RequestException("timeout")
    monkeypatch.setattr(soil_water.requests, "get", fake_get)
    monkeypatch.setattr(soil_water.time, "sleep", lambda s: calls.append(("sleep", s)))
    monkeypatch.setattr(soil_water, "fetch_soilgrids", _REAL_FETCH)
    isolated_db.init_db()
    fid = isolated_db.add_water_field("UZ", "irrigation",
                                      "POLYGON((66.909 39.046, 66.910 39.046, 66.910 39.047, 66.909 39.047, 66.909 39.046))")
    t = soil_water.field_taw(isolated_db.get_water_field_by_id(fid), quick=True)
    assert t["source"] == "default" and calls == [soil_water.QUICK_TIMEOUT_S]
