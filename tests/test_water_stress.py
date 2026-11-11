"""Testy silnika anomalii stresu wodnego (water_stress.py) — Krok 2 planu.

Warstwa czystej logiki (z-score, klasyfikacja) bez sieci, plus jeden test
orkiestracji (fetch_and_store_observation) z zamockowanymi
analytics.fetch_multi_series / sentinel1.fetch_s1_series /
weather.fetch_weather_series (wzorem app_client w conftest.py — testy nie
łączą się z prawdziwym internetem)."""
from datetime import date, timedelta

import pytest

import water_stress as ws


SAMPLE_WKT = "POLYGON((17.0 51.1, 17.001 51.1, 17.001 51.101, 17.0 51.101, 17.0 51.1))"


def _mk_obs(days_ago, ndwi_mean, vv_db=None, dry_spell=False):
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return {"obs_date": d, "ndwi_mean": ndwi_mean, "vv_db": vv_db, "dry_spell": dry_spell}


# --- _zscore / own_history_anomaly / cohort_anomaly ----------------------

def test_zscore_none_gdy_za_malo_probek():
    assert ws._zscore(1.0, [1.0, 1.0]) is None  # 2 < OWN_HISTORY_MIN_SAMPLES


def test_zscore_none_gdy_brak_wariancji():
    assert ws._zscore(1.0, [0.5, 0.5, 0.5]) is None


def test_zscore_liczy_odchylenie():
    z = ws._zscore(1.0, [4.0, 5.0, 6.0])  # mean=5, stdev=1 -> (1-5)/1=-4
    assert z == -4.0


def test_own_history_anomaly_wykrywa_spadek():
    obs = [_mk_obs(10, 0.30), _mk_obs(9, 0.31), _mk_obs(8, 0.29),
           _mk_obs(7, 0.30), _mk_obs(0, 0.10)]  # ostatnia wyraźnie niższa
    result = ws.own_history_anomaly(obs, "ndwi_mean")
    assert result is not None
    assert result["z_score"] <= ws.Z_SCORE_RISK_THRESHOLD


def test_own_history_anomaly_brak_gdy_za_malo_historii():
    obs = [_mk_obs(1, 0.30), _mk_obs(0, 0.10)]
    assert ws.own_history_anomaly(obs, "ndwi_mean") is None


def test_cohort_anomaly_brak_gdy_za_malo_pol():
    assert ws.cohort_anomaly(0.10, [0.30]) is None  # 1 < COHORT_MIN_FIELDS


def test_cohort_anomaly_liczy_odchylenie():
    result = ws.cohort_anomaly(0.10, [0.30, 0.32, 0.29])
    assert result is not None
    assert result["z_score"] < 0


# --- get_cohort_fields ----------------------------------------------------

def test_get_cohort_fields_filtruje_po_region_i_trybie(isolated_db):
    isolated_db.init_db()
    f1 = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT, region="Dolny Slask")
    f2 = isolated_db.add_water_field("B", "rainfed", SAMPLE_WKT, region="Dolny Slask")
    isolated_db.add_water_field("C", "irrigation", SAMPLE_WKT, region="Dolny Slask")  # inny tryb
    isolated_db.add_water_field("D", "rainfed", SAMPLE_WKT, region="Samarkand")  # inny region

    field1 = isolated_db.get_water_field_by_id(f1)
    cohort = ws.get_cohort_fields(field1)
    assert [f["id"] for f in cohort] == [f2]


def test_get_cohort_fields_pusta_gdy_brak_region(isolated_db):
    isolated_db.init_db()
    f1 = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    field1 = isolated_db.get_water_field_by_id(f1)
    assert ws.get_cohort_fields(field1) == []


# --- last_irrigation_context ----------------------------------------------

def test_last_irrigation_context_znajduje_najblizsze_wczesniejsze_zdarzenie():
    events = [
        {"event_date": "2026-06-01", "method": "zalewowe"},
        {"event_date": "2026-06-10", "method": "kroplowe"},
        {"event_date": "2026-06-20", "method": "deszczowanie"},  # po dacie obserwacji
    ]
    ctx = ws.last_irrigation_context(events, "2026-06-12")
    assert ctx == {"days_since_last_event": 2, "method": "kroplowe"}


def test_last_irrigation_context_brak_zdarzen():
    assert ws.last_irrigation_context([], "2026-06-12") == {
        "days_since_last_event": None, "method": None}


# --- classify_water_signal --------------------------------------------------

def test_classify_brak_obserwacji():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    result = ws.classify_water_signal(field, [], [], [])
    assert result["risk_flag"] is False
    assert "niekalibrowane" in result["calibration_status"]


def test_classify_brak_anomalii_gdy_stabilna_historia():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    obs = [_mk_obs(i, 0.30) for i in range(5, 0, -1)] + [_mk_obs(0, 0.29)]
    result = ws.classify_water_signal(field, obs, [], [])
    assert result["risk_flag"] is False
    assert result["risk_type"] is None


def test_classify_flaguje_deficyt_w_trybie_deszczowym():
    field = {"parcel_id": "141201_1.0001.6509", "irrigation_mode": "rainfed"}
    obs = [_mk_obs(d, v) for d, v in zip(
        range(5, 0, -1), [0.30, 0.31, 0.29, 0.30, 0.31])] + [_mk_obs(0, 0.05)]
    result = ws.classify_water_signal(field, obs, [], [])
    assert result["risk_flag"] is True
    assert result["risk_type"] == "water_stress_risk"
    assert "mulczowanie" in result["message_pl"]
    assert "niekalibrowany" in result["message_pl"]


def test_classify_irrigation_mode_wiadomosc_i_recent_event_daje_performance_anomaly():
    field = {"parcel_id": "UZB-1", "irrigation_mode": "irrigation"}
    obs_dates_vals = list(zip(range(5, 0, -1), [0.30, 0.31, 0.29, 0.30, 0.31])) + [(0, 0.05)]
    obs = [_mk_obs(d, v) for d, v in obs_dates_vals]
    today = date.today().isoformat()
    recent_event = [{"event_date": (date.today() - timedelta(days=2)).isoformat(),
                     "method": "kroplowe"}]
    result = ws.classify_water_signal(field, obs, [], recent_event)
    assert result["risk_flag"] is True
    assert result["risk_type"] == "irrigation_performance_anomaly"
    assert "stres wodny" in result["message_pl"]


def test_classify_kohorta_rozroznia_problem_punktowy_od_regionalnego():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    obs = [_mk_obs(d, v) for d, v in zip(
        range(5, 0, -1), [0.30, 0.31, 0.29, 0.30, 0.31])] + [_mk_obs(0, 0.05)]

    # kohorta rowniez nisko, ZBLIZONA DO WARTOSCI POLA (nie tylko "tez niska")
    # -> efekt regionalny. Kohorta musi realnie odpowiadac wartosci pola, bo
    # przy n=3 male roznice w kohorcie daja niestabilny z-score.
    result_regional = ws.classify_water_signal(field, obs, [0.04, 0.05, 0.06], [])
    assert "regionalny" in result_regional["message_pl"]

    # kohorta normalna -> problem punktowy
    result_specific = ws.classify_water_signal(field, obs, [0.30, 0.31, 0.29], [])
    assert "specyficzna dla tego pola" in result_specific["message_pl"]


# --- fetch_and_store_observation (orkiestracja, zamockowana siec) --------

def test_fetch_and_store_observation_zapisuje_i_zwraca_obserwacje(isolated_db, monkeypatch):
    isolated_db.init_db()
    field_id = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT, region="Dolny Slask")
    field = isolated_db.get_water_field_by_id(field_id)

    import core
    monkeypatch.setattr(core, "cdse_token", lambda: "faketoken")

    import analytics
    d = date.today()
    monkeypatch.setattr(analytics, "fetch_multi_series_strict",
                        lambda geom, t_start, t_end, token=None: [
                            {"date": d, "ndvi": 0.5, "ndwi": 0.12, "valid_frac": 0.9}])

    import sentinel1
    monkeypatch.setattr(sentinel1, "fetch_s1_series",
                        lambda geom, t_start, t_end: [
                            {"date": d, "vv_db": -11.2, "vh_db": -18.0, "rvi": 0.4}])
    monkeypatch.setattr(sentinel1, "fetch_s1_relative_orbits",
                        lambda geom, t_start, t_end: {d: 22})

    import weather as weather_mod
    monkeypatch.setattr(weather_mod, "fetch_weather_series",
                        lambda geom, t_start, t_end: [
                            {"date": d - timedelta(days=i), "precip_mm": 0.0, "et0_mm": 3.0}
                            for i in range(31)])
    monkeypatch.setattr(weather_mod, "detect_dry_spell_dates", lambda series: {d})

    obs = ws.fetch_and_store_observation(field)
    assert obs is not None
    assert obs["ndwi_mean"] == 0.12
    assert obs["vv_db"] == -11.2
    assert obs["dry_spell"] is True
    assert obs["cwb_14d_mm"] == -42.0
    assert obs["s1_rel_orbit"] == 22
    assert obs["cwb_30d_mm"] == -90.0
    assert isolated_db.get_water_observations(field_id)[0]["cwb_14d_mm"] == -42.0

    stored = isolated_db.get_water_observations(field_id)
    assert len(stored) == 1
    assert stored[0]["ndwi_mean"] == 0.12

    refreshed = isolated_db.get_water_field_by_id(field_id)
    assert refreshed["last_checked_at"] is not None


def test_fetch_and_store_observation_none_gdy_brak_danych_optycznych(isolated_db, monkeypatch):
    isolated_db.init_db()
    field_id = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    field = isolated_db.get_water_field_by_id(field_id)

    import core
    monkeypatch.setattr(core, "cdse_token", lambda: "faketoken")
    import analytics
    monkeypatch.setattr(analytics, "fetch_multi_series_strict",
                        lambda geom, t_start, t_end, token=None: [])

    assert ws.fetch_and_store_observation(field) is None


# --- select_water_scene (tylko czyste sceny) -------------------------------

def test_select_water_scene_odrzuca_zachmurzone_i_bierze_najnowsza_czysta():
    d = date.today()
    optical = [
        {"date": d - timedelta(days=6), "ndwi": 0.30, "valid_frac": 0.95},
        {"date": d - timedelta(days=3), "ndwi": 0.28, "valid_frac": 0.80},
        {"date": d, "ndwi": 0.05, "valid_frac": 0.35},  # w większości chmura
    ]
    scene = ws.select_water_scene(optical)
    assert scene["date"] == d - timedelta(days=3)


def test_select_water_scene_none_gdy_brak_czystej_sceny():
    d = date.today()
    assert ws.select_water_scene([{"date": d, "ndwi": 0.2, "valid_frac": 0.4}]) is None


def test_fetch_and_store_observation_nie_uzywa_fallbacku_bez_maski(isolated_db, monkeypatch):
    """Moduł wodny nie może sięgać po sceny bez maski chmur."""
    isolated_db.init_db()
    field_id = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    field = isolated_db.get_water_field_by_id(field_id)
    import core
    monkeypatch.setattr(core, "cdse_token", lambda: "faketoken")
    import analytics
    monkeypatch.setattr(analytics, "fetch_multi_series_strict",
                        lambda geom, t_start, t_end, token=None: [])

    def _boom(*a, **k):
        raise AssertionError("fallback bez maski SCL nie powinien być wołany")
    monkeypatch.setattr(analytics, "fetch_multi_series", _boom)
    assert ws.fetch_and_store_observation(field) is None


# --- kohorta: uprawa + ta sama data --------------------------------------

def test_get_cohort_fields_rozroznia_uprawe(isolated_db):
    isolated_db.init_db()
    f1 = isolated_db.add_water_field("A", "irrigation", SAMPLE_WKT, region="Samarkand", crop_type="Bawełna")
    f2 = isolated_db.add_water_field("B", "irrigation", SAMPLE_WKT, region="Samarkand", crop_type="bawełna ")
    isolated_db.add_water_field("C", "irrigation", SAMPLE_WKT, region="Samarkand", crop_type="pszenica")
    isolated_db.add_water_field("D", "irrigation", SAMPLE_WKT, region="Samarkand")
    cohort = ws.get_cohort_fields(isolated_db.get_water_field_by_id(f1))
    assert [f["id"] for f in cohort] == [f2]


def test_cohort_values_near_date_pomija_nieaktualne_obserwacje(isolated_db):
    isolated_db.init_db()
    f2 = isolated_db.add_water_field("B", "rainfed", SAMPLE_WKT, region="R")
    f3 = isolated_db.add_water_field("C", "rainfed", SAMPLE_WKT, region="R")
    isolated_db.add_water_observation(f2, "2026-06-09", ndwi_mean=0.25)
    isolated_db.add_water_observation(f2, "2026-06-11", ndwi_mean=0.21)  # najbliższa
    isolated_db.add_water_observation(f3, "2026-06-01", ndwi_mean=0.40)  # 9 dni -> pomijamy
    cohort = [isolated_db.get_water_field_by_id(f2), isolated_db.get_water_field_by_id(f3)]
    assert ws.cohort_values_near_date(cohort, "2026-06-10") in ([0.25], [0.21])
    assert ws.cohort_values_near_date(cohort, "2026-06-12") == [0.21]


def test_update_water_field_whitelist(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("A", "rainfed", SAMPLE_WKT)
    isolated_db.update_water_field(fid, crop_type="kukurydza")
    assert isolated_db.get_water_field_by_id(fid)["crop_type"] == "kukurydza"
    with pytest.raises(ValueError):
        isolated_db.update_water_field(fid, irrigation_mode="irrigation")


# --- kontekst bilansu P-ET0 w werdykcie ------------------------------------

def _declining_obs(cwb):
    obs = [_mk_obs(d, v) for d, v in zip(range(5, 0, -1), [0.30, 0.31, 0.29, 0.30, 0.31])]
    last = _mk_obs(0, 0.05)
    last["cwb_14d_mm"] = cwb
    return obs + [last]


def test_classify_opisuje_deficyt_atmosferyczny():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    r = ws.classify_water_signal(field, _declining_obs(-45.0), [], [])
    assert r["risk_flag"] is True
    assert r["weather_context"]["atmospheric_deficit"] is True
    assert "zgodny z pogodą" in r["message_pl"]


def test_classify_przy_mokrym_bilansie_wskazuje_inne_przyczyny():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    r = ws.classify_water_signal(field, _declining_obs(25.0), [], [])
    assert r["weather_context"]["atmospheric_deficit"] is False
    assert "inne" in r["message_pl"] and "przyczyny" in r["message_pl"]


# --- odporna anomalia + NDMI rezydualny ------------------------------------

def _obs_pair(days_ago, ndmi, ndvi, cwb=None):
    o = _mk_obs(days_ago, ndmi)
    o["ndvi_mean"] = ndvi
    o["cwb_14d_mm"] = cwb
    return o


def test_plaska_historia_mala_zmiana_nie_daje_alarmu():
    """Dawniej idealnie płaska historia + zmiana o 0.01 -> ogromny z-score."""
    obs = [_mk_obs(d, 0.30) for d in range(6, 0, -1)] + [_mk_obs(0, 0.29)]
    obs[2]["ndwi_mean"] = 0.301  # minimalna wariancja
    r = ws.own_history_anomaly(obs, "ndwi_mean")
    assert r["significant_drop"] is False


def test_spadek_ndmi_przy_stabilnym_ndvi_to_wczesny_stres():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    hist = [_obs_pair(d, m, v) for d, m, v in zip(
        range(6, 0, -1), [0.30, 0.31, 0.29, 0.30, 0.31, 0.30], [0.80, 0.81, 0.79, 0.80, 0.80, 0.81])]
    obs = hist + [_obs_pair(0, 0.20, 0.79)]
    r = ws.classify_water_signal(field, obs, [], [])
    assert r["risk_flag"] is True
    assert r["components"]["own_ndwi_residual"]["significant_drop"] is True
    assert r["phenology_explained"] is False


def test_proporcjonalny_spadek_ndmi_i_ndvi_to_fenologia_bez_alarmu():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    # historia z naturalną zmiennością NDVI, z której da się dopasować nachylenie ~0.8
    hist = [_obs_pair(d, m, v) for d, m, v in zip(
        range(6, 0, -1), [0.30, 0.34, 0.26, 0.31, 0.29, 0.30], [0.75, 0.80, 0.70, 0.76, 0.74, 0.75])]
    obs = hist + [_obs_pair(0, 0.14, 0.55)]  # dojrzewanie: NDVI -0.20, NDMI -0.16
    r = ws.classify_water_signal(field, obs, [], [])
    assert r["components"]["own_ndwi"]["significant_drop"] is True
    assert r["risk_flag"] is False
    assert r["phenology_explained"] is True
    assert "dojrzewanie" in r["message_pl"]


def test_proporcjonalny_spadek_przy_deficycie_pogodowym_jednak_flagowany():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    hist = [_obs_pair(d, m, v) for d, m, v in zip(
        range(6, 0, -1), [0.30, 0.34, 0.26, 0.31, 0.29, 0.30], [0.75, 0.80, 0.70, 0.76, 0.74, 0.75])]
    obs = hist + [_obs_pair(0, 0.14, 0.55, cwb=-50.0)]  # więdnięcie w suszy
    r = ws.classify_water_signal(field, obs, [], [])
    assert r["risk_flag"] is True
    assert r["risk_type"] == "water_stress_risk"


def test_nachylenie_ndmi_ndvi_przyciete_do_zakresu():
    pairs = [(0.5, 0.1), (0.6, 0.4), (0.7, 0.7)]  # nachylenie 3.0
    assert ws._ndmi_ndvi_slope(pairs) == ws.RESIDUAL_SLOPE_BOUNDS[1]
    flat = [(0.80, 0.3), (0.81, 0.32), (0.80, 0.28)]
    assert ws._ndmi_ndvi_slope(flat) == ws.RESIDUAL_SLOPE_DEFAULT


# --- VV: ta sama orbita względna + gęsty łan ---------------------------------

def _vv_obs(days_ago, vv, orbit, ndvi=0.3):
    o = _mk_obs(days_ago, 0.30, vv_db=vv)
    o["s1_rel_orbit"] = orbit
    o["ndvi_mean"] = ndvi
    return o


def test_vv_porownywane_tylko_w_obrebie_tej_samej_orbity():
    # orbita 22 ma systematycznie ~2 dB niżej niż orbita 124 (inny kąt padania)
    obs = [_vv_obs(12, -10.0, 124), _vv_obs(10, -12.0, 22), _vv_obs(9, -10.1, 124),
           _vv_obs(7, -12.1, 22), _vv_obs(6, -9.9, 124), _vv_obs(4, -11.9, 22),
           _vv_obs(0, -12.0, 22)]
    r = ws.own_vv_anomaly(obs)
    assert r["rel_orbit"] == 22
    assert r["n_samples"] == 3
    assert r["significant_drop"] is False


def test_vv_ignorowane_przy_gestym_lanie():
    field = {"parcel_id": "A", "irrigation_mode": "rainfed"}
    obs = [_vv_obs(d, -10.0 + 0.1 * (d % 2), 22, ndvi=0.8) for d in range(6, 0, -1)]
    obs.append(_vv_obs(0, -14.0, 22, ndvi=0.8))
    r = ws.classify_water_signal(field, obs, [], [])
    assert r["components"]["own_vv"]["significant_drop"] is True
    assert r["components"]["own_vv"]["used_for_decision"] is False
    assert r["risk_flag"] is False


# --- równomierność NDMI w polu ------------------------------------------------

def _uni_obs(days_ago, p10, p50, low_frac, mean=0.30):
    o = _mk_obs(days_ago, mean)
    o.update({"ndwi_p10": p10, "ndwi_p50": p50, "ndwi_low_frac": low_frac})
    return o


def test_niejednorodnosc_przy_normalnej_sredniej_daje_alarm_rownomiernosci():
    field = {"parcel_id": "UZB-7", "irrigation_mode": "irrigation"}
    hist = [_uni_obs(d, 0.26, 0.31, 0.03) for d in range(6, 0, -1)]
    hist[1]["ndwi_p10"] = 0.255
    obs = hist + [_uni_obs(0, 0.12, 0.31, 0.25)]   # mediana bez zmian, dół rozkładu odpłynął
    r = ws.classify_water_signal(field, obs, [], [])
    assert r["risk_flag"] is True
    assert r["risk_type"] == "irrigation_uniformity_anomaly"
    assert "25%" in r["message_pl"] and "bruzd" in r["message_pl"]


def test_rownomiernosc_bez_alarmu_gdy_maly_udzial_suchych_pikseli():
    hist = [_uni_obs(d, 0.26, 0.31, 0.03) for d in range(6, 0, -1)]
    obs = hist + [_uni_obs(0, 0.18, 0.31, 0.08)]
    r = ws.uniformity_anomaly(obs)
    assert r["flagged"] is False


def test_alarm_rownomiernosci_nie_wchodzi_do_kalibracji_progu_spadku(isolated_db, monkeypatch):
    import json
    comps = json.dumps({"own_ndwi": {"z_score": -3.0}})
    alarms = [{"triage_status": "confirmed", "risk_type": "irrigation_uniformity_anomaly",
               "components_json": comps}] * 20
    monkeypatch.setattr(isolated_db, "get_water_alarms", lambda **kw: alarms)
    assert ws.compute_suggested_water_threshold(min_samples=5) is None


def test_inner_buffer_utm_poprawny_dla_uzbekistanu():
    from pyproj import Geod
    from shapely.geometry import box
    g = box(66.9, 39.6, 66.91, 39.61)
    geod = Geod(ellps="WGS84")
    a0 = abs(geod.geometry_area_perimeter(g)[0])
    a1 = abs(geod.geometry_area_perimeter(ws.inner_buffer_utm(g, 10.0))[0])
    side = a0 ** 0.5
    assert abs((side - a1 ** 0.5) / 2 - 10.0) < 0.3     # efektywny bufor ~10 m


def test_inner_buffer_utm_mala_dzialka_bez_bufora():
    from shapely.geometry import box
    g = box(17.0, 51.1, 17.0002, 51.1002)   # ~14 x 22 m
    assert ws.inner_buffer_utm(g, 10.0).equals(g)
