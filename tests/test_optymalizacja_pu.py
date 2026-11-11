"""Testy optymalizacji zużycia PU Sentinel Hub (listopad 2026):
UTM, wyjścia 16-bit, valid_frac względem działki, pełny dzień t_end,
wspólne zapytanie NDVI/multi (single-flight), poprawiony S1, obsługa 429
z Retry-After i liczenie PU. Bez sieci — _post_with_retry/requests.post
zamockowane."""
import threading
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import Polygon

import analytics
import core
import sentinel1
import sh_utils

# Trójkąt prostokątny pod Toruniem — zajmuje dokładnie połowę swojego bbox
# (jak wydłużona/ukośna działka).
TRI = Polygon([(18.60, 53.00), (18.61, 53.00), (18.60, 53.01)])


# --- sh_utils ------------------------------------------------------------------

@pytest.mark.parametrize("lon,epsg", [(14.5, 32633), (17.9, 32633), (18.0, 32634),
                                       (23.9, 32634), (24.1, 32635)])
def test_utm_epsg_dla_polski(lon, epsg):
    assert sh_utils.utm_epsg(lon, 52.0) == epsg


def test_stat_bounds_utm_i_liczba_pikseli_dzialki():
    bounds, n_poly = sh_utils.stat_bounds(TRI)
    assert bounds["properties"]["crs"].endswith("/32634")
    # pole trójkąta ~ 0.5 * 671 m * 1112 m ~ 37 ha -> ~3700 pikseli 10 m
    assert 3500 < n_poly < 3900


def test_valid_fraction_wzgledem_dzialki_nie_bbox():
    n_poly = 1000.0
    # bbox ma 2000 pikseli, 1000 poza działką (noData) — bezchmurny dzień
    assert sh_utils.valid_fraction({"sampleCount": 2000, "noDataCount": 1000}, n_poly) == 1.0
    # 40% działki pod chmurą
    assert sh_utils.valid_fraction({"sampleCount": 2000, "noDataCount": 1400}, n_poly) == 0.6
    # bez liczby pikseli działki — dawny wzór
    assert sh_utils.valid_fraction({"sampleCount": 2000, "noDataCount": 1000}, None) == 0.5
    assert sh_utils.valid_fraction({"sampleCount": 0}, n_poly) is None


def test_zakres_czasu_obejmuje_caly_dzien_t_end():
    tr = sh_utils.stats_time_range(date(2026, 8, 1), date(2026, 11, 9))
    assert tr == {"from": "2026-08-01T00:00:00Z", "to": "2026-11-10T00:00:00Z"}


# --- S2: fetch_multi_series ---------------------------------------------------

def _multi_item(day, n_all, n_bad, ndvi_enc=5500):
    return {
        "interval": {"from": f"{day}T00:00:00Z", "to": "x"},
        "outputs": {
            "ndvi": {"bands": {"B0": {"stats": {"mean": ndvi_enc, "sampleCount": n_all, "noDataCount": n_bad}}}},
            "evi":  {"bands": {"B0": {"stats": {"mean": 4000}}}},
            "savi": {"bands": {"B0": {"stats": {"mean": 4500}}}},
            "ndre": {"bands": {"B0": {"stats": {"mean": 3000}}}},
            "ndwi": {"bands": {"B0": {"stats": {"mean": 1000}}}},
            "reip": {"bands": {"B0": {"stats": {"mean": 15360}}}},
            "ndti": {"bands": {"B0": {"stats": {"mean": 2200}}}},
        },
    }


class _FakeResp:
    status_code = 200
    def __init__(self, data):
        self._data = data
    def json(self):
        return {"data": self._data}


def _fake_post_factory(items, calls, delay=0.0):
    def _fake(url, payload, headers, timeout, refresh_token_fn=None):
        calls.append(payload)
        if delay:
            time.sleep(delay)
        return _FakeResp(items)
    return _fake


def test_multi_payload_utm_int16_pelny_dzien_i_valid_frac_dzialki():
    _, n_poly = sh_utils.stat_bounds(TRI)
    n = int(round(n_poly))
    items = [_multi_item("2026-09-10", 2 * n, n),            # bezchmurnie
             _multi_item("2026-09-12", 2 * n, n + n // 2)]   # pół działki pod chmurą
    calls = []
    with patch("analytics._post_with_retry", side_effect=_fake_post_factory(items, calls)), \
         patch("analytics._get_token", return_value="t"):
        recs = analytics.fetch_multi_series(TRI, date(2026, 9, 1), date(2026, 9, 30))
    p = calls[0]
    assert p["input"]["bounds"]["properties"]["crs"].endswith("/32634")
    assert p["aggregation"]["timeRange"]["to"] == "2026-10-01T00:00:00Z"
    assert '"FLOAT32"' not in p["aggregation"]["evalscript"]
    assert recs[0]["valid_frac"] == pytest.approx(1.0, abs=0.01)
    assert recs[1]["valid_frac"] == pytest.approx(0.5, abs=0.01)
    assert recs[0]["ndvi"] == pytest.approx(0.55)
    assert recs[0]["reip"] == pytest.approx(718.4)
    assert recs[0]["ndti"] == pytest.approx(0.22)


def test_ndvi_core_i_multi_dziela_jedno_zapytanie():
    _, n_poly = sh_utils.stat_bounds(TRI)
    n = int(round(n_poly))
    items = [_multi_item("2026-09-10", 2 * n, n, ndvi_enc=6123)]
    calls = []
    with patch("analytics._post_with_retry", side_effect=_fake_post_factory(items, calls)), \
         patch("analytics._get_token", return_value="t"):
        ndvi = core.fetch_ndvi_series(TRI, date(2026, 9, 1), date(2026, 9, 30))
        multi = analytics.fetch_multi_series(TRI, date(2026, 9, 1), date(2026, 9, 30))
    assert len(calls) == 1
    assert ndvi == [{"date": date(2026, 9, 10), "ndvi_mean": 0.612, "valid_frac": 1.0}]
    assert multi[0]["ndvi"] == pytest.approx(0.6123)


def test_ndvi_core_przycina_koniec_do_dzis():
    items = []
    calls = []
    with patch("analytics._post_with_retry", side_effect=_fake_post_factory(items, calls)), \
         patch("analytics._get_token", return_value="t"):
        core.fetch_ndvi_series(TRI, date(2026, 8, 1), date(2099, 3, 10))
    from datetime import timedelta
    to = calls[0]["aggregation"]["timeRange"]["to"]
    assert to == f"{(date.today() + timedelta(days=1)).isoformat()}T00:00:00Z"


def test_single_flight_rownolegle_watki():
    items = [_multi_item("2026-09-10", 100, 0)]
    calls = []
    results = []
    with patch("analytics._post_with_retry", side_effect=_fake_post_factory(items, calls, delay=0.2)), \
         patch("analytics._get_token", return_value="t"):
        ths = [threading.Thread(target=lambda: results.append(
            analytics.fetch_multi_series(TRI, date(2026, 9, 1), date(2026, 9, 30)))) for _ in range(4)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    assert len(calls) == 1
    assert len(results) == 4 and all(len(r) == 1 for r in results)


def test_ndvi_core_bledy_sentinel_hub_jako_apperror():
    with patch("analytics._post_with_retry", side_effect=RuntimeError("Brak połączenia z Sentinel Hub: x")), \
         patch("analytics._get_token", return_value="t"):
        with pytest.raises(core.AppError):
            core.fetch_ndvi_series(TRI, date(2026, 9, 1), date(2026, 9, 30))


# --- S1 -------------------------------------------------------------------------

def _s1_item(day, vv_enc, vh_enc):
    return {"interval": {"from": f"{day}T00:00:00Z", "to": "x"},
            "outputs": {"vv_lin": {"bands": {"B0": {"stats": {"mean": vv_enc}}}},
                        "vh_lin": {"bands": {"B0": {"stats": {"mean": vh_enc}}}}}}


def test_s1_db_ze_sredniej_liniowej_i_datamask(monkeypatch):
    calls = []
    monkeypatch.setattr(sentinel1, "_cdse_token", lambda force_refresh=False: "t")
    with patch("sentinel1._post_with_retry",
               side_effect=_fake_post_factory([_s1_item("2026-10-01", 1000, 250)], calls)):
        recs = sentinel1.fetch_s1_series(TRI, date(2026, 10, 1), date(2026, 10, 31))
    es = calls[0]["aggregation"]["evalscript"]
    assert '"dataMask"' in es.split("output")[0]          # dataMask pobierany z wejścia
    assert "leeFilter" not in es                          # martwy filtr JS usunięty
    assert calls[0]["input"]["bounds"]["properties"]["crs"].endswith("/32634")
    r = recs[0]
    assert r["vv_lin"] == pytest.approx(0.1)
    assert r["vv_db"] == pytest.approx(-10.0)
    assert r["vh_db"] == pytest.approx(-16.02, abs=0.01)
    assert r["rvi"] == pytest.approx(0.8)
    assert r["cross_ratio_db"] == pytest.approx(-6.02, abs=0.01)


def test_s1_filtr_speckle_konfigurowalny(monkeypatch):
    monkeypatch.setattr(sentinel1, "_cdse_token", lambda force_refresh=False: "t")
    for setting, expected in [("LEE", True), ("NONE", False)]:
        sh_utils.clear_caches()
        monkeypatch.setattr(sentinel1, "S1_SPECKLE_FILTER", setting)
        calls = []
        with patch("sentinel1._post_with_retry", side_effect=_fake_post_factory([], calls)):
            sentinel1.fetch_s1_series(TRI, date(2026, 10, 1), date(2026, 10, 31))
        assert ("speckleFilter" in calls[0]["input"]["data"][0]["processing"]) is expected


# --- 429 / PU -------------------------------------------------------------------

def _resp(code, headers=None):
    m = MagicMock(status_code=code)
    m.headers = headers or {}
    return m


def test_429_retry_after_w_ms_ustawia_globalna_pauze():
    with patch("sentinel1.requests.post") as mock_post, patch("sentinel1.time.sleep") as mock_sleep:
        mock_post.side_effect = [_resp(429, {"Retry-After": "3000"}),
                                 _resp(200, {"x-processingunits-spent": "1.25"})]
        r = sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
    assert r.status_code == 200
    waits = [c.args[0] for c in mock_sleep.call_args_list]
    assert any(3.0 <= w <= 3.6 for w in waits), waits
    st = sentinel1.pu_stats()
    assert st["pu"] == pytest.approx(1.25)
    assert st["requests"] == 2 and st["rate_limited"] == 1


def test_budzet_pu_wstrzymuje_kolejne_zapytanie(monkeypatch):
    monkeypatch.setattr(sentinel1, "_PU_BUDGET_PER_MIN", 2.0)
    with patch("sentinel1.requests.post") as mock_post, patch("sentinel1.time.sleep") as mock_sleep:
        mock_post.return_value = _resp(200, {"x-processingunits-spent": "1.5"})
        sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
        sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
        assert mock_sleep.call_count == 0          # 1.5 + 1.5 szacunku < ... drugi jeszcze przechodzi
        sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
    assert mock_sleep.call_count == 1               # budżet 2 PU/min przekroczony -> czekanie
    assert 55 <= mock_sleep.call_args.args[0] <= 60


# --- zdjęcie --------------------------------------------------------------------

def test_zdjecie_w_natywnej_rozdzielczosci():
    assert core._compute_image_size(500, 400) == (50, 40)
    w, h = core._compute_image_size(100, 80)
    assert max(w, h) == 32
