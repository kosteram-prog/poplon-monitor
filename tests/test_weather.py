"""Testy weather.py: parsowanie odpowiedzi Open-Meteo, detekcja przymrozku
i posuchy, wybór pierwszego przymrozku sezonu."""
from datetime import date
from unittest.mock import MagicMock, patch

import weather


def _fake_geom(lat=52.0, lon=19.0):
    m = MagicMock()
    centroid = MagicMock()
    centroid.y = lat
    centroid.x = lon
    m.centroid = centroid
    return m


def test_fetch_weather_series_parsuje_odpowiedz_open_meteo():
    payload = {
        "daily": {
            "time": ["2026-09-01", "2026-09-02", "2026-09-03"],
            "temperature_2m_min": [5.0, -3.0, 1.0],
            "temperature_2m_max": [15.0, 5.0, 10.0],
            "temperature_2m_mean": [10.0, 1.0, 5.5],
            "precipitation_sum": [0.0, 2.5, 0.0],
            "shortwave_radiation_sum": [12.0, 8.0, 10.0],
            "et0_fao_evapotranspiration": [2.1, 0.8, 1.5],
        }
    }
    with patch("weather.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: payload)
        series = weather.fetch_weather_series(_fake_geom(), date(2026, 9, 1), date(2026, 9, 3))

    assert series == [
        {"date": date(2026, 9, 1), "tmin_c": 5.0, "tmax_c": 15.0, "tmean_c": 10.0,
         "precip_mm": 0.0, "radiation_mj": 12.0, "et0_mm": 2.1, "vpd_max_kpa": None, "sm_0_7": None, "sm_7_28": None, "sm_28_100": None},
        {"date": date(2026, 9, 2), "tmin_c": -3.0, "tmax_c": 5.0, "tmean_c": 1.0,
         "precip_mm": 2.5, "radiation_mj": 8.0, "et0_mm": 0.8, "vpd_max_kpa": None, "sm_0_7": None, "sm_7_28": None, "sm_28_100": None},
        {"date": date(2026, 9, 3), "tmin_c": 1.0, "tmax_c": 10.0, "tmean_c": 5.5,
         "precip_mm": 0.0, "radiation_mj": 10.0, "et0_mm": 1.5, "vpd_max_kpa": None, "sm_0_7": None, "sm_7_28": None, "sm_28_100": None},
    ]
    # centroid, nie pełna geometria — sprawdzamy że API dostało lat/lon centroidu
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["latitude"] == 52.0
    assert kwargs["params"]["longitude"] == 19.0
    assert "et0_fao_evapotranspiration" in kwargs["params"]["daily"]
    assert kwargs["params"]["timezone"] == "auto"


def test_fetch_weather_series_rzuca_przy_bledzie_api():
    with patch("weather.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=500, text="błąd serwera")
        try:
            weather.fetch_weather_series(_fake_geom(), date(2026, 9, 1), date(2026, 9, 3))
            assert False, "powinien rzucić RuntimeError"
        except RuntimeError:
            pass


def test_detect_frost_dates_uzywa_progu_tmin():
    series = [
        {"date": date(2026, 11, 1), "tmin_c": 1.0, "precip_mm": 0.0},
        {"date": date(2026, 11, 2), "tmin_c": -2.0, "precip_mm": 0.0},   # dokładnie na progu
        {"date": date(2026, 11, 3), "tmin_c": -5.0, "precip_mm": 0.0},
        {"date": date(2026, 11, 4), "tmin_c": None, "precip_mm": 0.0},   # brak danych -> pomiń
    ]
    frost = weather.detect_frost_dates(series)
    assert frost == {date(2026, 11, 2), date(2026, 11, 3)}


def test_detect_dry_spell_dates_liczy_sume_w_oknie():
    # 20 dni bez opadu -> posucha od (window_days)-tego dnia
    series = [{"date": date(2026, 8, 1 + i), "tmin_c": 10.0,
               "precip_mm": 0.0} for i in range(20)]
    dry = weather.detect_dry_spell_dates(series, window_days=14, max_precip_mm=5.0)
    # Pierwsze 14 dni (indeksy 0-13, od dnia 1 do 14 sierpnia) - okno wsteczne
    # jest krótsze niż window_days, więc suma opadów w krótszym oknie i tak
    # wynosi 0 <= 5.0 -> też liczy się jako "posucha" (brak opadów od początku
    # serii). Kluczowe: dzień z realnym opadem PRZERYWA serię.
    assert date(2026, 8, 20) in dry

    series_z_deszczem = list(series)
    series_z_deszczem[15] = {"date": date(2026, 8, 16), "tmin_c": 10.0, "precip_mm": 20.0}
    dry2 = weather.detect_dry_spell_dates(series_z_deszczem, window_days=14, max_precip_mm=5.0)
    # Dzień tuż po realnym opadzie (16-29 sierpnia w oknie 14 dni) NIE jest
    # już posuchą - opad w oknie przekracza próg.
    assert date(2026, 8, 20) not in dry2


def test_first_frost_date_respektuje_after_i_wybiera_najwczesniejszy():
    series = [
        {"date": date(2026, 9, 5), "tmin_c": -3.0, "precip_mm": 0.0},   # wczesny "fałszywy" mróz
        {"date": date(2026, 11, 10), "tmin_c": -2.5, "precip_mm": 0.0},
        {"date": date(2026, 11, 20), "tmin_c": -4.0, "precip_mm": 0.0},
    ]
    # bez "after" -> łapie wrześniowy przypadek
    assert weather.first_frost_date(series) == date(2026, 9, 5)
    # z "after" = 1 listopada -> pomija wrześniowy, wybiera najwcześniejszy PO tej dacie
    assert weather.first_frost_date(series, after=date(2026, 11, 1)) == date(2026, 11, 10)


def test_first_frost_date_none_gdy_brak_przymrozku():
    series = [{"date": date(2026, 9, 1), "tmin_c": 10.0, "precip_mm": 0.0}]
    assert weather.first_frost_date(series) is None


# --- GDD (stopniodni wzrostu) ---

def test_cumulative_gdd_sumuje_max_zero_ponad_baze():
    series = [
        {"date": date(2026, 9, 1), "tmean_c": 15.0},   # GDD = 10
        {"date": date(2026, 9, 2), "tmean_c": 2.0},    # poniżej bazy -> GDD = 0
        {"date": date(2026, 9, 3), "tmean_c": 10.0},   # GDD = 5
    ]
    assert weather.cumulative_gdd(series, date(2026, 9, 1), date(2026, 9, 3)) == 15.0


def test_cumulative_gdd_fallback_na_tmin_tmax_gdy_brak_tmean():
    series = [{"date": date(2026, 9, 1), "tmin_c": 10.0, "tmax_c": 20.0, "tmean_c": None}]
    # (10+20)/2 = 15 -> GDD = 15-5 = 10
    assert weather.cumulative_gdd(series, date(2026, 9, 1), date(2026, 9, 1)) == 10.0


def test_cumulative_gdd_none_gdy_brak_danych_w_oknie():
    series = [{"date": date(2026, 9, 1), "tmean_c": 15.0}]
    assert weather.cumulative_gdd(series, date(2026, 10, 1), date(2026, 10, 5)) is None


def test_cumulative_gdd_ogranicza_sie_do_okna():
    series = [
        {"date": date(2026, 9, 1), "tmean_c": 15.0},
        {"date": date(2026, 9, 10), "tmean_c": 15.0},  # poza oknem
    ]
    assert weather.cumulative_gdd(series, date(2026, 9, 1), date(2026, 9, 2)) == 10.0


def test_growing_conditions_summary_agreguje_pola():
    series = [
        {"date": date(2026, 9, 1), "tmean_c": 15.0, "precip_mm": 2.0, "radiation_mj": 18.0},
        {"date": date(2026, 9, 2), "tmean_c": 10.0, "precip_mm": 0.5, "radiation_mj": 12.0},
    ]
    summary = weather.growing_conditions_summary(series, date(2026, 9, 1), date(2026, 9, 2))
    assert summary["window_days"] == 2
    assert summary["cumulative_gdd"] == 15.0
    assert summary["gdd_base_temp_c"] == weather.GDD_BASE_TEMP_C
    assert summary["total_precip_mm"] == 2.5
    assert summary["total_radiation_mj"] == 30.0


# --- ET0 i klimatyczny bilans wodny ---------------------------------------

def test_climatic_water_balance_sumuje_p_minus_et0():
    from datetime import date, timedelta
    end = date(2026, 7, 14)
    series = [{"date": end - timedelta(days=i), "precip_mm": 1.0, "et0_mm": 4.0}
              for i in range(20)]
    assert weather.climatic_water_balance(series, end, 14) == -42.0


def test_climatic_water_balance_none_gdy_za_duzo_brakow():
    from datetime import date, timedelta
    end = date(2026, 7, 14)
    series = [{"date": end - timedelta(days=i), "precip_mm": 0.0, "et0_mm": 3.0}
              for i in range(5)]
    assert weather.climatic_water_balance(series, end, 14) is None


def test_climatic_water_balance_uzupelnia_pojedyncze_braki_srednia():
    from datetime import date, timedelta
    end = date(2026, 7, 14)
    series = [{"date": end - timedelta(days=i), "precip_mm": 0.0,
               "et0_mm": None if i == 0 else 2.0} for i in range(14)]
    assert weather.climatic_water_balance(series, end, 14) == -28.0



# --- Open-Meteo: klucz komercyjny i pamięć podręczna ------------------------

def test_om_url_z_kluczem_przechodzi_na_serwery_customer(monkeypatch):
    monkeypatch.delenv("OPEN_METEO_API_KEY", raising=False)
    assert weather.om_url("archive") == "https://archive-api.open-meteo.com/v1/archive"
    monkeypatch.setenv("OPEN_METEO_API_KEY", "abc")
    assert weather.om_url("archive") == "https://customer-archive-api.open-meteo.com/v1/archive"
    assert weather.om_url("forecast") == "https://customer-api.open-meteo.com/v1/forecast"


def test_om_get_dokleja_apikey_i_korzysta_z_cache(isolated_db, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setattr(weather, "WEATHER_CACHE_ENABLED", True)
    monkeypatch.setenv("OPEN_METEO_API_KEY", "abc")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params)))
        return MagicMock(status_code=200, json=lambda: {"daily": {"time": ["2026-07-01"]}})
    monkeypatch.setattr(weather.requests, "get", fake_get)
    p = {"latitude": 52.2, "longitude": 16.49, "daily": "precipitation_sum"}
    assert weather.om_get("archive", p, 3600) == {"daily": {"time": ["2026-07-01"]}}
    assert weather.om_get("archive", p, 3600) == {"daily": {"time": ["2026-07-01"]}}
    assert len(calls) == 1                         # drugie z pamięci podręcznej
    assert calls[0][1]["apikey"] == "abc" and calls[0][0].startswith("https://customer-")
    assert "apikey" not in p                       # parametry wołającego nietknięte


def test_om_get_429_to_czytelny_blad(monkeypatch):
    monkeypatch.setattr(weather.requests, "get", lambda url, params=None, timeout=None:
                        MagicMock(status_code=429, text='{"reason":"Daily API request limit exceeded"}'))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="429"):
        weather.om_get("forecast", {"latitude": 1, "longitude": 2}, 60)
