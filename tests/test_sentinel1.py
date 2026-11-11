"""Testy sentinel1.py: retry/backoff na 429/5xx/401 oraz Cross-Ratio jako
pomocniczy (nigdy samodzielnie wyzwalający) sygnał w detect_tillage."""
from datetime import date
from unittest.mock import MagicMock, patch

import sentinel1


def test_post_with_retry_sukces_od_razu():
    with patch("sentinel1.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        r = sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
        assert r.status_code == 200
        assert mock_post.call_count == 1


def test_post_with_retry_429_az_do_sukcesu():
    with patch("sentinel1.requests.post") as mock_post, patch("sentinel1.time.sleep"):
        resp_429 = MagicMock(status_code=429)
        resp_200 = MagicMock(status_code=200)
        mock_post.side_effect = [resp_429, resp_429, resp_200]
        r = sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
        assert r.status_code == 200
        assert mock_post.call_count == 3


def test_post_with_retry_wyczerpuje_proby_bez_zawieszenia():
    with patch("sentinel1.requests.post") as mock_post, patch("sentinel1.time.sleep"):
        mock_post.return_value = MagicMock(status_code=429)
        r = sentinel1._post_with_retry("http://fake", {}, {}, timeout=5)
        assert r.status_code == 429
        assert mock_post.call_count == sentinel1._MAX_RETRIES + 1


def test_post_with_retry_401_odswieza_token():
    with patch("sentinel1.requests.post") as mock_post:
        resp_401 = MagicMock(status_code=401)
        resp_200 = MagicMock(status_code=200)
        mock_post.side_effect = [resp_401, resp_200]
        calls = []
        r = sentinel1._post_with_retry(
            "http://fake", {}, {"Authorization": "Bearer stary"}, timeout=5,
            refresh_token_fn=lambda: calls.append(1) or "nowy-token",
        )
        assert r.status_code == 200
        assert len(calls) == 1


def test_cross_ratio_podnosi_pewnosc_zdarzenia():
    s1_bez_potwierdzenia = [
        {"date": date(2026, 9, 1), "vv_db": -16.0, "rvi": 0.15, "cross_ratio_db": -3.0},
        {"date": date(2026, 9, 8), "vv_db": -12.0, "rvi": 0.15, "cross_ratio_db": -3.2},
    ]
    events = sentinel1.detect_tillage(s1_bez_potwierdzenia)
    assert len(events) == 1 and events[0]["confidence"] == "medium"

    s1_z_potwierdzeniem = [
        {"date": date(2026, 9, 1), "vv_db": -16.0, "rvi": 0.15, "cross_ratio_db": -3.0},
        {"date": date(2026, 9, 8), "vv_db": -12.0, "rvi": 0.15, "cross_ratio_db": -0.5},
    ]
    events2 = sentinel1.detect_tillage(s1_z_potwierdzeniem)
    assert len(events2) == 1 and events2[0]["confidence"] == "high"
    assert "Cross-Ratio" in events2[0]["message"]


def test_cross_ratio_nie_wyzwala_zdarzenia_samodzielnie():
    s1 = [
        {"date": date(2026, 9, 1), "vv_db": -16.0, "rvi": 0.5, "cross_ratio_db": -3.0},
        {"date": date(2026, 9, 8), "vv_db": -15.8, "rvi": 0.5, "cross_ratio_db": -0.5},
    ]
    events = sentinel1.detect_tillage(s1)
    assert events == []


def test_detect_tillage_dziala_bez_pola_cross_ratio():
    s1 = [
        {"date": date(2026, 9, 1), "vv_db": -16.0, "rvi": 0.15},
        {"date": date(2026, 9, 8), "vv_db": -12.0, "rvi": 0.15},
    ]
    events = sentinel1.detect_tillage(s1)
    assert len(events) == 1 and events[0]["confidence"] == "medium"


def test_conf_up_nie_wywala_keyerror_na_nieznanej_wartosci():
    """Utwardzenie: _CONF_UP.get(x, x) — nieznana etykieta pewności nie
    powinna nigdy podnieść KeyError, tylko zostać niezmieniona."""
    assert sentinel1._CONF_UP.get("nieznana", "nieznana") == "nieznana"
    assert sentinel1._CONF_UP.get("low", "low") == "medium"
    assert sentinel1._CONF_UP.get("high", "high") == "high"



def test_fetch_s1_relative_orbits_parsuje_stac_i_filtruje_kierunek(monkeypatch):
    from datetime import date
    from unittest.mock import MagicMock
    from shapely.geometry import box
    import sentinel1
    page = {"features": [
        {"properties": {"datetime": "2026-11-10T05:08:27Z", "sat:orbit_state": "descending",
                        "sat:relative_orbit": 22, "sar:instrument_mode": "IW"}},
        {"properties": {"datetime": "2026-11-08T16:34:45Z", "sat:orbit_state": "ascending",
                        "sat:relative_orbit": 175, "sar:instrument_mode": "IW"}},
        {"properties": {"datetime": "2026-11-05T05:00:30Z", "sat:orbit_state": "descending",
                        "sat:relative_orbit": 124, "sar:instrument_mode": "IW"}},
    ], "links": []}
    monkeypatch.setattr(sentinel1.requests, "get",
                        lambda url, params=None, timeout=None: MagicMock(status_code=200, json=lambda: page))
    out = sentinel1.fetch_s1_relative_orbits(box(17.0, 51.1, 17.01, 51.11),
                                             date(2026, 11, 1), date(2026, 11, 11))
    assert out == {date(2026, 11, 10): 22, date(2026, 11, 5): 124}


def test_fetch_s1_relative_orbits_nieblokujace_przy_bledzie(monkeypatch):
    from datetime import date
    from shapely.geometry import box
    import sentinel1

    def boom(*a, **k):
        raise sentinel1.requests.ConnectionError("brak sieci")
    monkeypatch.setattr(sentinel1.requests, "get", boom)
    assert sentinel1.fetch_s1_relative_orbits(box(17.0, 51.1, 17.01, 51.11),
                                              date(2026, 11, 1), date(2026, 11, 11)) == {}
