"""Testy terrain.py: estymacja nachylenia/ekspozycji metodą różnic
skończonych i kategoryzacja ryzyka erozji."""
from unittest.mock import MagicMock, patch

import terrain


def _fake_geom(lat=52.0, lon=19.0):
    m = MagicMock()
    centroid = MagicMock()
    centroid.y = lat
    centroid.x = lon
    m.centroid = centroid
    return m


def test_slope_aspect_plaski_teren():
    slope, aspect = terrain._slope_aspect_from_elevations(
        elev_n=100, elev_s=100, elev_e=100, elev_w=100, offset_m=90.0)
    assert slope == 0.0


def test_slope_aspect_stok_opadajacy_na_polnoc():
    # niżej na północy -> spadek w kierunku N -> aspect ~0°
    slope, aspect = terrain._slope_aspect_from_elevations(
        elev_n=90, elev_s=100, elev_e=95, elev_w=95, offset_m=90.0)
    assert slope > 0
    assert aspect == 0.0


def test_slope_aspect_stok_opadajacy_na_wschod():
    # niżej na wschodzie -> spadek w kierunku E -> aspect ~90°
    slope, aspect = terrain._slope_aspect_from_elevations(
        elev_n=95, elev_s=95, elev_e=90, elev_w=100, offset_m=90.0)
    assert aspect == 90.0


def test_erosion_risk_progi():
    assert terrain._erosion_risk(1.0) == "niskie"
    assert terrain._erosion_risk(3.0) == "niskie"
    assert terrain._erosion_risk(4.5) == "srednie"
    assert terrain._erosion_risk(6.0) == "srednie"
    assert terrain._erosion_risk(10.0) == "wysokie"


def test_fetch_terrain_context_parsuje_5_punktow():
    payload = {"elevation": [120.0, 118.0, 122.0, 121.0, 119.0]}  # c,n,s,e,w
    with patch("terrain.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: payload)
        ctx = terrain.fetch_terrain_context(_fake_geom())

    assert ctx is not None
    assert ctx["elevation_m"] == 120.0
    assert ctx["slope_deg"] > 0
    assert ctx["erosion_risk"] in ("niskie", "srednie", "wysokie")
    # 5 punktów wysłanych do API (centroid + N/S/E/W)
    _, kwargs = mock_get.call_args
    assert len(kwargs["params"]["latitude"].split(",")) == 5


def test_fetch_terrain_context_none_przy_bledzie_api():
    with patch("terrain.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=500)
        assert terrain.fetch_terrain_context(_fake_geom()) is None


def test_fetch_terrain_context_none_przy_niekompletnych_danych():
    with patch("terrain.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200,
                                          json=lambda: {"elevation": [120.0, None, 122.0, 121.0, 119.0]})
        assert terrain.fetch_terrain_context(_fake_geom()) is None


def test_fetch_terrain_context_none_przy_wyjatku_sieciowym():
    import requests
    with patch("terrain.requests.get", side_effect=requests.RequestException("timeout")):
        assert terrain.fetch_terrain_context(_fake_geom()) is None
