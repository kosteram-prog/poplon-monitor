"""Testy insar.py: wyszukanie sceny SLC, zlecenie/status/pobranie zadania
koherencji (CDSE On-Demand Production API — zamockowane HTTP), wycięcie
średniej koherencji z syntetycznego rastra, interpretacja progu."""
import io
import zipfile
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import insar


def _fake_synthetic_zip(low_value=0.15, high_value=0.75, crs="EPSG:4326"):
    """Buduje zip z syntetycznym 1-pasmowym GeoTIFF 10x10 w EPSG:4326:
    lewa połowa = low_value, prawa połowa = high_value."""
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(19.0, 52.01, 0.001, 0.001)
    data = np.full((10, 10), high_value, dtype="float32")
    data[:, :5] = low_value

    with rasterio.MemoryFile() as memfile:
        with memfile.open(driver="GTiff", height=10, width=10, count=1, dtype="float32",
                          crs=crs, transform=transform, nodata=-9999.0) as dst:
            dst.write(data, 1)
        tif_bytes = memfile.read()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.tif", tif_bytes)
    return buf.getvalue()


# --- find_slc_product ---

def test_find_slc_product_wybiera_najblizsza_date():
    payload = {
        "value": [
            {"Id": "a", "Name": "S1A_far", "ContentDate": {"Start": "2026-09-01T10:00:00.000Z"}},
            {"Id": "b", "Name": "S1A_near", "ContentDate": {"Start": "2026-09-08T10:00:00.000Z"}},
        ]
    }
    with patch("insar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: payload)
        from shapely.geometry import Polygon
        geom = Polygon([(19, 52), (19.01, 52), (19.01, 52.01), (19, 52.01)])
        result = insar.find_slc_product(geom, date(2026, 9, 9))

    assert result["name"] == "S1A_near"
    assert result["content_date"] == date(2026, 9, 8)


def test_find_slc_product_none_gdy_brak_wynikow():
    with patch("insar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"value": []})
        from shapely.geometry import Polygon
        geom = Polygon([(19, 52), (19.01, 52), (19.01, 52.01), (19, 52.01)])
        assert insar.find_slc_product(geom, date(2026, 9, 9)) is None


def test_find_slc_product_rzuca_przy_bledzie_api():
    with patch("insar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=500, text="błąd")
        from shapely.geometry import Polygon
        geom = Polygon([(19, 52), (19.01, 52), (19.01, 52.01), (19, 52.01)])
        with pytest.raises(RuntimeError):
            insar.find_slc_product(geom, date(2026, 9, 9))


# --- submit_coherence_order / get_order_status / download_order_result ---

def test_submit_coherence_order_zwraca_order_id():
    with patch("insar._odp_token", return_value="tok"), \
         patch("insar.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"Id": "order-123"})
        order_id = insar.submit_coherence_order("S1A_scene")

    assert order_id == "order-123"
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["WorkflowName"] == insar.COHERENCE_WORKFLOW_NAME
    assert kwargs["json"]["InputProductReference"]["Reference"] == "S1A_scene"


def test_submit_coherence_order_rzuca_gdy_brak_id():
    with patch("insar._odp_token", return_value="tok"), \
         patch("insar.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
        with pytest.raises(RuntimeError):
            insar.submit_coherence_order("S1A_scene")


def test_get_order_status_normalizuje_pola():
    with patch("insar._odp_token", return_value="tok"), \
         patch("insar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"Status": "queued", "EstimatedDate": "2026-09-10T10:00:00Z"})
        result = insar.get_order_status("order-123")

    assert result["status"] == "queued"
    assert result["estimated_date"] == "2026-09-10T10:00:00Z"


def test_download_order_result_zwraca_bajty():
    with patch("insar._odp_token", return_value="tok"), \
         patch("insar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, content=b"zip-content")
        assert insar.download_order_result("order-123") == b"zip-content"


def test_download_order_result_rzuca_przy_bledzie():
    with patch("insar._odp_token", return_value="tok"), \
         patch("insar.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=404, text="not found")
        with pytest.raises(RuntimeError):
            insar.download_order_result("order-123")


# --- extract_mean_coherence (rasterio na syntetycznym rastrze) ---

def test_extract_mean_coherence_wycina_po_geometrii():
    from shapely.geometry import box
    zip_bytes = _fake_synthetic_zip(low_value=0.15, high_value=0.75)

    # geometria pokrywająca tylko lewą (niską) połowę rastra
    geom_low = box(19.0, 52.003, 19.004, 52.009)
    mean_low = insar.extract_mean_coherence(zip_bytes, geom_low)
    assert mean_low == pytest.approx(0.15, abs=0.01)

    # geometria pokrywająca tylko prawą (wysoką) połowę
    geom_high = box(19.006, 52.003, 19.009, 52.009)
    mean_high = insar.extract_mean_coherence(zip_bytes, geom_high)
    assert mean_high == pytest.approx(0.75, abs=0.01)


def test_extract_mean_coherence_none_gdy_brak_rastra_w_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "brak GeoTIFF")
    from shapely.geometry import box
    geom = box(19.0, 52.0, 19.01, 52.01)
    assert insar.extract_mean_coherence(buf.getvalue(), geom) is None


def test_extract_mean_coherence_none_gdy_geometria_poza_scena():
    zip_bytes = _fake_synthetic_zip()
    from shapely.geometry import box
    geom_daleko = box(0.0, 0.0, 0.01, 0.01)   # daleko od rastra (19E, 52N)
    assert insar.extract_mean_coherence(zip_bytes, geom_daleko) is None


# --- interpret_coherence ---

def test_interpret_coherence_niska_wartosc_flaguje_zaburzenie():
    r = insar.interpret_coherence(0.10)
    assert r["disturbance_likely"] is True


def test_interpret_coherence_wysoka_wartosc_brak_zaburzenia():
    r = insar.interpret_coherence(0.70)
    assert r["disturbance_likely"] is False


def test_interpret_coherence_none_gdy_brak_danych():
    r = insar.interpret_coherence(None)
    assert r["disturbance_likely"] is None
