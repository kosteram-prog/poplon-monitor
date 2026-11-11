"""Wspólne fixture'y pytest dla poplon-monitor.

Testy uruchamiamy z katalogu głównego repo: `pytest` (albo `pytest tests/`).
Wymaga to, żeby cwd był katalogiem repo (dla `StaticFiles(directory="static")`
w app.py) — to standardowe zachowanie pytest przy uruchomieniu z rootdir.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest


@pytest.fixture(autouse=True)
def reset_sentinel_hub_state():
    """Pamięć podręczna szeregów (sh_utils.cached_call), cache geometrii ULDK
    i stan limitera Sentinel Hub (globalna pauza po 429, budżet PU) to
    globalny stan procesu — bez resetu wynik/pauza z jednego testu
    przeciekałyby do kolejnych (np. ten sam poligon i daty w dwóch testach
    parsowania, albo pauza po 429 usypiająca następny test)."""
    import core
    import sentinel1
    import sh_utils
    sh_utils.clear_caches()
    sentinel1._reset_rate_limiter()
    with core._geom_cache_lock:
        core._geom_cache.clear()
    yield
    sh_utils.clear_caches()
    sentinel1._reset_rate_limiter()


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Każdy test dostaje własną, pustą bazę SQLite — testy nigdy nie
    dotykają prawdziwego monitoring.db."""
    import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    yield db


@pytest.fixture(autouse=True)
def reset_calibrated_thresholds():
    """core.NDVI_COVER_HIGH/MID to globalny, mutowalny stan procesu
    (kalibracja podmienia go w pamięci — patrz calibration.py). Bez resetu
    kalibracja wykonana w jednym teście przeciekałaby do kolejnych."""
    import core
    orig_high, orig_mid = core.NDVI_COVER_HIGH, core.NDVI_COVER_MID
    yield
    core.NDVI_COVER_HIGH, core.NDVI_COVER_MID = orig_high, orig_mid


@pytest.fixture(autouse=True)
def reset_water_threshold():
    """water_stress.Z_SCORE_RISK_THRESHOLD to globalny, mutowalny stan procesu
    (apply_water_threshold podmienia go w pamieci, patrz reload_water_threshold)
    -- bez resetu kalibracja wykonana w jednym tescie przecieklaby do kolejnych,
    analogicznie do reset_calibrated_thresholds powyzej."""
    import water_stress
    orig = water_stress.Z_SCORE_RISK_THRESHOLD
    yield
    water_stress.Z_SCORE_RISK_THRESHOLD = orig


@pytest.fixture(autouse=True)
def reset_confidence_weights():
    """analytics.CONFIDENCE_WEIGHTS_BY_PHASE to globalny, mutowalny stan
    procesu (calibration.apply_weights()/reload_weights() podmieniają go w
    pamięci). Bez resetu sugestia wag zastosowana w jednym teście
    przeciekałaby do kolejnych — analogicznie do reset_calibrated_thresholds."""
    import copy
    import analytics
    orig = copy.deepcopy(analytics.CONFIDENCE_WEIGHTS_BY_PHASE)
    yield
    for phase, weights in orig.items():
        analytics.CONFIDENCE_WEIGHTS_BY_PHASE[phase] = weights
    analytics.CONFIDENCE_WEIGHTS = analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]


@pytest.fixture
def app_client(monkeypatch, isolated_db):
    """TestClient z uruchomionym lifespan (db.init_db, core.reload_thresholds,
    tabele analityczne/SAR, scheduler) i zamockowanymi zależnościami
    sieciowymi (ULDK/CDSE geometria + token), żeby testy nie łączyły się
    z prawdziwym internetem."""
    import core
    monkeypatch.setattr(core, "fetch_parcel_geometry",
                         lambda parcel_id: ("FAKE_GEOM", {"powierzchnia_ha": 1.23}))
    monkeypatch.setattr(core, "inner_buffer", lambda geom, m: (geom, None, None))
    monkeypatch.setattr(core, "cdse_token", lambda: "faketoken")

    import app as appmod
    from fastapi.testclient import TestClient
    with TestClient(appmod.app) as client:
        yield client


@pytest.fixture
def fake_ndvi_verdict():
    """Werdykt NDVI 'OBECNY' w kształcie zwracanym przez core.check_parcel —
    do podstawienia w miejsce core.check_parcel w testach batcha/fuzji."""
    def _make(parcel_id="141201_1.0001.6509"):
        return {
            "parcel_id": parcel_id,
            "verdict": {
                "status": "OBECNY", "confidence": "wysoka", "details": [],
                "detail_codes": [], "compliance_issue": False,
                "requires_operator_approval": False, "low_quality_scenes": 0,
                "metrics": {}, "termination": None,
            },
            "category": "green",
            "series": [],
            "parcel_info": {},
        }
    return _make


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Testy nigdy nie łączą się z internetem. Każde zapytanie HTTP, którego
    test sam nie zamockował (requests.get/post, _post_with_retry itp.), kończy
    się ConnectionError — kod produkcyjny traktuje to jak chwilową awarię
    źródła (ścieżki nieblokujące). Mocki na poziomie requests.get / funkcji
    modułu ustawiane w teście mają pierwszeństwo."""
    import requests

    def _blocked(self, method, url, *a, **k):
        raise requests.ConnectionError(f"sieć wyłączona w testach: {method} {url}")
    monkeypatch.setattr(requests.sessions.Session, "request", _blocked)


@pytest.fixture(autouse=True)
def no_weather_cache(monkeypatch):
    """Pamięć podręczna Open-Meteo wyłączona w testach (testy podmieniają
    odpowiedzi API i nie mogą dostawać wyników z poprzedniego testu).
    Testy samej pamięci podręcznej włączają ją jawnie."""
    import weather
    monkeypatch.setattr(weather, "WEATHER_CACHE_ENABLED", False)


@pytest.fixture(autouse=True)
def reset_biomass_models():
    """analytics.BIOMASS_CIRE_MODEL / N_CIRE_MODEL to globalny stan procesu
    (zatwierdzenie modelu w biomass_calibration podmienia go w pamięci)."""
    import analytics
    b, n = analytics.BIOMASS_CIRE_MODEL, analytics.N_CIRE_MODEL
    analytics.BIOMASS_CIRE_MODEL = analytics.N_CIRE_MODEL = None
    yield
    analytics.BIOMASS_CIRE_MODEL, analytics.N_CIRE_MODEL = b, n
