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
