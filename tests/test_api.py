"""Testy endpointów FastAPI (app.py) przez TestClient — bez sieci, z
zamockowanym core.check_parcel/check_multi tam gdzie trzeba."""
import time

import pytest


def test_check_zwraca_werdykt(app_client, monkeypatch, fake_ndvi_verdict):
    import core
    monkeypatch.setattr(core, "check_parcel",
                         lambda *a, **k: fake_ndvi_verdict())
    r = app_client.post("/api/check", json={
        "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
    })
    assert r.status_code == 200
    assert r.json()["verdict"]["status"] == "OBECNY"


def test_batch_zachowuje_werdykt_ndvi_gdy_fuzja_sie_wywala(app_client, monkeypatch, fake_ndvi_verdict):
    """Regresja krytycznej naprawy: awaria /api/check_multi (S1/multi) w
    batchu NIE MOŻE nadpisywać poprawnego werdyktu NDVI statusem
    NIEROZSTRZYGNIETE — ma zostać zachowany oryginalny werdykt z dopisaną
    notatką informacyjną."""
    import core
    import app as appmod
    monkeypatch.setattr(core, "check_parcel", lambda *a, **k: fake_ndvi_verdict())
    monkeypatch.setattr(appmod, "check_multi",
                         lambda req: (_ for _ in ()).throw(RuntimeError("Sentinel Hub timeout")))

    r = app_client.post("/api/batch/start", json={"items": [
        {"parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026}
    ]})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    data = None
    for _ in range(50):
        r2 = app_client.get(f"/api/batch/status/{job_id}")
        data = r2.json()
        if data["finished"]:
            break
        time.sleep(0.1)
    assert data is not None and data["finished"]

    item = data["items"][0]
    assert item["state"] == "done"
    verdict = item["result"]["verdict"]
    assert verdict["status"] == "OBECNY"
    assert item["result"]["category"] == "green"
    assert any("Pełna analiza wielowskaźnikowa niedostępna" in d for d in verdict["details"])


def test_batch_stosuje_fused_category_z_pelnej_analizy(app_client, monkeypatch, fake_ndvi_verdict):
    """Regresja dla równoległego uruchomienia core.check_parcel + check_multi
    w _process_one (patrz ThreadPoolExecutor w app.py): gdy pełna analiza
    wielowskaźnikowa się powiedzie, kategoria i werdykt w wyniku batcha mają
    pochodzić z NIEJ (fused_category/fused_verdict), nie zostać przy
    wstępnej ocenie z samego NDVI."""
    import core
    import app as appmod
    monkeypatch.setattr(core, "check_parcel", lambda *a, **k: fake_ndvi_verdict())
    monkeypatch.setattr(appmod, "check_multi", lambda req: {
        "fused_verdict": {"status": "PRAWDOPODOBNY", "confidence": "średnia",
                           "details": ["Szczyt NDVI: 0.35."], "detail_codes": [],
                           "compliance_issue": False, "requires_operator_approval": True,
                           "low_quality_scenes": 0},
        "fused_category": "yellow",
        "category": "yellow",
        "summary": {"confidence_score": 42},
    })

    r = app_client.post("/api/batch/start", json={"items": [
        {"parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026}
    ]})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    data = None
    for _ in range(50):
        r2 = app_client.get(f"/api/batch/status/{job_id}")
        data = r2.json()
        if data["finished"]:
            break
        time.sleep(0.1)
    assert data is not None and data["finished"]

    item = data["items"][0]
    assert item["state"] == "done"
    result = item["result"]
    # Kategoria/werdykt musi pochodzić z pełnej analizy (yellow/PRAWDOPODOBNY),
    # NIE zostać na wstępnej ocenie z samego NDVI (fake_ndvi_verdict = green/OBECNY).
    assert result["category"] == "yellow"
    assert result["verdict"]["status"] == "PRAWDOPODOBNY"
    assert result["multi_ready"] is True


def test_batch_limit_rownoczesnych_zadan(app_client):
    import app as appmod
    appmod.MAX_CONCURRENT_BATCH_JOBS = 1
    with appmod._jobs_lock:
        appmod._jobs["fake-active-job"] = {
            "created": time.time(), "items": [], "done_count": 0, "finished": False,
        }
    r = app_client.post("/api/batch/start", json={"items": [
        {"parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026}
    ]})
    assert r.status_code == 429


def test_batch_odrzuca_pusta_liste(app_client):
    r = app_client.post("/api/batch/start", json={"items": []})
    assert r.status_code == 422


def test_calibration_apply_waliduje_progi(app_client):
    r = app_client.post("/api/calibration/apply",
                         json={"new_cover_high": 0.1, "new_cover_mid": 0.9})
    assert r.status_code == 422


def test_calibration_apply_aktualizuje_progi_bez_auth_gdy_klucz_nieustawiony(app_client, monkeypatch):
    monkeypatch.delenv("OPERATOR_API_KEY", raising=False)
    r = app_client.post("/api/calibration/apply",
                         json={"new_cover_high": 0.45, "new_cover_mid": 0.32})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    import core
    assert core.NDVI_COVER_HIGH == 0.45
    assert core.NDVI_COVER_MID == 0.32


def test_calibration_apply_wymaga_klucza_gdy_ustawiony(app_client, monkeypatch):
    monkeypatch.setenv("OPERATOR_API_KEY", "sekret123")
    r = app_client.post("/api/calibration/apply",
                         json={"new_cover_high": 0.40, "new_cover_mid": 0.28})
    assert r.status_code == 401

    r2 = app_client.post("/api/calibration/apply",
                          json={"new_cover_high": 0.40, "new_cover_mid": 0.28},
                          headers={"X-Operator-Key": "sekret123"})
    assert r2.status_code == 200


def test_check_500_nie_wycieka_szczegolow_wyjatku(app_client, monkeypatch, caplog):
    """Niespodziewany wyjątek (np. błąd w kodzie, nie AppError) nie może
    zwracać klientowi surowej treści wyjątku — tylko typ wyjątku i ogólny
    komunikat; pełny szczegół trafia do logów serwera."""
    import core

    def _boom(*a, **k):
        raise RuntimeError("sekret: wewnętrzny szczegół implementacji /mnt/data/klucz.txt")

    monkeypatch.setattr(core, "check_parcel", _boom)
    r = app_client.post("/api/check", json={
        "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
    })
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "sekret" not in detail
    assert "klucz.txt" not in detail
    assert "RuntimeError" in detail


def test_monitor_add_i_list(app_client):
    r = app_client.post("/api/monitor/add", json={
        "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
        "farmer_name": "Jan Kowalski", "farmer_phone": "+48600000000",
    })
    assert r.status_code == 200

    r2 = app_client.get("/api/monitor/list?status=all")
    assert r2.status_code == 200


# --- Koherencja InSAR: /api/parcel/{id}/insar_coherence + /api/insar_coherence/{order_id} ---
# (funkcja "na życzenie operatora" — patrz insar.py. app_client mockuje
# core.fetch_parcel_geometry na zwykły string "FAKE_GEOM", więc tu
# nadpisujemy go realną geometrią shapely, bo endpoint woła geom.wkt.)

def _patch_real_geometry(monkeypatch):
    import core
    from shapely.geometry import Polygon
    geom = Polygon([(19.0, 52.0), (19.01, 52.0), (19.01, 52.01), (19.0, 52.01)])
    monkeypatch.setattr(core, "fetch_parcel_geometry",
                        lambda parcel_id: (geom, {"powierzchnia_ha": 1.23}))
    return geom


def test_insar_coherence_request_zwraca_order_id_natychmiast(app_client, monkeypatch):
    """Zlecenie NIE czeka na wynik — zwraca order_id od razu (status 'queued')."""
    _patch_real_geometry(monkeypatch)
    import insar
    monkeypatch.setattr(insar, "find_slc_product",
                        lambda geom, d, **k: {"id": "p1", "name": "S1A_scene", "content_date": d})
    monkeypatch.setattr(insar, "submit_coherence_order", lambda name, **k: "order-abc")

    r = app_client.post("/api/parcel/141201_1.0001.6509/insar_coherence",
                        json={"event_date": "2026-09-08"})
    assert r.status_code == 200
    data = r.json()
    assert data["order_id"] == "order-abc"
    assert data["status"] == "queued"
    assert data["product_used"] == "S1A_scene"


def test_insar_coherence_request_404_gdy_brak_sceny_slc(app_client, monkeypatch):
    _patch_real_geometry(monkeypatch)
    import insar
    monkeypatch.setattr(insar, "find_slc_product", lambda geom, d, **k: None)

    r = app_client.post("/api/parcel/141201_1.0001.6509/insar_coherence",
                        json={"event_date": "2026-09-08"})
    assert r.status_code == 404


def test_insar_coherence_status_gdy_jeszcze_w_toku(app_client, monkeypatch):
    geom = _patch_real_geometry(monkeypatch)
    import insar
    monkeypatch.setattr(insar, "find_slc_product",
                        lambda g, d, **k: {"id": "p1", "name": "S1A_scene", "content_date": d})
    monkeypatch.setattr(insar, "submit_coherence_order", lambda name, **k: "order-xyz")
    r = app_client.post("/api/parcel/141201_1.0001.6509/insar_coherence",
                        json={"event_date": "2026-09-08"})
    order_id = r.json()["order_id"]

    monkeypatch.setattr(insar, "get_order_status",
                        lambda oid: {"status": "queued", "estimated_date": "2026-09-10T10:00:00Z"})
    r2 = app_client.get(f"/api/insar_coherence/{order_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "queued"


def test_insar_coherence_status_gdy_ukonczone_zwraca_interpretacje(app_client, monkeypatch):
    _patch_real_geometry(monkeypatch)
    import insar
    monkeypatch.setattr(insar, "find_slc_product",
                        lambda g, d, **k: {"id": "p1", "name": "S1A_scene", "content_date": d})
    monkeypatch.setattr(insar, "submit_coherence_order", lambda name, **k: "order-done")
    r = app_client.post("/api/parcel/141201_1.0001.6509/insar_coherence",
                        json={"event_date": "2026-09-08"})
    order_id = r.json()["order_id"]

    monkeypatch.setattr(insar, "get_order_status", lambda oid: {"status": "completed"})
    monkeypatch.setattr(insar, "download_order_result", lambda oid: b"fake-zip-bytes")
    monkeypatch.setattr(insar, "extract_mean_coherence", lambda zb, geom: 0.12)

    r2 = app_client.get(f"/api/insar_coherence/{order_id}")
    assert r2.status_code == 200
    data = r2.json()
    assert data["status"] == "completed"
    assert data["disturbance_likely"] is True
    assert data["mean_coherence"] == 0.12


def test_insar_coherence_status_404_dla_nieznanego_order_id(app_client):
    r = app_client.get("/api/insar_coherence/nieznane-id-123")
    assert r.status_code == 404


# --- Eksport PDF (report_pdf.py) ---

def _fake_multi_response(parcel_id):
    return {
        "parcel_id": parcel_id,
        "parcel_info": {"teryt": "141201_1"},
        "confidence": {"score": 60.0, "score_low": 52.0, "score_high": 68.0,
                       "grade": "B", "label": "Poplon prawdopodobny"},
        "fused_verdict": {"status": "OBECNY"},
        "fused_category": "green",
        "category_reasons": [],
        "peak_record": None,
        "multi_records": [],
        "biomass": None,
        "agro_events": [],
        "weather_context": {}, "weather_error": None,
        "terrain_context": None, "growing_conditions": None,
    }


def test_parcel_report_pdf_zwraca_dokument(app_client, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "check_multi",
                        lambda req: _fake_multi_response(req.parcel_id))
    r = app_client.get("/api/parcel/141201_1.0001.6509/report_pdf",
                       params={"typ": "scierniskowy", "rok": 2026})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_farmer_report_pdf_laczy_dzialki_jednego_rolnika(app_client, monkeypatch):
    import app as appmod
    import db
    monkeypatch.setattr(appmod, "check_multi",
                        lambda req: _fake_multi_response(req.parcel_id))
    db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026,
                            farmer_phone="600100200", farmer_name="Jan Kowalski")
    db.add_monitored_parcel("141201_1.0001.6510", "ozimy", 2026,
                            farmer_phone="600100200", farmer_name="Jan Kowalski")
    db.add_monitored_parcel("141201_1.0001.9999", "scierniskowy", 2026,
                            farmer_phone="500999888", farmer_name="Inny Rolnik")

    r = app_client.get("/api/farmer_report_pdf", params={"farmer_phone": "600100200"})
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"


def test_farmer_report_pdf_404_gdy_brak_dzialek(app_client):
    r = app_client.get("/api/farmer_report_pdf", params={"farmer_phone": "000000000"})
    assert r.status_code == 404


def test_farmer_report_pdf_nie_wywala_calego_raportu_przez_jedna_dzialke(app_client, monkeypatch):
    """Błąd analizy jednej działki (np. chwilowy błąd Sentinel Hub) nie może
    zablokować zbiorczego raportu pozostałych działek tego samego rolnika."""
    import app as appmod
    import db

    def _flaky_check_multi(req):
        if req.parcel_id.endswith("9999"):
            raise RuntimeError("Sentinel Hub timeout")
        return _fake_multi_response(req.parcel_id)

    monkeypatch.setattr(appmod, "check_multi", _flaky_check_multi)
    db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026,
                            farmer_phone="600100200")
    db.add_monitored_parcel("141201_1.0001.9999", "scierniskowy", 2026,
                            farmer_phone="600100200")

    r = app_client.get("/api/farmer_report_pdf", params={"farmer_phone": "600100200"})
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
