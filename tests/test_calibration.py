"""Testy sugestii/aplikacji wag Confidence Score w calibration.py —
analogiczne do istniejącej sugestii progów NDVI, ale dla wag wskaźników."""
import pytest


def _make_photo(db, parcel_db_id, alarm_id, verdict, coverage, ndvi,
                 evi=None, savi=None, ndre=None, ndwi=None, rvi=None, reip=None):
    pid = db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake.jpg", ndvi_at_alarm=ndvi,
                              evi_at_alarm=evi, savi_at_alarm=savi, ndre_at_alarm=ndre,
                              ndwi_at_alarm=ndwi, rvi_at_alarm=rvi, reip_at_alarm=reip)
    db.update_photo_analysis(pid, ai_analysis="{}", ai_coverage_pct=coverage, ai_verdict=verdict)
    return pid


def test_compute_suggested_weights_wymaga_min_samples(isolated_db):
    import calibration
    isolated_db.init_db()
    assert calibration.compute_suggested_weights(min_samples=10) is None


def test_compute_suggested_weights_faworyzuje_lepiej_rozdzielajacy_wskaznik(isolated_db):
    """EVI idealnie rozdziela zdjęcia wegetacja/gleba, NDRE ma identyczną
    wartość po obu stronach (zerowa separacja) -> po sugestii EVI powinien
    zyskać wagę względem obecnej, NDRE nie powinien zyskać."""
    import calibration
    db = isolated_db
    db.init_db()
    parcel_db_id = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")

    # NDVI: separacja umiarkowana (nie saturuje progów bare/high).
    # EVI: separacja niemal maksymalna (veg blisko "high", bare poniżej "bare").
    # NDRE: identyczna wartość po obu stronach -> zerowa separacja, pomijany.
    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "vegetation_present", 60,
                    ndvi=0.35, evi=0.45, ndre=0.25)
    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "bare_soil", 5,
                    ndvi=0.20, evi=0.08, ndre=0.25)

    result = calibration.compute_suggested_weights(min_samples=10, phase="peak")
    assert result is not None
    assert result["suggested_weights"]["evi"] > result["current_weights"]["evi"]
    assert result["suggested_weights"]["ndre"] <= result["current_weights"]["ndre"] + 1e-6
    total = sum(result["suggested_weights"].values())
    assert abs(total - 1.0) < 0.01


def test_compute_suggested_weights_zachowuje_wage_bez_wystarczajacych_danych(isolated_db):
    """Wskaźniki z za małą liczbą próbek (< min_per_indicator) nie są
    korygowane — zachowują obecną wagę fazy zamiast zostać wyzerowane."""
    import calibration
    import analytics
    db = isolated_db
    db.init_db()
    parcel_db_id = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")

    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "vegetation_present", 60, ndvi=0.5, evi=0.45)
    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "bare_soil", 5, ndvi=0.1, evi=0.08)
    # savi/ndre/ndwi/rvi nie mają żadnych próbek w tym teście

    result = calibration.compute_suggested_weights(min_samples=10, phase="peak")
    assert result is not None
    current = analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]
    assert result["suggested_weights"]["rvi"] == pytest.approx(current["rvi"], abs=0.01)
    assert result["suggested_weights"]["savi"] == pytest.approx(current["savi"], abs=0.01)


def test_apply_weights_scala_a_nie_nadpisuje_poprzedniego_override(isolated_db):
    """Naprawa: dawniej apply_weights('peak', {...}) całkowicie zastępowało
    poprzednie nadpisanie tej fazy. Drugie wywołanie z inną, kompletną paczką
    wag powinno zastąpić wszystkie 6 kluczy (tak jak przesłano) — ale jeśli
    operator zatwierdzi tylko PODZBIÓR wskaźników w drugim wywołaniu, reszta
    z pierwszego wywołania powinna przetrwać (merge, nie replace)."""
    import calibration
    import analytics
    isolated_db.init_db()

    calibration.apply_weights("peak", {
        "ndvi": 0.30, "evi": 0.30, "savi": 0.10, "ndre": 0.10, "rvi": 0.10, "ndwi": 0.10,
    })
    # drugie wywołanie zmienia TYLKO evi — savi/ndre/rvi/ndwi/ndvi z pierwszego
    # wywołania powinny przetrwać
    calibration.apply_weights("peak", {"evi": 0.35})

    assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]["evi"] == pytest.approx(0.35)
    assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]["ndvi"] == pytest.approx(0.30)
    assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]["savi"] == pytest.approx(0.10)


def test_compute_suggested_weights_karze_wskaznik_z_ujemna_separacja(isolated_db):
    """Naprawa: wskaźnik z wystarczającą liczbą próbek, ale zerową/ujemną
    separacją (potwierdzony słaby/odwrotny sygnał) powinien dostać KARĘ
    (waga obniżona), a nie po prostu zachować obecną wagę jak przy braku
    danych."""
    import calibration
    db = isolated_db
    db.init_db()
    parcel_db_id = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")

    # NDVI/EVI: dobra separacja. SAVI: dobrze spróbkowany, ale ODWROTNY sygnał
    # (wyższy dla gołej gleby niż dla wegetacji).
    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "vegetation_present", 60,
                    ndvi=0.35, evi=0.45, savi=0.10)
    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "bare_soil", 5,
                    ndvi=0.20, evi=0.08, savi=0.35)

    result = calibration.compute_suggested_weights(min_samples=10, phase="peak")
    assert result is not None
    assert result["indicator_separation"]["savi"] < 0
    assert result["suggested_weights"]["savi"] == pytest.approx(
        result["current_weights"]["savi"] * calibration.NEGATIVE_SEPARATION_PENALTY, abs=0.01)
    assert result["suggested_weights"]["savi"] < result["current_weights"]["savi"]


def test_apply_weights_zapisuje_i_przeladowuje(isolated_db):
    import calibration
    import analytics
    isolated_db.init_db()

    ok = calibration.apply_weights("peak", {
        "ndvi": 0.30, "evi": 0.30, "savi": 0.10, "ndre": 0.10, "rvi": 0.10, "ndwi": 0.10,
    })
    assert ok is True
    assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]["evi"] == pytest.approx(0.30)
    # inne fazy nie powinny zostać naruszone
    assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["early"] == analytics.DEFAULT_CONFIDENCE_WEIGHTS_BY_PHASE["early"]

    # reload_weights() od zera powinien odtworzyć ten sam zapisany override
    analytics.reload_weights()
    assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]["evi"] == pytest.approx(0.30)


def test_compute_suggested_weights_ndre_widzi_blend_z_reip_nie_sam_surowy_ndre(isolated_db):
    """Regresja z audytu: klucz 'ndre' w compute_confidence_score to od
    kroku 3 integracji pasm ŚREDNIA NDRE+REIP (analytics.blend_red_edge_norm),
    nie sam surowy NDRE. Gdyby calibration.py dalej oceniało separację
    czystego ndre_at_alarm, sugestia wagi dla 'ndre' ignorowałaby REIP mimo
    że REIP faktycznie współdecyduje o score w produkcji.

    Test: surowy NDRE jest CELOWO identyczny po obu stronach (zerowa
    separacja sam w sobie), ale REIP wyraźnie różni wegetację od gołej
    gleby. Jeśli kalibracja poprawnie używa blendu, separacja 'ndre' musi
    wyjść DODATNIA (ciągnięta przez REIP) — nie zerowa."""
    import calibration
    db = isolated_db
    db.init_db()
    parcel_db_id = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")

    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "vegetation_present", 60,
                    ndvi=0.35, evi=0.45, ndre=0.25, reip=720.0)
    for _ in range(6):
        _make_photo(db, parcel_db_id, alarm_id, "bare_soil", 5,
                    ndvi=0.20, evi=0.08, ndre=0.25, reip=706.0)

    result = calibration.compute_suggested_weights(min_samples=10, phase="peak")
    assert result is not None
    assert result["indicator_separation"]["ndre"] > 0.1


def test_calibration_status_ostrzega_gdy_wagi_nigdy_niekalibrowane(app_client):
    r = app_client.get("/api/calibration/status")
    assert r.status_code == 200
    body = r.json()
    assert body["weights_never_calibrated_warning"] is True
    assert body["weights_calibrated_phases"] == []


def test_calibration_status_nie_ostrzega_po_zatwierdzeniu_wag(app_client, monkeypatch):
    monkeypatch.delenv("OPERATOR_API_KEY", raising=False)
    r = app_client.post("/api/calibration/apply_weights", json={
        "phase": "peak",
        "weights": {"ndvi": 0.28, "evi": 0.26, "savi": 0.12, "ndre": 0.16, "rvi": 0.10, "ndwi": 0.08},
    })
    assert r.status_code == 200

    r2 = app_client.get("/api/calibration/status")
    body = r2.json()
    assert body["weights_never_calibrated_warning"] is False
    assert "peak" in body["weights_calibrated_phases"]


def test_calibration_suggest_weights_endpoint_brak_danych(app_client):
    r = app_client.get("/api/calibration/suggest_weights")
    assert r.status_code == 200
    assert r.json()["status"] == "not_enough_data"


def test_calibration_apply_weights_waliduje_sume(app_client):
    r = app_client.post("/api/calibration/apply_weights", json={
        "phase": "peak", "weights": {"ndvi": 0.9, "evi": 0.9},
    })
    assert r.status_code == 422


def test_calibration_apply_weights_wymaga_klucza_gdy_ustawiony(app_client, monkeypatch):
    monkeypatch.setenv("OPERATOR_API_KEY", "sekret123")
    body = {"phase": "peak", "weights": {
        "ndvi": 0.28, "evi": 0.26, "savi": 0.12, "ndre": 0.16, "rvi": 0.10, "ndwi": 0.08,
    }}
    r = app_client.post("/api/calibration/apply_weights", json=body)
    assert r.status_code == 401

    r2 = app_client.post("/api/calibration/apply_weights", json=body,
                         headers={"X-Operator-Key": "sekret123"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"
