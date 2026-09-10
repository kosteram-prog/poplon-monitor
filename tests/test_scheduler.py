"""Testy scheduler.py: dołączanie pełnego snapshotu wskaźników (nie tylko
NDVI) do alarmu, potrzebnego przez calibration.py do sugestii wag."""
from datetime import date

import pytest


def test_get_multi_indicators_near_dopasowuje_najblizsza_date(isolated_db):
    import analytics
    db = isolated_db
    db.init_db()
    pid = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    analytics.init_analytics_tables(db_path=db.DB_PATH)
    analytics.save_multi_observation(pid, {
        "date": date(2026, 9, 1), "ndvi": 0.5, "evi": 0.4, "savi": 0.35,
        "ndre": 0.3, "ndwi": 0.2, "valid_frac": 1.0,
    }, rvi=0.15, db_path=db.DB_PATH)

    result = analytics.get_multi_indicators_near(pid, date(2026, 9, 3),
                                                  max_delta_days=5, db_path=db.DB_PATH)
    assert result["evi"] == pytest.approx(0.4)
    assert result["rvi"] == pytest.approx(0.15)

    result_far = analytics.get_multi_indicators_near(pid, date(2026, 10, 1),
                                                      max_delta_days=5, db_path=db.DB_PATH)
    assert result_far == {}


def test_add_alarm_with_indicators_dolacza_multi(isolated_db):
    import scheduler
    import analytics
    db = isolated_db
    db.init_db()
    pid = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    analytics.init_analytics_tables(db_path=db.DB_PATH)
    analytics.save_multi_observation(pid, {
        "date": date(2026, 9, 1), "ndvi": 0.5, "evi": 0.4, "savi": 0.35,
        "ndre": 0.3, "ndwi": 0.2, "valid_frac": 1.0,
    }, rvi=0.15, db_path=db.DB_PATH)

    aid = scheduler._add_alarm_with_indicators(pid, "no_vegetation", "2026-09-02", 0.12, "test")
    alarm = db.get_alarm_by_id(aid)
    assert alarm["evi_value"] == pytest.approx(0.4)
    assert alarm["rvi_value"] == pytest.approx(0.15)


def test_add_alarm_with_indicators_dziala_bez_multi_observations(isolated_db):
    """Brak dopasowania w multi_observations (albo tabela jeszcze
    nieistniejąca) nie może zablokować utworzenia alarmu — wskaźniki
    dodatkowe zostają po prostu None."""
    import scheduler
    db = isolated_db
    db.init_db()
    pid = db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    aid = scheduler._add_alarm_with_indicators(pid, "no_vegetation", "2026-09-02", 0.12, "test")
    alarm = db.get_alarm_by_id(aid)
    assert alarm is not None
    assert alarm["evi_value"] is None
