"""Testy warstwy bazy danych: kalibrowalne progi (settings) i whitelist
kolumn przy dodawaniu działki do monitoringu."""
import pytest


def test_settings_roundtrip_i_reload_thresholds(isolated_db):
    import core
    isolated_db.init_db()

    assert core.NDVI_COVER_HIGH == core.DEFAULT_NDVI_COVER_HIGH
    assert core.NDVI_COVER_MID == core.DEFAULT_NDVI_COVER_MID

    isolated_db.set_setting("ndvi_cover_high", "0.50")
    isolated_db.set_setting("ndvi_cover_mid", "0.35")
    core.reload_thresholds()

    assert core.NDVI_COVER_HIGH == 0.50
    assert core.NDVI_COVER_MID == 0.35


def test_reload_thresholds_bez_zapisanych_ustawien_uzywa_domyslnych(isolated_db):
    import core
    isolated_db.init_db()
    core.reload_thresholds()
    assert core.NDVI_COVER_HIGH == core.DEFAULT_NDVI_COVER_HIGH
    assert core.NDVI_COVER_MID == core.DEFAULT_NDVI_COVER_MID


def test_add_monitored_parcel_odrzuca_nieznane_kolumny(isolated_db):
    isolated_db.init_db()
    with pytest.raises(ValueError):
        isolated_db.add_monitored_parcel(
            "141201_1.0001.6509", "scierniskowy", 2026,
            evil="DROP TABLE monitored_parcels",
        )


def test_add_monitored_parcel_akceptuje_znane_kolumny(isolated_db):
    isolated_db.init_db()
    pid = isolated_db.add_monitored_parcel(
        "141201_1.0001.6509", "scierniskowy", 2026,
        farmer_name="Jan Kowalski", farmer_phone="+48600000000",
    )
    assert pid == 1
    rows = isolated_db.get_monitored_parcels()
    assert len(rows) == 1
    assert rows[0]["farmer_name"] == "Jan Kowalski"


def test_add_alarm_bez_dodatkowych_wskaznikow_backward_compat(isolated_db):
    """Stare wywołania add_alarm(pid, typ, data, ndvi, msg) — bez nowych
    kwargs evi_value/savi_value/... — muszą nadal działać (kolumny nullable)."""
    isolated_db.init_db()
    pid = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    aid = isolated_db.add_alarm(pid, "no_vegetation", "2026-09-01", 0.12, "test")
    alarm = isolated_db.get_alarm_by_id(aid)
    assert alarm["ndvi_value"] == 0.12
    assert alarm["evi_value"] is None
    assert alarm["rvi_value"] is None


def test_add_alarm_zapisuje_pelny_snapshot_wskaznikow(isolated_db):
    isolated_db.init_db()
    pid = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    aid = isolated_db.add_alarm(pid, "no_vegetation", "2026-09-01", 0.12, "test",
                                evi_value=0.10, savi_value=0.09, ndre_value=0.08,
                                ndwi_value=0.05, rvi_value=0.15)
    alarm = isolated_db.get_alarm_by_id(aid)
    assert alarm["evi_value"] == 0.10
    assert alarm["savi_value"] == 0.09
    assert alarm["ndre_value"] == 0.08
    assert alarm["ndwi_value"] == 0.05
    assert alarm["rvi_value"] == 0.15


def test_add_farmer_photo_zapisuje_pelny_snapshot_wskaznikow(isolated_db):
    isolated_db.init_db()
    pid = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    aid = isolated_db.add_alarm(pid, "no_vegetation", "2026-09-01", 0.12, "test")
    photo_id = isolated_db.add_farmer_photo(
        aid, pid, "/tmp/fake.jpg", ndvi_at_alarm=0.12,
        evi_at_alarm=0.10, savi_at_alarm=0.09, ndre_at_alarm=0.08,
        ndwi_at_alarm=0.05, rvi_at_alarm=0.15,
    )
    photos = isolated_db.get_photos_by_alarm(aid)
    photo = next(p for p in photos if p["id"] == photo_id)
    assert photo["evi_at_alarm"] == 0.10
    assert photo["rvi_at_alarm"] == 0.15
