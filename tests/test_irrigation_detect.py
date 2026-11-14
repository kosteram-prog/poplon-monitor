"""Czy satelity widzą nawodnienia z dziennika (irrigation_detect.py)."""
from datetime import date, timedelta

import irrigation_detect as idt


def _bal(rain=None, start=date(2026, 6, 1), n=120):
    rain = rain or {}
    return {"days": [{"date": (start + timedelta(days=k)).isoformat(), "is_forecast": False,
                      "p_eff_mm": rain.get((start + timedelta(days=k)).isoformat(), 0.0)} for k in range(n)]}


def test_optram_i_termika_widza_nawodnienie():
    ev = [{"event_date": "2026-07-10", "amount_mm": 120}]
    opt = [{"obs_date": "2026-07-05", "w": 0.25, "ndvi": 0.5}, {"obs_date": "2026-07-12", "w": 0.6, "ndvi": 0.5}]
    th = [{"scene_date": "2026-07-03", "tsi": 0.55, "ndvi": 0.5}, {"scene_date": "2026-07-11", "tsi": 0.2, "ndvi": 0.5}]
    r = idt.irrigation_visibility(ev, [], opt, th, _bal())
    row = r["events"][0]
    assert row["verdict"] == "seen" and row["signals"]["optram"]["status"] == "seen"
    assert row["signals"]["tsi"]["status"] == "seen" and row["signals"]["tsi"]["delta"] == -0.35
    assert row["signals"]["vv"]["status"] == "no_data" and row["rain_confound"] is False
    assert r["summary"] == {"seen": 1, "not_seen": 0, "no_data": 0, "total": 1}


def test_nie_widac_i_deszcz_zaklocajacy():
    ev = [{"event_date": "2026-07-10", "amount_mm": 120}]
    opt = [{"obs_date": "2026-07-05", "w": 0.30, "ndvi": 0.5}, {"obs_date": "2026-07-12", "w": 0.33, "ndvi": 0.5}]
    r = idt.irrigation_visibility(ev, [], opt, [], _bal({"2026-07-08": 15.0}))
    row = r["events"][0]
    assert row["verdict"] == "not_seen" and row["rain_confound"] is True and row["rain_mm"] == 15.0


def test_zwarty_lan_wylacza_optram_i_radar_ale_nie_ndmi():
    ev = [{"event_date": "2026-08-01", "amount_mm": 100}]
    opt = [{"obs_date": "2026-07-28", "w": 0.3, "ndvi": 0.85}, {"obs_date": "2026-08-03", "w": 0.7, "ndvi": 0.85}]
    obs = [{"obs_date": "2026-07-28", "ndwi_mean": 0.20, "ndvi_mean": 0.85, "vv_db": -12, "s1_rel_orbit": 5},
           {"obs_date": "2026-08-03", "ndwi_mean": 0.26, "ndvi_mean": 0.85, "vv_db": -9, "s1_rel_orbit": 5}]
    row = idt.irrigation_visibility(ev, obs, opt, [], _bal())["events"][0]
    assert row["signals"]["optram"] == {"status": "n_a", "reason": "dense_canopy", "ndvi": 0.85}
    assert row["signals"]["vv"]["status"] == "n_a"
    assert row["signals"]["ndmi"]["status"] == "seen" and row["verdict"] == "seen"


def test_radar_tylko_ta_sama_orbita():
    ev = [{"event_date": "2026-05-10", "amount_mm": 80}]
    obs = [{"obs_date": "2026-05-04", "ndvi_mean": 0.2, "vv_db": -14, "s1_rel_orbit": 5},
           {"obs_date": "2026-05-12", "ndvi_mean": 0.2, "vv_db": -10, "s1_rel_orbit": 78}]   # inna orbita
    assert idt.irrigation_visibility(ev, obs, [], [], _bal(start=date(2026, 3, 1)))["events"][0]["signals"]["vv"]["status"] == "no_data"
    obs.append({"obs_date": "2026-05-14", "ndvi_mean": 0.2, "vv_db": -11.5, "s1_rel_orbit": 5})
    vv = idt.irrigation_visibility(ev, obs, [], [], _bal(start=date(2026, 3, 1)))["events"][0]["signals"]["vv"]
    assert vv["status"] == "seen" and vv["delta"] == 2.5 and vv["date_after"] == "2026-05-14"


def test_kandydaci_spoza_dziennika():
    ev = [{"event_date": "2026-07-10", "amount_mm": 120}]
    opt = [{"obs_date": "2026-07-05", "w": 0.2, "ndvi": 0.5}, {"obs_date": "2026-07-12", "w": 0.6, "ndvi": 0.5},   # wpis
           {"obs_date": "2026-07-20", "w": 0.25, "ndvi": 0.5}, {"obs_date": "2026-07-27", "w": 0.65, "ndvi": 0.5},  # bez wpisu
           {"obs_date": "2026-08-04", "w": 0.2, "ndvi": 0.5}, {"obs_date": "2026-08-09", "w": 0.6, "ndvi": 0.5}]   # deszcz
    th = [{"scene_date": "2026-07-19", "tsi": 0.6, "ndvi": 0.5}, {"scene_date": "2026-07-27", "tsi": 0.3, "ndvi": 0.5}]
    r = idt.irrigation_visibility(ev, [], opt, th, _bal({"2026-08-07": 20.0}))
    assert r["unlogged"] == [{"from": "2026-07-19", "to": "2026-07-27", "signals": ["optram", "tsi"]}]


def test_okno_wykresu_pomija_stare_wpisy():
    ev = [{"event_date": "2025-07-10", "amount_mm": 120}, {"event_date": "2026-07-10", "amount_mm": 100}]
    r = idt.irrigation_visibility(ev, [], [], [], _bal(), start=date(2026, 3, 1))
    assert [x["date"] for x in r["events"]] == ["2026-07-10"] and r["summary"]["no_data"] == 1
