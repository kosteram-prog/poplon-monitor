"""Testy core.assess() — logika werdyktu NDVI (ścierniskowy/ozimy),
w tym przesunięcie okna oceny przy późnym siewie."""
from datetime import date

import core


def _series(entries):
    return [{"date": d, "ndvi_mean": n, "valid_frac": vf} for d, n, vf in entries]


def test_ozimy_okno_oceny_zimowej_konczy_sie_15_lutego():
    """Okno OCENY zgodności ('winter') musi kończyć się na realnym wymogu
    programu LEN (FLD_KSR_004f pkt 3: poplon ozimy musi pozostać do 15.02) —
    używa istniejącej stałej OZIMY_MAINTENANCE_END, nie osobnej, oderwanej
    daty. Okno DANYCH (t_end) jest szersze (do 10.03), żeby operator widział
    trend kawałek za granicą wymogu bez wpływu na sam werdykt."""
    rok = 2025
    winter_start, winter_end = core.CFG["ozimy"]["winter"](rok)
    assert winter_end == date(rok + 1, 2, 15)
    assert winter_end == core.OZIMY_MAINTENANCE_END(rok)
    assert core.CFG["ozimy"]["t_end"](rok) == date(rok + 1, 3, 10)
    assert core.CFG["ozimy"]["t_end"](rok) > winter_end


def test_scheduler_season_end_ozimy_zgodny_z_core_cfg():
    import scheduler
    rok = 2025
    assert scheduler._season_end("ozimy", rok) == core.CFG["ozimy"]["t_end"](rok)
    assert scheduler._season_end("scierniskowy", rok) == core.CFG["scierniskowy"]["t_end"](rok)


def test_scierniskowy_normalny_bieg_bez_daty_siewu():
    rok = 2026
    series = _series([
        (date(rok, 7, 20), 0.20, 0.9),
        (date(rok, 9, 20), 0.50, 0.9),
        (date(rok, 9, 25), 0.48, 0.9),
    ])
    res = core.assess(series, "scierniskowy", rok)
    assert res["status"] == "OBECNY"


def test_scierniskowy_siew_wczesny_okno_bez_zmian():
    rok = 2026
    series = _series([
        (date(rok, 7, 20), 0.20, 0.9),
        (date(rok, 9, 20), 0.50, 0.9),
        (date(rok, 9, 25), 0.48, 0.9),
    ])
    res = core.assess(series, "scierniskowy", rok, sowing_date=date(rok, 7, 1))
    assert res["status"] == "OBECNY"
    assert not any("Okno oceny skrócone" in d for d in res["details"])


def test_scierniskowy_siew_pozny_okno_przesuniete():
    rok = 2026
    series = _series([
        (date(rok, 7, 20), 0.20, 0.9),
        (date(rok, 9, 20), 0.50, 0.9),
        (date(rok, 9, 25), 0.48, 0.9),
    ])
    res = core.assess(series, "scierniskowy", rok, sowing_date=date(rok, 8, 10))
    assert any("Okno oceny skrócone" in d for d in res["details"])


def test_ozimy_siew_normalny_okno_bez_zmian():
    rok = 2026
    series = _series([
        (date(rok, 8, 15), 0.15, 0.9),
        (date(rok, 10, 20), 0.45, 0.9),
        (date(rok, 10, 25), 0.44, 0.9),
    ])
    res = core.assess(series, "ozimy", rok, sowing_date=date(rok, 9, 1))
    assert res["status"] == "OBECNY"
    assert not any("Okno oceny skrócone" in d for d in res["details"])


def test_ozimy_siew_pozny_ale_dopuszczalny_okno_przesuniete():
    """Regresja: przed naprawą sceny sprzed wschodów (bo siew był późny,
    25.10) dawały fałszywy status BRAK, bo trafiały do niezmienionego okna
    oceny. Po naprawie okno się przesuwa i te sceny są odrzucane —> uczciwe
    NIEROZSTRZYGNIETE zamiast fałszywego BRAK."""
    rok = 2026
    series = _series([
        (date(rok, 10, 15), 0.10, 0.9),  # sprzed wschodów
        (date(rok, 10, 28), 0.12, 0.9),  # sprzed wschodów
    ])
    res = core.assess(series, "ozimy", rok, sowing_date=date(rok, 10, 25))
    assert any(
        "Okno oceny skrócone" in d or "wykracza poza standardowe okno" in d
        for d in res["details"]
    )
    assert res["status"] == "NIEROZSTRZYGNIETE"
    assert res["status"] != "BRAK"


def test_ozimy_bez_naprawy_bylby_falszywy_brak():
    """Test kontrolny — potwierdza że naprawa faktycznie coś zmienia
    (bez niego max(NDVI) w niezmodyfikowanym oknie wypadłby <= NDVI_BARE_MAX)."""
    rok = 2026
    series = _series([
        (date(rok, 10, 15), 0.10, 0.9),
        (date(rok, 10, 28), 0.12, 0.9),
    ])
    cover_w = core.CFG["ozimy"]["cover"](rok)
    in_window = [r for r in series if cover_w[0] <= r["date"] <= cover_w[1]]
    old_cover_ndvi = max(r["ndvi_mean"] for r in in_window)
    assert old_cover_ndvi <= core.NDVI_BARE_MAX
