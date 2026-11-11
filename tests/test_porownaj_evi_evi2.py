"""Testy skryptu porównania EVI/EVI2 (bez sieci)."""
from unittest.mock import MagicMock

import porownaj_evi_evi2 as pe


def test_fit_linear_odtwarza_zaleznosc():
    xs = [0.1, 0.2, 0.3, 0.5]
    a, b, r2 = pe.fit_linear(xs, [0.95 * x - 0.01 for x in xs])
    assert abs(a - 0.95) < 1e-9 and abs(b + 0.01) < 1e-9 and r2 > 0.999


def test_summarize_przelicza_progi():
    rows = [{"evi": e, "evi2": 0.94 * e} for e in (0.12, 0.2, 0.3, 0.45, 0.6)]
    s = pe.summarize(rows)
    import analytics
    assert s["mapped_thresholds"]["high"] == round(0.94 * analytics.THRESHOLDS["evi"]["high"], 3)
    assert s["mean_abs_norm_diff_mapped"] < s["mean_abs_norm_diff_unmapped"]
    assert s["mean_rel_diff_pct"] == -6.0


def test_fetch_evi_pairs_parsuje_i_odrzuca_zachmurzone(monkeypatch):
    from datetime import date
    from shapely.geometry import box
    import core
    import sentinel1
    monkeypatch.setattr(core, "cdse_token", lambda: "t")

    def stats(mean, n=1000, bad=0):
        return {"bands": {"B0": {"stats": {"mean": mean, "sampleCount": n, "noDataCount": bad}}}}
    data = {"data": [
        {"interval": {"from": "2026-09-01T00:00:00Z"},
         "outputs": {"evi": stats(4000), "evi2": stats(3800), "ndvi": stats(7000)}},
        {"interval": {"from": "2026-09-04T00:00:00Z"},   # 90% chmur
         "outputs": {"evi": stats(1000), "evi2": stats(900), "ndvi": stats(2000, bad=900)}},
    ]}
    monkeypatch.setattr(sentinel1, "_post_with_retry",
                        lambda *a, **k: MagicMock(status_code=200, json=lambda: data))
    g = box(17.0, 51.1, 17.003, 51.102)
    import sh_utils
    monkeypatch.setattr(sh_utils, "stat_bounds", lambda geom: ({}, 1000))
    rows = pe.fetch_evi_pairs(g, date(2026, 9, 1), date(2026, 9, 10))
    assert rows == [{"date": "2026-09-01", "valid_frac": 1.0, "evi": 0.4, "evi2": 0.38, "ndvi": 0.7}]
