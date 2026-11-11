"""Porównanie EVI (z B02) i EVI2 (bez B02) na Twoich działkach — materiał do
przeliczenia progów THRESHOLDS["evi"] i współczynnika EVI w modelu biomasy
(analytics.py) po zmianie evalscriptu z listopada 2026.

Dla każdej działki i każdej czystej sceny (maska SCL, >= 60% czystych pikseli)
liczy średnie EVI, EVI2 i NDVI z działki (Statistical API, jedno zapytanie na
działkę na cały okres), dopasowuje zależność EVI2 = a·EVI + b i przelicza
obecne progi EVI na skalę EVI2.

Użycie (z katalogu aplikacji, potrzebne CDSE_CLIENT_ID / CDSE_CLIENT_SECRET):

    python porownaj_evi_evi2.py --z-bazy --od 2026-08-01 --do 2026-11-10
    python porownaj_evi_evi2.py pola.geojson --od 2026-08-01 --do 2026-11-10

--z-bazy: działki z monitored_parcels w monitoring.db (geometria z ULDK).
pola.geojson / .csv: jak w water_replay.py.
Wynik: evi_evi2.csv i evi_evi2_podsumowanie.txt w bieżącym katalogu.
"""
import argparse
import csv
import math
import statistics
import sys
from datetime import date, datetime

EVALSCRIPT_EVI_EVI2 = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "evi",  bands: 1, sampleType: "INT16" },
      { id: "evi2", bands: 1, sampleType: "INT16" },
      { id: "ndvi", bands: 1, sampleType: "INT16" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}
function evaluatePixel(s) {
  const bad = [0, 1, 3, 8, 9, 10, 11];
  if (s.dataMask !== 1 || bad.indexOf(s.SCL) !== -1) {
    return { evi: [0], evi2: [0], ndvi: [0], dataMask: [0] };
  }
  const NIR = s.B08, RED = s.B04, BLUE = s.B02;
  const evi  = 2.5 * (NIR - RED) / (NIR + 6.0*RED - 7.5*BLUE + 1.0 + 1e-9);
  const evi2 = 2.5 * (NIR - RED) / (NIR + 2.4*RED + 1.0 + 1e-9);
  const ndvi = (NIR - RED) / (NIR + RED + 1e-9);
  return {
    evi:  [Math.round(10000 * Math.max(-1, Math.min(2.5, evi)))],
    evi2: [Math.round(10000 * Math.max(-1, Math.min(2.5, evi2)))],
    ndvi: [Math.round(10000 * Math.max(-1, Math.min(1, ndvi)))],
    dataMask: [1]
  };
}
"""

MIN_VALID_FRAC = 0.6


def fetch_evi_pairs(geom_4326, t_start: date, t_end: date) -> list:
    import analytics
    import core
    from sentinel1 import _post_with_retry
    from sh_utils import stat_bounds, stats_time_range, valid_fraction

    bounds, n_px = stat_bounds(geom_4326)
    payload = {
        "input": {"bounds": bounds, "data": [{"type": "sentinel-2-l2a",
                                              "dataFilter": {"maxCloudCoverage": 90}}]},
        "aggregation": {"timeRange": stats_time_range(t_start, t_end),
                        "aggregationInterval": {"of": "P1D"},
                        "resx": 10, "resy": 10, "evalscript": EVALSCRIPT_EVI_EVI2},
    }
    headers = {"Authorization": f"Bearer {core.cdse_token()}", "Content-Type": "application/json"}
    r = _post_with_retry(analytics.CDSE_STATS_URL, payload, headers, timeout=240,
                         refresh_token_fn=analytics._get_token)
    if r.status_code != 200:
        raise RuntimeError(f"Statistical API {r.status_code}: {r.text[:300]}")
    out = []
    for item in r.json().get("data", []):
        o = item.get("outputs", {})
        try:
            st = o["ndvi"]["bands"]["B0"]["stats"]
            vf = valid_fraction(st, n_px)
            vals = {k: float(o[k]["bands"]["B0"]["stats"]["mean"]) / 10000.0
                    for k in ("evi", "evi2", "ndvi")}
        except (KeyError, TypeError, ValueError):
            continue
        if vf is None or vf < MIN_VALID_FRAC or not all(math.isfinite(v) for v in vals.values()):
            continue
        d = datetime.fromisoformat(item["interval"]["from"].replace("Z", "+00:00")).date()
        out.append({"date": d.isoformat(), "valid_frac": round(vf, 3),
                    **{k: round(v, 4) for k, v in vals.items()}})
    return out


def fit_linear(xs: list, ys: list) -> tuple:
    """(a, b, r2) dla y = a·x + b (MNK)."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    a = sxy / sxx if sxx else 1.0
    b = my - a * mx
    r2 = (sxy * sxy) / (sxx * syy) if sxx and syy else 1.0
    return a, b, r2


def summarize(rows: list) -> dict:
    import analytics
    if len(rows) < 3:
        return {"n": len(rows)}
    xs = [r["evi"] for r in rows]
    ys = [r["evi2"] for r in rows]
    a, b, r2 = fit_linear(xs, ys)
    old = analytics.THRESHOLDS["evi"]
    mapped = {k: round(a * v + b, 3) for k, v in old.items()}
    rel = [(r["evi2"] - r["evi"]) / r["evi"] for r in rows if r["evi"] > 0.15]
    norm = analytics._normalize_indicator
    score_diff = [norm(r["evi2"], mapped["bare"], mapped["high"]) - norm(r["evi"], old["bare"], old["high"])
                  for r in rows]
    score_diff_unmapped = [norm(r["evi2"], old["bare"], old["high"]) - norm(r["evi"], old["bare"], old["high"])
                           for r in rows]
    return {"n": len(rows), "a": round(a, 4), "b": round(b, 4), "r2": round(r2, 4),
            "mean_rel_diff_pct": round(100 * statistics.mean(rel), 1) if rel else None,
            "old_thresholds": dict(old), "mapped_thresholds": mapped,
            "mean_abs_norm_diff_unmapped": round(statistics.mean(abs(v) for v in score_diff_unmapped), 4),
            "mean_abs_norm_diff_mapped": round(statistics.mean(abs(v) for v in score_diff), 4),
            "biomass_coef_evi_old": analytics.BIOMASS_COEFFICIENTS["evi"],
            "biomass_coef_evi_mapped": round(analytics.BIOMASS_COEFFICIENTS["evi"] / a, 2) if a else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Porównanie EVI i EVI2 na działkach.")
    ap.add_argument("pola", nargs="?", help="GeoJSON/CSV (jak w water_replay.py)")
    ap.add_argument("--z-bazy", action="store_true", help="działki z monitored_parcels (monitoring.db)")
    ap.add_argument("--od", required=True)
    ap.add_argument("--do", required=True)
    args = ap.parse_args(argv)
    t0, t1 = date.fromisoformat(args.od), date.fromisoformat(args.do)

    from shapely import wkt as shapely_wkt
    import core
    targets = []
    if args.z_bazy:
        import db
        for p in db.get_monitored_parcels():
            try:
                geom, _ = core.fetch_parcel_geometry(p["parcel_id"])
                targets.append((p["parcel_id"], geom))
            except Exception as e:
                print(f"[{p['parcel_id']}] brak geometrii z ULDK: {e}", file=sys.stderr)
    if args.pola:
        import water_replay
        for f in water_replay.load_fields(args.pola):
            targets.append((f["parcel_id"], shapely_wkt.loads(f["geometry_wkt"])))
    if not targets:
        print("Brak działek: podaj plik albo --z-bazy.", file=sys.stderr)
        return 2

    rows = []
    for pid, geom in targets:
        g_buf, _, _ = core.inner_buffer(geom, 10.0)
        try:
            pairs = fetch_evi_pairs(g_buf, t0, t1)
        except Exception as e:
            print(f"[{pid}] błąd: {e}", file=sys.stderr)
            continue
        print(f"[{pid}] czystych scen: {len(pairs)}")
        rows += [{"parcel_id": pid, **r} for r in pairs]

    with open("evi_evi2.csv", "w", encoding="utf-8-sig", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(rows)
    s = summarize(rows)
    if s.get("n", 0) < 3:
        msg = f"Za mało czystych scen ({s.get('n', 0)}) do dopasowania — wydłuż okres albo dodaj działki."
    else:
        msg = "\n".join([
            f"Scen: {s['n']} · EVI2 = {s['a']}·EVI + {s['b']} (R² = {s['r2']})",
            f"Średnia różnica EVI2 względem EVI (EVI > 0.15): {s['mean_rel_diff_pct']}%",
            f"Obecne progi EVI:        {s['old_thresholds']}",
            f"Progi przeliczone (EVI2): {s['mapped_thresholds']}",
            f"Średnia zmiana składnika EVI w Confidence Score (0-1): "
            f"bez przeliczenia {s['mean_abs_norm_diff_unmapped']}, po przeliczeniu {s['mean_abs_norm_diff_mapped']}",
            f"Współczynnik EVI w modelu biomasy: {s['biomass_coef_evi_old']} -> {s['biomass_coef_evi_mapped']}",
            "",
            "Wklej do analytics.py, jeśli zmiana bez przeliczenia jest dla Ciebie istotna:",
            f'    "evi":  {s["mapped_thresholds"]},',
        ])
    with open("evi_evi2_podsumowanie.txt", "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    print("\n" + msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
