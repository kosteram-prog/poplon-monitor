"""
Czy satelity widzą nawodnienia? (moduł wodny, 11.2026)

Dla każdego wpisu z dziennika nawodnień porównujemy sygnały satelitarne
tuż przed i tuż po dacie nawodnienia:

  * OPTRAM W (wilgotność wierzchniej warstwy gleby, Sentinel-2) — wzrost,
    sensowny przy niezwartym łanie (NDVI < OPTRAM_MAX_NDVI);
  * radar VV (Sentinel-1, ta sama orbita względna) — wzrost w dB, tylko przy
    słabym łanie (NDVI < VV_MAX_NDVI), bo gęsty łan zasłania glebę;
  * termika Landsat (TSI) — spadek (łan po nawodnieniu chłodniejszy), przy
    okrywie NDVI ≥ TSI_MIN_NDVI;
  * woda w liściach (NDMI) — wzrost, przy łanie NDVI ≥ NDMI_MIN_NDVI (reaguje,
    gdy rośliny przed nawodnieniem były już w niedoborze).

Deszcz między pomiarami (opad efektywny z bilansu ≥ RAIN_CONFOUND_MM) jest
zgłaszany — wtedy reakcja może pochodzić od deszczu.

Osobno: skoki OPTRAM / termiki, których nie tłumaczy ani wpis w dzienniku,
ani deszcz — kandydaci na nawodnienia spoza dziennika.

Progi robocze, niekalibrowane (oceny na polach FAO-UZ i w sezonie 2027).
"""
from datetime import date, timedelta
from typing import Optional

OPTRAM_RISE = 0.15
OPTRAM_MAX_NDVI = 0.75
VV_RISE_DB = 1.5
VV_MAX_NDVI = 0.5
TSI_DROP = 0.15
TSI_MIN_NDVI = 0.3
NDMI_RISE = 0.03
NDMI_MIN_NDVI = 0.35
RAIN_CONFOUND_MM = 10.0

# okna (dni) — przed nawodnieniem / po nawodnieniu
WIN = {"optram": (10, 6), "vv": (12, 6), "tsi": (16, 8), "ndmi": (10, 8)}

CAND_OPTRAM_RISE = 0.20
CAND_TSI_DROP = 0.20
CAND_MAX_GAP = {"optram": 10, "tsi": 16}


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def _series(obs: list, opt_series: list, thermal: list) -> dict:
    """{sygnał: [(data, wartość, ndvi, orbita)]} posortowane po dacie."""
    out = {"optram": [], "vv": [], "tsi": [], "ndmi": []}
    for r in opt_series or []:
        if r.get("w") is not None:
            out["optram"].append((_d(r["obs_date"]), r["w"], r.get("ndvi"), None))
    for r in thermal or []:
        if r.get("tsi") is not None:
            out["tsi"].append((_d(r["scene_date"]), r["tsi"], r.get("ndvi"), None))
    seen_s1 = set()
    for o in obs or []:
        if o.get("ndwi_mean") is not None:
            out["ndmi"].append((_d(o["obs_date"]), o["ndwi_mean"], o.get("ndvi_mean"), None))
        if o.get("vv_db") is not None:
            s1d = _d(o.get("s1_date") or o["obs_date"])
            key = (s1d, o.get("s1_rel_orbit"))
            if key not in seen_s1:
                seen_s1.add(key)
                out["vv"].append((s1d, o["vv_db"], o.get("ndvi_mean"), o.get("s1_rel_orbit")))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def _rain_between(rain: dict, a: date, b: date) -> float:
    return sum(v for d, v in rain.items() if a < d <= b)


def _check(kind: str, pts: list, ev: date, rain: dict) -> dict:
    before_days, after_days = WIN[kind]
    before = [p for p in pts if ev - timedelta(days=before_days) <= p[0] < ev]
    after = [p for p in pts if ev < p[0] <= ev + timedelta(days=after_days)]
    if kind == "vv" and before and after:
        # ta sama orbita względna — inaczej różnica kąta padania daje skoki 1–2 dB
        # najbliższa para przed/po z tej samej orbity
        pairs = [(bb, aa) for aa in after for bb in before if bb[3] == aa[3]]
        if not pairs:
            return {"status": "no_data"}
        b, a = min(pairs, key=lambda x: (x[1][0] - x[0][0]).days)
    elif before and after:
        b, a = before[-1], after[0]
    else:
        return {"status": "no_data"}
    ndvi = b[2] if b[2] is not None else a[2]
    if ndvi is not None:
        if kind == "optram" and ndvi >= OPTRAM_MAX_NDVI:
            return {"status": "n_a", "reason": "dense_canopy", "ndvi": round(ndvi, 2)}
        if kind == "vv" and ndvi >= VV_MAX_NDVI:
            return {"status": "n_a", "reason": "dense_canopy", "ndvi": round(ndvi, 2)}
        if kind == "tsi" and ndvi < TSI_MIN_NDVI:
            return {"status": "n_a", "reason": "low_cover", "ndvi": round(ndvi, 2)}
        if kind == "ndmi" and ndvi < NDMI_MIN_NDVI:
            return {"status": "n_a", "reason": "low_cover", "ndvi": round(ndvi, 2)}
    delta = a[1] - b[1]
    seen = {"optram": delta >= OPTRAM_RISE, "vv": delta >= VV_RISE_DB,
            "tsi": -delta >= TSI_DROP, "ndmi": delta >= NDMI_RISE}[kind]
    return {"status": "seen" if seen else "not_seen",
            "before": round(b[1], 3), "after": round(a[1], 3), "delta": round(delta, 3),
            "date_before": b[0].isoformat(), "date_after": a[0].isoformat(),
            "rain_mm": round(_rain_between(rain, b[0], a[0]), 1)}


def irrigation_visibility(events: list, obs: list, opt_series: list, thermal: list,
                          balance: Optional[dict], start: Optional[date] = None) -> dict:
    rain = {}
    for d in (balance or {}).get("days") or []:
        if not d.get("is_forecast") and d.get("p_eff_mm"):
            rain[_d(d["date"])] = d["p_eff_mm"]
    s = _series(obs, opt_series, thermal)
    rows = []
    for e in sorted(events or [], key=lambda x: str(x.get("event_date"))):
        ev = _d(e["event_date"])
        if start and ev < start:
            continue
        sig = {k: _check(k, s[k], ev, rain) for k in ("optram", "vv", "tsi", "ndmi")}
        checked = [v for v in sig.values() if v["status"] in ("seen", "not_seen")]
        verdict = "seen" if any(v["status"] == "seen" for v in checked) else ("not_seen" if checked else "no_data")
        rain_mm = max((v.get("rain_mm") or 0 for v in checked), default=0)
        rows.append({"date": ev.isoformat(), "mm": e.get("amount_mm"), "signals": sig, "verdict": verdict,
                     "rain_mm": round(rain_mm, 1), "rain_confound": rain_mm >= RAIN_CONFOUND_MM})
    summary = {k: sum(1 for r in rows if r["verdict"] == k) for k in ("seen", "not_seen", "no_data")}
    summary["total"] = len(rows)
    return {"events": rows, "summary": summary,
            "unlogged": unlogged_candidates(s, events, rain, start)}


def unlogged_candidates(s: dict, events: list, rain: dict, start: Optional[date] = None) -> list:
    """Skoki OPTRAM / termiki bez wpisu w dzienniku i bez deszczu."""
    ev_dates = [_d(e["event_date"]) for e in events or []]
    found = []
    for kind, thr in (("optram", CAND_OPTRAM_RISE), ("tsi", CAND_TSI_DROP)):
        pts = s[kind]
        for prev, cur in zip(pts, pts[1:]):
            if start and cur[0] < start:
                continue
            if (cur[0] - prev[0]).days > CAND_MAX_GAP[kind]:
                continue
            delta = cur[1] - prev[1]
            jump = delta >= thr if kind == "optram" else -delta >= thr
            if not jump:
                continue
            ndvi = prev[2] if prev[2] is not None else cur[2]
            if kind == "optram" and ndvi is not None and ndvi >= OPTRAM_MAX_NDVI:
                continue
            if kind == "tsi" and (ndvi is None or ndvi < TSI_MIN_NDVI):
                continue
            if any(prev[0] - timedelta(days=2) <= d <= cur[0] for d in ev_dates):
                continue
            if _rain_between(rain, prev[0], cur[0]) >= RAIN_CONFOUND_MM:
                continue
            found.append({"from": prev[0].isoformat(), "to": cur[0].isoformat(), "signal": kind,
                          "before": round(prev[1], 3), "after": round(cur[1], 3)})
    # nakładające się okna z dwóch sygnałów -> jeden kandydat
    found.sort(key=lambda c: c["from"])
    merged = []
    for c in found:
        if merged and c["from"] <= merged[-1]["to"]:
            m = merged[-1]
            m["to"] = max(m["to"], c["to"])
            m["signals"] = sorted(set(m["signals"]) | {c["signal"]})
        else:
            merged.append({"from": c["from"], "to": c["to"], "signals": [c["signal"]]})
    return merged
