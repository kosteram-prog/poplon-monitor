"""
Podejrzenie zasolenia z wielolecia Sentinel-2 (moduł wodny, 11.2026).

Zasolenie rzadko widać w jednym roku — widać je jako plamy, które ROK W ROK
mają słabszy łan, a wiosną (goła gleba) bywają jaśniejsze od reszty pola
(białe naloty soli). Dla każdego z ostatnich SCAN_YEARS sezonów jedno
zapytanie Process API (mozaika ORBIT, maska SCL) daje per piksel:

  * szczyt zieleni łanu w sezonie wegetacyjnym (15.03–30.09, cokolwiek
    rosło — płodozmian bawełna/pszenica nie gubi lat): 90. percentyl NDVI,
  * jasność gołej gleby wiosną (15.02–30.04, NDVI < BARE_NDVI):
    mediana SI = √(B03·B04) (wskaźnik zasolenia z pasm widzialnych),
  * liczbę czystych scen w obu oknach.

Każdy rok normalizujemy medianą pola (zmiana uprawy, pogoda roku), więc
liczy się tylko „słabiej / jaśniej niż reszta pola”. Klasy:

  3 — silne podejrzenie: słaby łan w ≥ WEAK_PERSIST lat i jasna gleba
      wiosną w ≥ BRIGHT_PERSIST lat z gołą glebą,
  2 — podejrzenie: słaby łan w ≥ WEAK_PERSIST lat,
  1 — do obserwacji: słaby łan w ≥ WATCH_PERSIST lat albo sama jasna gleba,
  0 — bez podejrzeń.

To NIE jest pomiar zasolenia: uporczywie słabe miejsca dają też podmokłości,
zagęszczenie, koniec bruzdy, płytka gleba. Wynik wskazuje, GDZIE zmierzyć EC
(punkty pomiarowe + punkt kontrolny w dobrej części pola). Progi robocze.
"""
import io
import json
import logging
from datetime import date
from typing import Optional

import numpy as np

log = logging.getLogger("salinity_scan")

SCAN_YEARS = 6
SPRING = ((2, 15), (4, 30))
GROWING = ((3, 15), (9, 30))
BARE_NDVI = 0.25
MIN_SUMMER_SCENES = 3
MIN_YEAR_MEDIAN_NDVI = 0.35     # rok bez uprawy (ugór, zła data) nie liczy się do wigoru
WEAK_REL = 0.85                 # szczyt NDVI < 85% mediany pola = słaby w tym roku
BRIGHT_REL = 1.10               # SI wiosną > 110% mediany pola = jaśniejsza gleba
WEAK_PERSIST = 0.6
WATCH_PERSIST = 0.4
BRIGHT_PERSIST = 0.5
MIN_VALID_YEARS = 3
MIN_SIEVE_PX = 4                # plamki < 4 px (0,04 ha) doklejane do sąsiadów
MIN_CORE_PX = 20                # pole mniejsze — bez odcinania brzegu (i tak przybliżone)
EDGE_COLOR = (170, 170, 170)
CLASS_COLORS = {0: (214, 228, 199), 1: (244, 208, 63), 2: (230, 126, 34), 3: (192, 57, 43)}
EST_PU_PER_YEAR = 1.0

EVALSCRIPT_SALT = """
//VERSION=3
// Podejrzenie zasolenia (salinity_scan.py). Wyjście FLOAT32:
// [NDVI_p90 lato, SI mediana wiosna (goła gleba), n scen lato, n scen goła gleba wiosną]
function setup() {
  return {
    input: [{ bands: ["B03", "B04", "B08", "SCL", "dataMask"] }],
    output: { id: "default", bands: 4, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}
const BAD = [0, 1, 3, 8, 9, 10, 11];
const S0 = Date.parse("__SPRING_FROM__"), S1 = Date.parse("__SPRING_TO__");
const C0 = Date.parse("__CROP_FROM__"), C1 = Date.parse("__CROP_TO__");
function q(a, p) {
  a.sort((x, y) => x - y);
  const pos = (a.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
function evaluatePixel(samples, scenes) {
  const nd = [], si = [];
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.dataMask !== 1 || BAD.indexOf(s.SCL) !== -1) continue;
    const t = scenes[i].date.getTime();
    const v = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-9);
    if (t >= C0 && t <= C1) nd.push(v);
    if (t >= S0 && t <= S1 && v < __BARE__) si.push(Math.sqrt(Math.max(0, s.B03 * s.B04)));
  }
  return [nd.length >= __MIN_SUMMER__ ? q(nd, 0.9) : NaN, si.length ? q(si, 0.5) : NaN, nd.length, si.length];
}
"""


def growing_window(year: int) -> tuple:
    return date(year, *GROWING[0]), date(year, *GROWING[1])


def season_years(crop: Optional[str], today: date, n: int = SCAN_YEARS) -> list:
    """Ostatnie n sezonów z zakończonym sezonem wegetacyjnym."""
    last = today.year if growing_window(today.year)[1] < today else today.year - 1
    return list(range(last - n + 1, last + 1))


def _evalscript(year: int, crop: Optional[str]) -> str:
    c0, c1 = growing_window(year)
    s0, s1 = date(year, *SPRING[0]), date(year, *SPRING[1])
    return (EVALSCRIPT_SALT.replace("__SPRING_FROM__", f"{s0.isoformat()}T00:00:00Z")
            .replace("__SPRING_TO__", f"{s1.isoformat()}T23:59:59Z")
            .replace("__CROP_FROM__", f"{c0.isoformat()}T00:00:00Z")
            .replace("__CROP_TO__", f"{c1.isoformat()}T23:59:59Z")
            .replace("__BARE__", str(BARE_NDVI)).replace("__MIN_SUMMER__", str(MIN_SUMMER_SCENES)))


def fetch_year(geom_4326, year: int, crop: Optional[str]):
    """Process API -> (tablica 4×H×W, transform)."""
    import analytics
    import core
    import rasterio
    import yield_mvi
    from shapely.geometry import mapping
    from sentinel1 import _post_with_retry
    c1 = growing_window(year)[1]
    resx, resy = yield_mvi._resolution_deg(geom_4326.centroid.y)
    payload = {
        "input": {
            "bounds": {"geometry": mapping(geom_4326),
                       "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{date(year, *SPRING[0]).isoformat()}T00:00:00Z",
                                                   "to": f"{max(c1, date(year, *SPRING[1])).isoformat()}T23:59:59Z"},
                                     "maxCloudCoverage": 80}}],
        },
        "output": {"resx": resx, "resy": resy,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": _evalscript(year, crop),
    }
    headers = {"Authorization": f"Bearer {core.cdse_token()}", "Content-Type": "application/json",
               "Accept": "image/tiff"}
    r = _post_with_retry(yield_mvi.CDSE_PROCESS_URL, payload, headers, timeout=240,
                         refresh_token_fn=analytics._get_token)
    if r.status_code != 200:
        raise RuntimeError(f"Sentinel Hub Process API błąd {r.status_code}: {r.text[:300]}")
    with rasterio.MemoryFile(r.content) as mem, mem.open() as ds:
        return ds.read().astype("float32"), ds.transform


def classify(stack: list, inside) -> dict:
    """stack: [(rok, tablica 4×H×W)] na tej samej siatce; inside: maska pola.
    Czysta numpy — testowalna bez sieci."""
    weak_n = np.zeros(inside.shape, dtype=float)
    vig_n = np.zeros(inside.shape, dtype=float)
    rel_sum = np.zeros(inside.shape, dtype=float)
    bright_n = np.zeros(inside.shape, dtype=float)
    bare_n = np.zeros(inside.shape, dtype=float)
    years_vigor, years_bare, skipped = [], [], []
    for year, arr in stack:
        nd, si = arr[0], arr[1]
        ok = inside & np.isfinite(nd)
        med = float(np.median(nd[ok])) if ok.sum() >= 4 else float("nan")
        if np.isfinite(med) and med >= MIN_YEAR_MEDIAN_NDVI and ok.sum() >= 0.5 * inside.sum():
            rel = np.where(ok, nd / med, np.nan)
            weak_n += np.where(ok & (rel < WEAK_REL), 1, 0)
            vig_n += ok
            rel_sum += np.where(ok, rel, 0)
            years_vigor.append(year)
        else:
            skipped.append(year)
        okb = inside & np.isfinite(si)
        if okb.sum() >= 0.5 * inside.sum():
            medb = float(np.median(si[okb]))
            if medb > 0:
                bright_n += np.where(okb & (si / medb > BRIGHT_REL), 1, 0)
                bare_n += okb
                years_bare.append(year)
    with np.errstate(invalid="ignore", divide="ignore"):
        persist = np.where(vig_n > 0, weak_n / vig_n, np.nan)
        bright = np.where(bare_n > 0, bright_n / bare_n, np.nan)
        rel_mean = np.where(vig_n > 0, rel_sum / vig_n, np.nan)
    enough = inside & (vig_n >= MIN_VALID_YEARS)
    p = np.nan_to_num(persist, nan=0.0)
    b = np.nan_to_num(bright, nan=0.0)
    has_bright = len(years_bare) >= 2
    cls = np.zeros(inside.shape, dtype="uint8")
    cls[enough & ((p >= WATCH_PERSIST) | (has_bright & (b >= BRIGHT_PERSIST)))] = 1
    cls[enough & (p >= WEAK_PERSIST)] = 2
    cls[enough & (p >= WEAK_PERSIST) & has_bright & (b >= BRIGHT_PERSIST)] = 3
    return {"classes": cls, "valid": enough, "persist": persist, "bright": bright, "rel_mean": rel_mean,
            "years_vigor": years_vigor, "years_bare": years_bare, "years_skipped": skipped}


def core_mask(inside):
    """Pole bez brzegu szerokości 1 piksela (10 m): piksele brzegowe mieszają łan z drogą,
    rowem czy kanałem i rok w rok wychodzą „słabsze” — test na żywo 11.2026."""
    core = inside.copy()
    core[1:, :] &= inside[:-1, :]
    core[:-1, :] &= inside[1:, :]
    core[:, 1:] &= inside[:, :-1]
    core[:, :-1] &= inside[:, 1:]
    core[0, :] = core[-1, :] = False
    core[:, 0] = core[:, -1] = False
    return core if core.sum() >= MIN_CORE_PX else inside


def _sieve(cls, valid):
    try:
        from rasterio.features import sieve
        out = sieve(cls, size=MIN_SIEVE_PX, mask=valid)
        return np.where(valid, out, 0).astype("uint8")
    except Exception as e:
        log.warning("Sito zasolenia niedostępne: %s", e)
        return cls


def sampling_points(cls, valid, rel_mean, transform, n: int = 3) -> list:
    """Do 3 punktów w największych plamach klasy ≥ 2 (albo 1, gdy brak) +
    punkt kontrolny w najlepszej części pola."""
    from rasterio.features import shapes
    from shapely.geometry import shape
    pts = []
    for level in (2, 1):
        mask = (cls >= level) & valid
        if mask.any():
            polys = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
            for k, pg in enumerate(sorted(polys, key=lambda g: g.area, reverse=True)[:n]):
                rp = pg.representative_point()
                pts.append({"n": k + 1, "lat": round(rp.y, 6), "lon": round(rp.x, 6), "kind": "suspect",
                            "level": level})
            break
    good = valid & (cls == 0) & np.isfinite(rel_mean)
    if good.any():
        r, c = np.unravel_index(np.nanargmax(np.where(good, rel_mean, -np.inf)), rel_mean.shape)
        x, y = transform * (c + 0.5, r + 0.5)
        pts.append({"n": "K", "lat": round(y, 6), "lon": round(x, 6), "kind": "control"})
    return pts


def render_png(cls, valid, inside, points, transform, max_side: int = 360) -> bytes:
    from PIL import Image, ImageDraw
    h, w = cls.shape
    k = max(1, int(max_side / max(h, w)))
    rgba = np.zeros((h, w, 4), dtype="uint8")
    for c, col in CLASS_COLORS.items():
        sel = valid & (cls == c)
        rgba[sel, :3] = col
        rgba[sel, 3] = 235
    nodata = inside & ~valid
    rgba[nodata] = (*EDGE_COLOR, 150)
    img = Image.fromarray(rgba, "RGBA").resize((w * k, h * k), Image.NEAREST)
    d = ImageDraw.Draw(img)
    inv = ~transform
    for p in points:
        cx, cy = inv * (p["lon"], p["lat"])
        x, y = cx * k, cy * k
        rad = max(6, k * 0.6)
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 255, 255, 255),
                  outline=(20, 20, 20, 255), width=2)
        d.text((x - 3, y - 6), str(p["n"]), fill=(20, 20, 20, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def scan_field(field: dict, today: Optional[date] = None, fetch=fetch_year) -> dict:
    """Pełny przebieg dla pola: pobranie lat, klasy, punkty, PNG. Zapis w bazie."""
    import db
    import yield_mvi
    from shapely import wkt as shapely_wkt
    today = today or date.today()
    geom = shapely_wkt.loads(field["geometry_wkt"])
    crop = field.get("crop_type")
    stack, transform, errors = [], None, []
    for y in season_years(crop, today):
        try:
            arr, tr = fetch(geom, y, crop)
        except Exception as e:
            errors.append({"year": y, "error": str(e)[:200]})
            continue
        if transform is None:
            transform, shape0 = tr, arr.shape
        if arr.shape != shape0:
            errors.append({"year": y, "error": "inna siatka"})
            continue
        stack.append((y, arr))
    if not stack:
        raise RuntimeError("Brak danych Sentinel-2 dla żadnego roku: " + "; ".join(e["error"] for e in errors))
    inside = yield_mvi.field_mask(geom, stack[0][1].shape[1:], transform)
    core = core_mask(inside)
    res = classify(stack, core)
    cls = _sieve(res["classes"], res["valid"])
    pts = sampling_points(cls, res["valid"], res["rel_mean"], transform)
    n_in = int(core.sum())
    n_val = int(res["valid"].sum())
    shares = {str(c): round(100.0 * int(((cls == c) & res["valid"]).sum()) / n_val, 1) if n_val else 0.0
              for c in (0, 1, 2, 3)}
    status = "ok" if n_val >= 0.5 * n_in and len(res["years_vigor"]) >= MIN_VALID_YEARS else "too_few_years"
    out = {"status": status, "created_at": today.isoformat(), "years_vigor": res["years_vigor"],
           "years_bare": res["years_bare"], "years_skipped": res["years_skipped"], "errors": errors,
           "pixels": n_in, "pixels_valid": n_val, "pixels_edge": int(inside.sum() - core.sum()), "pixel_m": yield_mvi.PIXEL_M,
           "share_pct": shares, "suspect_pct": round(shares["2"] + shares["3"], 1), "points": pts}
    png = render_png(cls, res["valid"], inside, pts, transform)
    db.save_salinity_scan(field["id"], json.dumps(out), png)
    return out


def estimate_pu(field: dict, today: Optional[date] = None) -> float:
    return round(EST_PU_PER_YEAR * len(season_years(field.get("crop_type"), today or date.today())), 1)
