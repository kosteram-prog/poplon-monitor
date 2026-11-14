"""OPTRAM — wilgotność wierzchniej warstwy gleby z Sentinel-2 (moduł wodny).

OPTRAM (OPtical TRApezoid Model, Sadeghi i in. 2017, Remote Sensing of
Environment 198) wykorzystuje to, że woda silnie pochłania promieniowanie
krótkofalowej podczerwieni (SWIR, Sentinel-2 B12, 2190 nm): im wilgotniejsza
gleba lub roślina, tym ciemniejsza w SWIR. Odbicie R_SWIR przekształca się
wzorem Kubelki-Munka w STR = (1 - R)^2 / (2R) — wielkość w przybliżeniu
liniowo rosnącą z wilgotnością.

Na wykresie STR (oś Y) względem NDVI (oś X) piksele regionu z wielu dat
układają się w trapez:
  * dolna krawędź („sucha”) — najsuchsze piksele przy danym pokryciu roślin:
        STR_d = i_d + s_d · NDVI
  * górna krawędź („mokra”) — najwilgotniejsze:
        STR_w = i_w + s_w · NDVI
Wilgotność względna piksela to jego położenie między krawędziami:
        W = (STR - STR_d) / (STR_w - STR_d),  0 = sucho, 1 = nasycenie.

Krawędzie trzeba wyznaczyć dla REGIONU (inne gleby, inny klimat = inny
trapez), dlatego kalibracja bierze kwadrat wokół pola (domyślnie 1,5 km
zapasu z każdej strony) i do 60 czystych scen z okresu (domyślnie 12
miesięcy), żeby w chmurze punktów były i susze, i mokre okresy.

Ograniczenia, które pokazujemy użytkownikowi:
  * W jest WZGLĘDNE (0–1), nie w % objętościowych — przeliczenie na m³/m³
    wymaga pomiarów w polu (sonda TDR) albo porównania z ERA5-Land;
  * przy zwartym łanie (NDVI > ~0,6) W odzwierciedla raczej uwodnienie
    roślin niż gleby;
  * dotyczy wierzchniej warstwy (kilka cm), nie strefy korzeniowej — dlatego
    to kontekst do werdyktu i bilansu wodnego, nie samodzielny alarm.
"""
import json
import logging
import math
from datetime import date, timedelta
from typing import Optional

log = logging.getLogger("optram")

PIXEL_M = 20.0                 # B12 ma natywnie 20 m
REGION_BUFFER_M = 1500.0       # zapas wokół pola dla chmury punktów kalibracji
MAX_REGION_KM = 8.0
MAX_SCENES = 60
CALIB_DAYS_DEFAULT = 365
NODATA = -32768

BIN_W = 0.02
NDVI_FIT_MIN = 0.10
NDVI_FIT_MAX = 0.90
MIN_BIN_PX = 80
Q_DRY = 0.02
Q_WET = 0.98
MIN_BINS = 6
SCATTER_N = 2500

MIN_SCENES_OK = 8
MIN_PX_OK = 5000
MIN_NDVI_SPAN_OK = 0.40
MIN_EDGE_GAP = 0.3             # STR_w - STR_d na całym zakresie NDVI
# W jest bardzo czułe na niezamaskowane cienie chmur (B12 ciemnieje = „mokro”)
# — test na żywo (pole 101807_5.0013.47) dał W = 1,0 na scenach z 63–77%
# czystych pikseli. Stąd ostrzejszy próg i mediana zamiast średniej z pola.
MIN_SERIES_VALID_FRAC = 0.9
SENTENCE_MAX_AGE_DAYS = 12
DENSE_CANOPY_NDVI = 0.6

W_DRY = 0.30
BARE_SOIL_NDVI = 0.2
# Minimalna zgodność z wilgotnością ERA5-Land 0–7 cm, żeby zdanie OPTRAM
# trafiało do werdyktu (przy >= ERA5_MIN_N wspólnych datach).
ERA5_MIN_N = 10
ERA5_MIN_R = 0.2
W_WET = 0.60

EVALSCRIPT_STACK = """
//VERSION=3
// OPTRAM — chmura punktów (NDVI, STR) do kalibracji trapezu. Dla każdej
// sceny 2 pasma INT16: NDVI·10000 i STR·1000; -32768 = brak (chmura, woda,
// śnieg, cień — zostają tylko klasy SCL 4 roślinność i 5 gleba).
const MAX_SCENES = __MAX_SCENES__;
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "B12", "SCL", "dataMask"] }],
    output: { id: "default", bands: 2, sampleType: "INT16" },
    mosaicking: "ORBIT"
  };
}
function preProcessScenes(collections) {
  const o = collections.scenes.orbits;
  if (o.length > MAX_SCENES) {
    const step = o.length / MAX_SCENES, keep = [];
    for (let i = 0; i < MAX_SCENES; i++) keep.push(o[Math.floor(i * step)]);
    collections.scenes.orbits = keep;
  }
  return collections;
}
function updateOutput(outputs, collection) {
  Object.values(outputs).forEach(o => { o.bands = 2 * Math.max(1, collection.scenes.length); });
}
function evaluatePixel(samples) {
  const out = new Array(2 * Math.max(1, samples.length)).fill(-32768);
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.dataMask !== 1 || (s.SCL !== 4 && s.SCL !== 5)) continue;
    const r = s.B12;
    if (!(r > 0.005 && r < 1.0)) continue;
    const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-9);
    const str = (1 - r) * (1 - r) / (2 * r);
    out[2 * i] = Math.round(10000 * Math.max(-1, Math.min(1, ndvi)));
    out[2 * i + 1] = Math.round(1000 * Math.min(32, str));
  }
  return out;
}
"""

EVALSCRIPT_SERIES = """
//VERSION=3
// OPTRAM — wilgotność względna W pola z gotowych krawędzi trapezu.
const ID = __ID__, SD = __SD__, IW = __IW__, SW = __SW__;
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "B12", "SCL", "dataMask"] }],
    output: [
      { id: "w", bands: 1, sampleType: "INT16" },
      { id: "ndvi", bands: 1, sampleType: "INT16" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}
function evaluatePixel(s) {
  const r = s.B12;
  if (s.dataMask !== 1 || (s.SCL !== 4 && s.SCL !== 5) || !(r > 0.005 && r < 1.0)) {
    return { w: [0], ndvi: [0], dataMask: [0] };
  }
  const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-9);
  const str = (1 - r) * (1 - r) / (2 * r);
  const d = ID + SD * ndvi, wet = IW + SW * ndvi;
  const w = Math.max(0, Math.min(1, (str - d) / Math.max(1e-6, wet - d)));
  return { w: [Math.round(10000 * w)], ndvi: [Math.round(10000 * ndvi)], dataMask: [1] };
}
"""


# ---------------------------------------------------------------------------
# Matematyka (bez I/O — testowalna)
# ---------------------------------------------------------------------------

def str_from_reflectance(r):
    """Kubelka-Munk: STR = (1-R)^2 / (2R)."""
    return (1.0 - r) ** 2 / (2.0 * r)


def _linfit(x, y):
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    s, i = np.polyfit(x, y, 1)
    pred = i + s * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(i), float(s), r2


def _robust_linfit(x, y):
    """Dopasowanie liniowe z jednym przebiegiem odrzucenia punktów odstających
    (reszta > 2,5 MAD) — pojedyncze koszyki NDVI z resztkami chmur/cieni nie
    powinny przechylać krawędzi."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, s, _ = _linfit(x, y)
    res = y - (i + s * x)
    mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
    keep = np.abs(res) <= max(2.5 * mad, 1e-9) if mad > 0 else np.ones_like(x, bool)
    if keep.sum() >= MIN_BINS and keep.sum() < len(x):
        i, s, r2 = _linfit(x[keep], y[keep])
        return i, s, r2, int(len(x) - keep.sum())
    i, s, r2 = _linfit(x, y)
    return i, s, r2, 0


def fit_trapezoid(ndvi, strv) -> dict:
    """Krawędzie trapezu z chmury punktów (NDVI, STR). Rzuca ValueError,
    gdy punktów albo zakresu NDVI jest za mało do dopasowania."""
    import numpy as np
    ndvi = np.asarray(ndvi, float)
    strv = np.asarray(strv, float)
    ok = np.isfinite(ndvi) & np.isfinite(strv) & (strv > 0) & (ndvi >= NDVI_FIT_MIN) & (ndvi <= NDVI_FIT_MAX)
    ndvi, strv = ndvi[ok], strv[ok]
    if ndvi.size < MIN_BIN_PX * MIN_BINS:
        raise ValueError(f"Za mało czystych pikseli do kalibracji ({ndvi.size}).")
    edges = np.arange(NDVI_FIT_MIN, NDVI_FIT_MAX + BIN_W / 2, BIN_W)
    idx = np.digitize(ndvi, edges) - 1
    bins = []
    for k in range(len(edges) - 1):
        m = idx == k
        n = int(m.sum())
        if n < MIN_BIN_PX:
            continue
        v = strv[m]
        bins.append({"ndvi": round(float(edges[k] + BIN_W / 2), 3), "n": n,
                     "dry": float(np.quantile(v, Q_DRY)), "wet": float(np.quantile(v, Q_WET))})
    if len(bins) < MIN_BINS:
        raise ValueError(f"Za wąski zakres NDVI w regionie ({len(bins)} przedziałów po {BIN_W}) — "
                         f"wydłuż okres kalibracji albo powiększ region.")
    x = [b["ndvi"] for b in bins]
    i_d, s_d, r2_d, out_d = _robust_linfit(x, [b["dry"] for b in bins])
    i_w, s_w, r2_w, out_w = _robust_linfit(x, [b["wet"] for b in bins])
    return {"i_d": round(i_d, 4), "s_d": round(s_d, 4), "i_w": round(i_w, 4), "s_w": round(s_w, 4),
            "r2_d": round(r2_d, 3), "r2_w": round(r2_w, 3),
            "ndvi_lo": x[0], "ndvi_hi": x[-1], "n_px": int(ndvi.size),
            "outlier_bins": out_d + out_w,
            "bins": [{**b, "dry": round(b["dry"], 3), "wet": round(b["wet"], 3)} for b in bins]}


def assess_quality(p: dict, n_scenes: int) -> tuple:
    """('good'|'limited'|'poor', [ostrzeżenia po polsku])."""
    warn, poor = [], False
    gaps = [(p["i_w"] + p["s_w"] * v) - (p["i_d"] + p["s_d"] * v) for v in (0.0, 0.45, 0.9)]
    if min(gaps) < MIN_EDGE_GAP:
        poor = True
        warn.append("Krawędzie trapezu schodzą się albo krzyżują w zakresie NDVI 0–0,9 — "
                    "wilgotność z tej kalibracji jest niewiarygodna.")
    if p["s_w"] <= p["s_d"]:
        warn.append("Krawędź mokra nie jest bardziej stroma niż sucha (nietypowy trapez) — "
                    "prawdopodobnie w okresie zabrakło mokrych terminów.")
    if n_scenes < MIN_SCENES_OK:
        warn.append(f"Liczba scen w okresie: {n_scenes} — wydłuż okres kalibracji.")
    if p["n_px"] < MIN_PX_OK:
        warn.append(f"Mało czystych pikseli: {p['n_px']}.")
    if p["ndvi_hi"] - p["ndvi_lo"] < MIN_NDVI_SPAN_OK:
        warn.append(f"Wąski zakres NDVI w regionie ({p['ndvi_lo']:.2f}–{p['ndvi_hi']:.2f}) — "
                    f"krawędzie są ekstrapolowane.")
    if min(p["r2_d"], p["r2_w"]) < 0.5:
        warn.append(f"Słabe dopasowanie krawędzi (R² sucha {p['r2_d']:.2f}, mokra {p['r2_w']:.2f}).")
    if poor:
        return "poor", warn
    return ("good" if not warn else "limited"), warn


def w_from(ndvi: float, strv: float, p: dict) -> float:
    d = p["i_d"] + p["s_d"] * ndvi
    wet = p["i_w"] + p["s_w"] * ndvi
    return max(0.0, min(1.0, (strv - d) / max(1e-6, wet - d)))


def stack_to_points(arr):
    """Tablica (2·N, H, W) INT16 z Process API -> (ndvi[], str[]) float."""
    import numpy as np
    a = np.asarray(arr)
    nd = a[0::2].astype("float64")
    st = a[1::2].astype("float64")
    ok = (nd != NODATA) & (st != NODATA)
    n_scenes = int(np.count_nonzero(ok.reshape(ok.shape[0], -1).any(axis=1)))
    return nd[ok] / 10000.0, st[ok] / 1000.0, n_scenes


def scatter_sample(ndvi, strv, n=SCATTER_N, seed=7) -> list:
    import numpy as np
    ndvi, strv = np.asarray(ndvi), np.asarray(strv)
    if ndvi.size > n:
        idx = np.random.default_rng(seed).choice(ndvi.size, n, replace=False)
        ndvi, strv = ndvi[idx], strv[idx]
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(ndvi, strv)]


def w_class(w: float) -> str:
    if w < W_DRY:
        return "sucho"
    if w < W_WET:
        return "umiarkowanie wilgotno"
    return "wilgotno"


def pearson(xs, ys) -> Optional[float]:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def percentile_of(value: float, values: list) -> Optional[float]:
    if not values:
        return None
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return 100.0 * (below + 0.5 * equal) / len(values)


# ---------------------------------------------------------------------------
# Geometria regionu kalibracji
# ---------------------------------------------------------------------------

def calibration_region(geom_4326, buffer_m: float = REGION_BUFFER_M):
    """Kwadrat (w lokalnym UTM) obejmujący pole z zapasem buffer_m."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as shp_transform
    from sh_utils import utm_epsg
    c = geom_4326.centroid
    epsg = utm_epsg(c.x, c.y)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    minx, miny, maxx, maxy = shp_transform(fwd, geom_4326).bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) / 2 + buffer_m
    half = min(half, MAX_REGION_KM * 500.0)
    return shp_transform(inv, box(cx - half, cy - half, cx + half, cy + half))


def _resolution_deg(lat: float) -> tuple:
    resy = PIXEL_M / 111320.0
    resx = PIXEL_M / (111320.0 * max(0.2, math.cos(math.radians(lat))))
    return resx, resy


# ---------------------------------------------------------------------------
# Sentinel Hub
# ---------------------------------------------------------------------------

def fetch_stack(region_4326, t_start: date, t_end: date):
    """Process API -> tablica (2·N, H, W) INT16."""
    import analytics
    import core
    import rasterio
    from shapely.geometry import mapping
    from sentinel1 import _post_with_retry
    from yield_mvi import CDSE_PROCESS_URL

    resx, resy = _resolution_deg(region_4326.centroid.y)
    payload = {
        "input": {
            "bounds": {"geometry": mapping(region_4326),
                       "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{t_start.isoformat()}T00:00:00Z",
                                                   "to": f"{t_end.isoformat()}T23:59:59Z"},
                                     "maxCloudCoverage": 60}}],
        },
        "output": {"resx": resx, "resy": resy,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT_STACK.replace("__MAX_SCENES__", str(MAX_SCENES)),
    }
    headers = {"Authorization": f"Bearer {core.cdse_token()}", "Content-Type": "application/json",
               "Accept": "image/tiff"}
    r = _post_with_retry(CDSE_PROCESS_URL, payload, headers, timeout=300,
                         refresh_token_fn=analytics._get_token)
    if r.status_code != 200:
        raise RuntimeError(f"Sentinel Hub Process API błąd {r.status_code}: {r.text[:300]}")
    with rasterio.MemoryFile(r.content) as mem, mem.open() as ds:
        return ds.read()


def series_evalscript(p: dict) -> str:
    return (EVALSCRIPT_SERIES.replace("__ID__", repr(float(p["i_d"])))
            .replace("__SD__", repr(float(p["s_d"]))).replace("__IW__", repr(float(p["i_w"])))
            .replace("__SW__", repr(float(p["s_w"]))))


def fetch_series(geom_4326, t_start: date, t_end: date, p: dict) -> list:
    """Statistical API -> [{obs_date, w, ndvi, valid_frac}] dla czystych dni."""
    import analytics
    import core
    from sentinel1 import _post_with_retry
    from sh_utils import stat_bounds, stats_time_range

    bounds, n_px = stat_bounds(geom_4326, res_m=PIXEL_M)
    payload = {
        "input": {"bounds": bounds, "data": [{"type": "sentinel-2-l2a",
                                              "dataFilter": {"maxCloudCoverage": 90}}]},
        "aggregation": {"timeRange": stats_time_range(t_start, t_end),
                        "aggregationInterval": {"of": "P1D"},
                        "resx": PIXEL_M, "resy": PIXEL_M, "evalscript": series_evalscript(p)},
        "calculations": {"w": {"statistics": {"default": {"percentiles": {"k": [50]}}}}},
    }
    headers = {"Authorization": f"Bearer {core.cdse_token()}", "Content-Type": "application/json"}
    r = _post_with_retry(analytics.CDSE_STATS_URL, payload, headers, timeout=240,
                         refresh_token_fn=analytics._get_token)
    if r.status_code != 200:
        raise RuntimeError(f"Sentinel Hub Statistical API błąd {r.status_code}: {r.text[:300]}")
    return parse_series(r.json(), n_px)


SPIKE_DELTA = 0.35       # odstępstwo od mediany sąsiednich scen
SPIKE_WINDOW_DAYS = 12   # okno sąsiadów po obu stronach


def despike(rows: list) -> list:
    """Odrzuca pojedyncze skoki W (niezamaskowany cień chmury albo mgiełka
    zmienia całe pole na jeden dzień): punkt, który odstaje od mediany
    sąsiadów z ±SPIKE_WINDOW_DAYS po OBU stronach o więcej niż SPIKE_DELTA.
    Prawdziwa zmiana (deszcz, nawodnienie) utrzymuje się w kolejnych scenach
    po jednej stronie, więc nie jest wycinana."""
    import statistics
    rows = sorted(rows, key=lambda r: r["obs_date"])
    ds = [date.fromisoformat(r["obs_date"]) for r in rows]
    out = []
    for i, r in enumerate(rows):
        before = [rows[j]["w"] for j in range(len(rows)) if 0 < (ds[i] - ds[j]).days <= SPIKE_WINDOW_DAYS]
        after = [rows[j]["w"] for j in range(len(rows)) if 0 < (ds[j] - ds[i]).days <= SPIKE_WINDOW_DAYS]
        if before and after:
            ref = statistics.median(before + after)
            side_b, side_a = statistics.median(before), statistics.median(after)
            if abs(r["w"] - ref) > SPIKE_DELTA and abs(r["w"] - side_b) > SPIKE_DELTA \
                    and abs(r["w"] - side_a) > SPIKE_DELTA:
                continue
        out.append(r)
    return out


def _median(stats: dict):
    """p50 z bloku percentiles (Statistical API), a gdy go brak — średnia."""
    pct = stats.get("percentiles") or {}
    for k in ("50.0", "50", 50.0):
        if pct.get(k) is not None:
            return pct[k]
    return stats["mean"]


def parse_series(data: dict, n_px) -> list:
    from datetime import datetime
    from sh_utils import valid_fraction
    # Uwaga: liczymy z pasma w, ale maskę bierzemy ze statystyk w — piksele
    # odrzucone w evalscripcie mają dataMask=0 we wszystkich wyjściach.
    out = []
    for item in data.get("data", []):
        o = item.get("outputs", {})
        try:
            st_w = o["w"]["bands"]["B0"]["stats"]
            st_n = o["ndvi"]["bands"]["B0"]["stats"]
            vf = valid_fraction(st_w, n_px)
            w = float(_median(st_w)) / 10000.0
            nd = float(st_n["mean"]) / 10000.0
        except (KeyError, TypeError, ValueError):
            continue
        if vf is None or vf < MIN_SERIES_VALID_FRAC or not (math.isfinite(w) and math.isfinite(nd)):
            continue
        d = datetime.fromisoformat(item["interval"]["from"].replace("Z", "+00:00")).date()
        out.append({"obs_date": d.isoformat(), "w": round(w, 4), "ndvi": round(nd, 4),
                    "valid_frac": round(vf, 3)})
    return out


# ---------------------------------------------------------------------------
# Operacje na polu (z zapisem do bazy)
# ---------------------------------------------------------------------------

def _field_geom(field: dict):
    from shapely import wkt as shapely_wkt
    import water_stress
    geom = shapely_wkt.loads(field["geometry_wkt"])
    g_buf = water_stress.inner_buffer_utm(geom, 20.0)
    return geom, g_buf


def calibrate_field(field: dict, date_from: Optional[date] = None, date_to: Optional[date] = None,
                    buffer_m: float = REGION_BUFFER_M) -> dict:
    """Kalibruje trapez dla regionu wokół pola, zapisuje parametry i od razu
    liczy serię W pola za ten sam okres. Zwraca zapisane parametry."""
    import db
    if not field.get("geometry_wkt"):
        raise ValueError("Pole nie ma geometrii.")
    date_to = date_to or date.today()
    date_from = date_from or (date_to - timedelta(days=CALIB_DAYS_DEFAULT))
    if (date_to - date_from).days < 60:
        raise ValueError("Okres kalibracji musi mieć co najmniej 60 dni (najlepiej 12 miesięcy).")
    geom, g_buf = _field_geom(field)
    region = calibration_region(geom, buffer_m)
    arr = fetch_stack(region, date_from, date_to)
    ndvi, strv, n_scenes = stack_to_points(arr)
    p = fit_trapezoid(ndvi, strv)
    quality, warnings = assess_quality(p, n_scenes)
    row = {**{k: p[k] for k in ("i_d", "s_d", "i_w", "s_w", "r2_d", "r2_w", "ndvi_lo", "ndvi_hi", "n_px")},
           "n_scenes": n_scenes, "date_from": date_from.isoformat(), "date_to": date_to.isoformat(),
           "region_wkt": region.wkt, "quality": quality,
           "warnings_json": json.dumps(warnings, ensure_ascii=False),
           "bins_json": json.dumps(p["bins"]), "scatter_json": json.dumps(scatter_sample(ndvi, strv))}
    db.save_optram_params(field["id"], row)
    db.set_setting(_refresh_marker(field["id"]), date.today().isoformat())
    db.clear_optram_series(field["id"])
    series = despike(fetch_series(g_buf, date_from, date_to, row))
    db.upsert_optram_series(field["id"], series)
    log.info(f"OPTRAM pole {field['id']}: {n_scenes} scen, {p['n_px']} px, jakość {quality}, "
             f"seria {len(series)} dat")
    return get_field_optram(field["id"])


REFRESH_MIN_INTERVAL_DAYS = 2   # scheduler co 48 h + „Sprawdź teraz” nie płacą dwa razy
REFRESH_MAX_DAYS = 30


def _refresh_marker(field_id: int) -> str:
    return f"optram_refresh:{field_id}"


def update_recent(field: dict, days: Optional[int] = None, force: bool = False) -> int:
    """Dopisuje świeże W (scheduler, „Sprawdź teraz”). 0, gdy pole nie ma
    kalibracji, jest ona bezużyteczna albo ostatnia próba była niedawno.
    Okno: od ostatniej zapisanej sceny minus SPIKE_WINDOW_DAYS (sąsiedzi do
    odrzucania skoków), najwyżej REFRESH_MAX_DAYS — audyt 11.2026: wcześniej
    zawsze 30 dni przy każdym przebiegu."""
    import db
    p = db.get_optram_params(field["id"])
    if not p or p.get("quality") == "poor":
        return 0
    end = date.today()
    last_try = db.get_setting(_refresh_marker(field["id"]))
    if not force and last_try and (end - date.fromisoformat(last_try[:10])).days < REFRESH_MIN_INTERVAL_DAYS:
        return 0
    if days is None:
        series = db.get_optram_series(field["id"])
        days = REFRESH_MAX_DAYS if not series else min(REFRESH_MAX_DAYS, max(
            SPIKE_WINDOW_DAYS + 2, (end - date.fromisoformat(series[-1]["obs_date"])).days + SPIKE_WINDOW_DAYS))
    _, g_buf = _field_geom(field)
    db.set_setting(_refresh_marker(field["id"]), end.isoformat())
    # despike na połączonej historii (nowa scena musi mieć sąsiadów z obu stron,
    # więc najświeższa wartość jest zapisywana, a weryfikowana przy kolejnym przebiegu)
    rows = fetch_series(g_buf, end - timedelta(days=days), end, p)
    merged = {r["obs_date"]: r for r in db.get_optram_series(field["id"])}
    merged.update({r["obs_date"]: r for r in rows})
    keep = {r["obs_date"] for r in despike(list(merged.values()))}
    db.clear_optram_series(field["id"])
    db.upsert_optram_series(field["id"], [r for d, r in sorted(merged.items()) if d in keep])
    return len([r for r in rows if r["obs_date"] in keep])


def era5_comparison(water_field_id: int, series: list) -> dict:
    """Zgodność W z wilgotnością ERA5-Land 0–7 cm z water_observations
    (te same daty) — szybki test wiarygodności kalibracji."""
    import db
    sm = {o["obs_date"]: o.get("sm_0_7") for o in db.get_water_observations(water_field_id)
          if o.get("sm_0_7") is not None}
    pairs = [(s["w"], sm[s["obs_date"]]) for s in series if s["obs_date"] in sm]
    r = pearson([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 6 else None
    return {"n": len(pairs), "r": None if r is None else round(r, 2),
            "points": [{"date": s["obs_date"], "sm_0_7": sm[s["obs_date"]]}
                       for s in series if s["obs_date"] in sm]}


def get_field_optram(water_field_id: int) -> dict:
    import db
    p = db.get_optram_params(water_field_id)
    if not p:
        return {"calibrated": False}
    series = db.get_optram_series(water_field_id)
    vals = [s["w"] for s in series]
    latest = series[-1] if series else None
    cur = None
    if latest:
        cur = {**latest, "class": w_class(latest["w"]),
               "season_pct": None if len(vals) < 8 else round(percentile_of(latest["w"], vals), 0),
               "dense_canopy": latest["ndvi"] >= DENSE_CANOPY_NDVI}
    return {
        "calibrated": True,
        "params": {k: p[k] for k in ("i_d", "s_d", "i_w", "s_w", "r2_d", "r2_w", "ndvi_lo", "ndvi_hi",
                                     "n_px", "n_scenes", "date_from", "date_to", "quality", "created_at")},
        "warnings": json.loads(p.get("warnings_json") or "[]"),
        "bins": json.loads(p.get("bins_json") or "[]"),
        "scatter": json.loads(p.get("scatter_json") or "[]"),
        "series": series, "current": cur,
        "era5": era5_comparison(water_field_id, series),
        "message_pl": optram_sentence(water_field_id, series=series, params=p).strip(),
        "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
    }


def optram_sentence(water_field_id: int, today: Optional[date] = None, series=None, params=None) -> str:
    """Zdanie do werdyktu modułu wodnego (pusty napis, gdy brak świeżego
    odczytu albo kalibracja jest bezużyteczna)."""
    import db
    today = today or date.today()
    p = params or db.get_optram_params(water_field_id)
    if not p or p.get("quality") == "poor":
        return ""
    series = series if series is not None else db.get_optram_series(water_field_id)
    if not series:
        return ""
    last = series[-1]
    if (today - date.fromisoformat(last["obs_date"])).days > SENTENCE_MAX_AGE_DAYS:
        return ""
    era = era5_comparison(water_field_id, series)
    if era["n"] >= ERA5_MIN_N and era["r"] is not None and era["r"] < ERA5_MIN_R:
        # Test na żywo (pole deszczowe w PL): r = -0,43 — wynik nie trafia do
        # werdyktu, zostaje tylko w panelu z ostrzeżeniem o słabej zgodności.
        return ""
    msg = (f" Wilgotność wierzchniej warstwy z satelity (OPTRAM, {last['obs_date']}): "
           f"{last['w']:.2f} w skali 0–1 — {w_class(last['w'])}")
    vals = [s["w"] for s in series]
    if len(vals) >= 8:
        msg += f"; {percentile_of(last['w'], vals):.0f}. percentyl wartości tego pola od kalibracji"
    msg += "."
    if last["ndvi"] >= DENSE_CANOPY_NDVI:
        msg += " Przy zwartym łanie wskaźnik pokazuje raczej uwodnienie roślin niż gleby."
    elif last["ndvi"] < BARE_SOIL_NDVI:
        msg += " Na gołej glebie wynik zależy też od barwy i uprawy gleby."
    if p.get("quality") == "limited":
        msg += " (kalibracja trapezu o ograniczonej jakości)"
    return msg


def attach_optram(field: dict, verdict: dict) -> dict:
    """Dokleja zdanie OPTRAM do werdyktu (nieblokujące, bez zapytań do API)."""
    try:
        s = optram_sentence(field["id"])
    except Exception as e:
        log.warning("OPTRAM niedostępny dla pola id=%s: %s", field.get("id"), e)
        return verdict
    if s:
        verdict["message_pl"] = (verdict.get("message_pl") or "") + s
        verdict["optram_sentence"] = s.strip()
    return verdict
