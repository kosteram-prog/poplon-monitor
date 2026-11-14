"""Mapy stref plonu metodą MVI (Moisture-sensitive Vegetation Index) dla LENs.

Idea (Pourmansouri i in., Agricultural Water Management 2026 — metoda z
infografiki): plon zgłoszony dla CAŁEGO pola rozkładamy na piksele 10 m
proporcjonalnie do wskaźnika łączącego wigor i wilgotność łanu w sezonie:

    EVI2_med  = mediana EVI2 ze scen sezonu         (wigor, odporna na szum)
    NDMI_p75  = 75. percentyl NDMI ze scen sezonu   (wilgotność łanu w dobrych warunkach)
    MVI       = EVI2_med × NDMI_p75,  MVI+ = max(MVI, 0)
    Y_i       = MVI+_i / mean(MVI+) × Y_zgłoszony   (średnia mapy = plon zgłoszony)

Różnice wobec artykułu: Sentinel-2 (10 m) zamiast Landsat (30 m), plon
zgłoszony przez rolnika dla pola zamiast statystyki powiatowej. MVI
walidowano na kukurydzy i soi w USA — dla polskich upraw traktuj mapę jako
WZGLĘDNĄ (gdzie lepiej/gorzej w polu) do czasu porównania z mapą z kombajnu.

Wynik: GeoTIFF plonu (t/ha), strefy jako GeoJSON (klasy względem średniej
pola), nakładka PNG do mapy, statystyki stref. Pliki w uploads/yield_maps/<id>/.
"""
import io
import json
import logging
import math
import os
from datetime import date
from typing import Optional

log = logging.getLogger("yield_mvi")

CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
YIELD_MAPS_DIR = os.path.join("uploads", "yield_maps")
PIXEL_M = 10.0
MIN_VALID_SCENES = 3            # jak w MVI: min. 3 czyste obserwacje na piksel
MIN_VALID_FRAC = 0.5            # min. udział pikseli pola z ważnym MVI
MIN_MEAN_MVI = 0.02             # poniżej — brak wigoru w okresie (zły okres / brak uprawy)
SIEVE_MIN_PX = 10               # strefy < 0.1 ha doklejane do sąsiednich
HIGH_CV_PCT = 30.0              # powyżej — ostrzeżenie: mapa prawdopodobnie przesadza z różnicami
MAX_FIELD_EXTENT_KM = 5.0
EDGE_BUFFER_M = 10.0            # pas brzegowy pomijany (1 piksel S2)

# Klasy stref względem średniego plonu pola (Y_i / Y_zgłoszony).
ZONE_BREAKS = [0.90, 0.97, 1.03, 1.10]
ZONE_LABELS = ["< 90%", "90–97%", "97–103%", "103–110%", "> 110%"]
ZONE_COLORS = ["#C0392B", "#E67E22", "#F4D03F", "#7FB241", "#1E7B34"]

# Domyślne okna sezonu (miesiąc, dzień) — okres, w którym łan najlepiej
# różnicuje plon; do nadpisania datami w żądaniu.
CROP_WINDOWS = {
    "kukurydza": ((6, 1), (9, 15)), "maize": ((6, 1), (9, 15)), "corn": ((6, 1), (9, 15)),
    "soja": ((6, 15), (9, 15)), "soybean": ((6, 15), (9, 15)),
    "pszenica": ((4, 15), (6, 30)), "wheat": ((4, 15), (6, 30)),
    "jęczmień": ((4, 15), (6, 20)), "jeczmien": ((4, 15), (6, 20)), "barley": ((4, 15), (6, 20)),
    "pszenżyto": ((4, 15), (6, 30)), "żyto": ((4, 10), (6, 20)),
    "rzepak": ((4, 1), (6, 10)), "rapeseed": ((4, 1), (6, 10)),
    "burak": ((7, 1), (9, 30)), "burak cukrowy": ((7, 1), (9, 30)), "sugar beet": ((7, 1), (9, 30)),
    "ziemniak": ((6, 15), (8, 31)), "ziemniaki": ((6, 15), (8, 31)), "potato": ((6, 15), (8, 31)),
    "bawełna": ((6, 15), (9, 15)), "bawelna": ((6, 15), (9, 15)), "cotton": ((6, 15), (9, 15)),
}
DEFAULT_WINDOW = ((6, 1), (9, 15))   # jak w artykule (czerwiec–wrzesień)

EVALSCRIPT_MVI = """
//VERSION=3
// MVI per piksel: mediana EVI2 i 75. percentyl NDMI z czystych scen okresu.
// Wyjście FLOAT32: [EVI2_med, NDMI_p75, MVI, liczba_scen]; NaN, gdy < 3 scen.
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "B8A", "B11", "SCL", "dataMask"] }],
    output: { id: "default", bands: 4, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}
function q(sorted, p) {
  const pos = (sorted.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
const BAD = [0, 1, 3, 8, 9, 10, 11];
function evaluatePixel(samples) {
  const evi2 = [], ndmi = [];
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.dataMask !== 1 || BAD.indexOf(s.SCL) !== -1) continue;
    evi2.push(2.5 * (s.B08 - s.B04) / (s.B08 + 2.4 * s.B04 + 1.0));
    ndmi.push((s.B8A - s.B11) / (s.B8A + s.B11 + 1e-9));
  }
  if (evi2.length < 3) return [NaN, NaN, NaN, evi2.length];
  evi2.sort((a, b) => a - b);
  ndmi.sort((a, b) => a - b);
  const e = q(evi2, 0.5), n = q(ndmi, 0.75);
  return [e, n, e * n, evi2.length];
}
"""


def crop_window(crop: Optional[str], year: int) -> tuple:
    key = (crop or "").strip().lower()
    (m1, d1), (m2, d2) = CROP_WINDOWS.get(key, DEFAULT_WINDOW)
    return date(year, m1, d1), date(year, m2, d2)


def _resolution_deg(lat: float) -> tuple:
    """(resx, resy) w stopniach odpowiadające ~10 m w danym miejscu."""
    resy = PIXEL_M / 111320.0
    resx = PIXEL_M / (111320.0 * max(0.2, math.cos(math.radians(lat))))
    return resx, resy


def fetch_mvi_raster(geom_4326, t_start: date, t_end: date):
    """Process API -> (tablica 4×H×W float32, transform rasterio, bounds).
    Piksele poza geometrią pola = NaN (dataMask = 0)."""
    import core
    import rasterio
    from shapely.geometry import mapping
    from sentinel1 import _post_with_retry
    import analytics

    resx, resy = _resolution_deg(geom_4326.centroid.y)
    payload = {
        "input": {
            "bounds": {"geometry": mapping(geom_4326),
                       "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{t_start.isoformat()}T00:00:00Z",
                                                   "to": f"{t_end.isoformat()}T23:59:59Z"},
                                     "maxCloudCoverage": 80}}],
        },
        "output": {"resx": resx, "resy": resy,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT_MVI,
    }
    headers = {"Authorization": f"Bearer {core.cdse_token()}", "Content-Type": "application/json",
               "Accept": "image/tiff"}
    r = _post_with_retry(CDSE_PROCESS_URL, payload, headers, timeout=240,
                         refresh_token_fn=analytics._get_token)
    if r.status_code != 200:
        raise RuntimeError(f"Sentinel Hub Process API błąd {r.status_code}: {r.text[:300]}")
    with rasterio.MemoryFile(r.content) as mem, mem.open() as ds:
        arr = ds.read().astype("float32")
        return arr, ds.transform, ds.bounds


def field_mask(geom_4326, shape, transform):
    """True dla pikseli wewnątrz obrysu pola."""
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    return ~geometry_mask([mapping(geom_4326)], out_shape=shape, transform=transform)


def disaggregate(arr, reported_yield: float, inside=None) -> dict:
    """Z rastra [EVI2_med, NDMI_p75, MVI, n] liczy plon per piksel i klasy
    stref. `inside` — maska pikseli pola (z obrysu); bez niej: piksele, dla
    których Process API zwrócił jakąkolwiek scenę. Bez I/O."""
    import numpy as np
    evi2, ndmi, mvi, nscenes = arr[0], arr[1], arr[2], arr[3]
    if inside is None:
        inside = np.isfinite(nscenes) & (nscenes > 0)
    valid = inside & np.isfinite(mvi) & (nscenes >= MIN_VALID_SCENES)
    n_inside = int(np.count_nonzero(inside))
    n_valid = int(np.count_nonzero(valid))
    if n_inside == 0 or n_valid / n_inside < MIN_VALID_FRAC:
        raise ValueError(f"Za mało pikseli z co najmniej {MIN_VALID_SCENES} czystymi scenami "
                         f"({n_valid} z {n_inside}). Wydłuż okres albo sprawdź zachmurzenie.")
    mvi_pos = np.where(valid, np.maximum(mvi, 0.0), np.nan)
    mean_mvi = float(np.nanmean(mvi_pos))
    if mean_mvi < MIN_MEAN_MVI:
        raise ValueError(f"Średni MVI pola jest bardzo niski ({mean_mvi:.3f}) — w wybranym okresie "
                         f"nie widać rozwiniętego łanu. Sprawdź uprawę i daty sezonu.")
    yld = mvi_pos / mean_mvi * float(reported_yield)
    rel = yld / float(reported_yield)
    zones = np.full(rel.shape, 0, dtype="uint8")
    zones[valid] = (np.digitize(rel[valid], ZONE_BREAKS) + 1).astype("uint8")   # 1..5
    zones = _sieve(zones, valid)
    px_ha = (PIXEL_M * PIXEL_M) / 10000.0
    zone_stats = []
    for k in range(1, len(ZONE_LABELS) + 1):
        m = zones == k
        cnt = int(np.count_nonzero(m))
        zone_stats.append({"zone": k, "label": ZONE_LABELS[k - 1], "color": ZONE_COLORS[k - 1],
                           "area_ha": round(cnt * px_ha, 2),
                           "share_pct": round(100.0 * cnt / n_valid, 1),
                           "mean_yield_t_ha": round(float(np.nanmean(yld[m])), 2) if cnt else None})
    vals = yld[valid]
    stats = {"reported_yield_t_ha": float(reported_yield),
             "map_mean_yield_t_ha": round(float(np.nanmean(vals)), 3),
             "p10_t_ha": round(float(np.nanpercentile(vals, 10)), 2),
             "p90_t_ha": round(float(np.nanpercentile(vals, 90)), 2),
             "cv_pct": round(100.0 * float(np.nanstd(vals) / np.nanmean(vals)), 1),
             "mean_mvi": round(mean_mvi, 4),
             "mean_evi2_med": round(float(np.nanmean(evi2[valid])), 3),
             "mean_ndmi_p75": round(float(np.nanmean(ndmi[valid])), 3),
             "median_scenes": int(np.nanmedian(nscenes[valid])),
             "valid_pixels": n_valid, "field_pixels": n_inside,
             "area_ha": round(n_valid * px_ha, 2), "zones": zone_stats}
    # Liniowe skalowanie Y ~ MVI przenosi CAŁĄ względną zmienność indeksu na
    # plon. Mapy z kombajnu mają zwykle CV ok. 10-25% — wyraźnie większy
    # rozrzut znaczy, że mapa przesadza z różnicami (wartości t/ha w strefach
    # traktować orientacyjnie, kolejność stref — jako wiarygodną).
    stats["high_variability"] = stats["cv_pct"] > HIGH_CV_PCT
    return {"yield": yld, "zones": zones, "valid": valid, "stats": stats}


def _sieve(zones, valid):
    """Usuwa drobne plamki stref (< SIEVE_MIN_PX) — doklejane do sąsiadów."""
    try:
        from rasterio.features import sieve
        import numpy as np
        out = sieve(zones, size=SIEVE_MIN_PX, mask=valid)
        return np.where(valid, out, 0).astype("uint8")
    except Exception as e:      # bez sita mapa i tak jest poprawna, tylko bardziej "ziarnista"
        log.warning("Sito stref niedostępne: %s", e)
        return zones


def zones_geojson(zones, valid, transform) -> dict:
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    feats = []
    for geom, val in shapes(zones, mask=valid, transform=transform):
        k = int(val)
        if k < 1:
            continue
        g = shape(geom)
        feats.append({"type": "Feature", "geometry": mapping(g),
                      "properties": {"zone": k, "label": ZONE_LABELS[k - 1], "color": ZONE_COLORS[k - 1]}})
    return {"type": "FeatureCollection", "features": feats}


def overlay_png(yld, valid, reported_yield: float) -> bytes:
    """Nakładka do mapy: kolor wg plonu względem średniej (0.7–1.3), przezroczyste tło."""
    import numpy as np
    from PIL import Image
    rel = np.nan_to_num(np.clip((yld / reported_yield - 0.7) / 0.6, 0, 1), nan=0.5)
    stops = [(0.0, (192, 57, 43)), (0.5, (244, 208, 63)), (1.0, (30, 123, 52))]
    rgba = np.zeros(yld.shape + (4,), dtype="uint8")
    for c in range(3):
        rgba[..., c] = np.interp(rel, [s[0] for s in stops], [s[1][c] for s in stops]).astype("uint8")
    rgba[..., 3] = np.where(valid, 210, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def write_geotiff(path: str, yld, transform):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=yld.shape[0], width=yld.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=float("nan"),
                       compress="deflate") as ds:
        ds.write(yld.astype("float32"), 1)
        ds.set_band_description(1, "plon_t_ha (MVI)")


def build_yield_map(geom_4326, reported_yield: float, t_start: date, t_end: date,
                    map_id: int) -> dict:
    """Pełny przebieg: raster MVI -> plon -> strefy -> pliki. Zwraca statystyki
    i ścieżki plików (względne do katalogu aplikacji)."""
    import water_geometry as wg
    wg.check_field_extent(geom_4326)
    arr, transform, bounds = fetch_mvi_raster(geom_4326, t_start, t_end)
    # Pas brzegowy (1 piksel) pomijany: piksele na granicy mieszają się z drogą,
    # miedzą i sąsiednią uprawą — test na żywo (28 ha) dał czerwoną obwódkę
    # wokół całego pola, zaniżoną średnią MVI wnętrza i sztucznie dużą strefę
    # „< 90%”. Małe pola (po buforze < 50% powierzchni) — bez bufora.
    import water_stress
    inner = water_stress.inner_buffer_utm(geom_4326, EDGE_BUFFER_M)
    inside = field_mask(inner, arr.shape[1:], transform)
    full = field_mask(geom_4326, arr.shape[1:], transform)
    if inside.sum() < 0.5 * full.sum():
        inside = full
    res = disaggregate(arr, reported_yield, inside=inside)
    out_dir = os.path.join(YIELD_MAPS_DIR, str(map_id))
    os.makedirs(out_dir, exist_ok=True)
    write_geotiff(os.path.join(out_dir, "plon_mvi.tif"), res["yield"], transform)
    gj = zones_geojson(res["zones"], res["valid"], transform)
    with open(os.path.join(out_dir, "strefy.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f)
    with open(os.path.join(out_dir, "nakladka.png"), "wb") as f:
        f.write(overlay_png(res["yield"], res["valid"], reported_yield))
    stats = res["stats"]
    stats["bounds"] = [bounds.left, bounds.bottom, bounds.right, bounds.top]
    # area_ha = analizowane wnętrze (bez pasa brzegowego), field_area_ha = całe pole
    stats["field_area_ha"] = wg.area_ha(geom_4326)
    return stats
