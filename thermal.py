"""Termika z Landsata 8/9 dla modułu wodnego (głównie pola nawadniane w Uzbekistanie).

Roślina z dostępem do wody chłodzi się transpiracją; gdy wody brakuje,
zamyka aparaty szparkowe i nagrzewa się — często kilka dni, zanim spadek
widać w NDVI/NDMI. Temperaturę powierzchni daje kanał termiczny Landsata
(TIRS B10, 10,9 µm, natywnie 100 m, udostępniany w siatce 30 m, nalot
~10:30–11:00 czasu lokalnego, L8+L9 razem co ~8 dni).

Na Copernicus Data Space dostępny jest tylko Landsat 8–9 Level-1
(`landsat-ot-l1`, od 2021): B10 to temperatura jasnościowa na górze
atmosfery, nie temperatura powierzchni. Dlatego:

1. Temperaturę powierzchni (LST) liczymy uproszczenie: poprawka na
   emisyjność z NDVI (metoda progowa Sobrino), bez korekcji atmosferycznej:
       LST = BT / (1 + (λ·BT/ρ)·ln ε),  λ = 10,895 µm, ρ = 14388 µm·K.
   Wartość bezwzględna może być zaniżona o kilka stopni (wilgotne powietrze
   = większy błąd) — pokazujemy ją jako przybliżoną.
2. Do decyzji używamy wskaźnika WZGLĘDNEGO w obrębie tej samej sceny i
   kwadratu ~3 km wokół pola (atmosfera praktycznie jednakowa, błąd
   bezwzględny się znosi):
       T_cold = 5. percentyl LST gęstej roślinności (NDVI ≥ 0,6) w okolicy
                — dobrze uwodniona, transpirująca roślinność,
       T_hot  = 95. percentyl LST gołej/suchej powierzchni (NDVI ≤ 0,2),
       TSI    = (T_pole − T_cold) / (T_hot − T_cold), przycięte do 0–1
   — indeks stresu termicznego w duchu CWSI: 0 = pole tak chłodne jak
   najlepiej uwodniona roślinność, 1 = nagrzane jak sucha gleba.
3. Rozrzut temperatur w polu (p90 − p10) przy dobrej okrywie podpowiada
   nierówne nawodnienie (tylko duże pola — piksel termiczny ma 100 m).

Progi (0,25 / 0,5, 3 K) są robocze i niekalibrowane — to kontekst do
werdyktu, bez osobnych alarmów.
"""
import io
import json
import logging
import math
import tarfile
from datetime import date, timedelta
from typing import Optional

log = logging.getLogger("thermal")

COLLECTION = "landsat-ot-l1"
DATA_START = date(2021, 1, 1)           # archiwum Landsat na CDSE
PIXEL_M = 30.0
FIELD_BUFFER_M = 45.0                   # pół piksela termicznego (100 m) do środka pola
REGION_BUFFER_M = 1500.0
MAX_SCENES = 60
MIN_FIELD_PX = 8
MIN_VALID_FRAC = 0.6
MIN_REF_PX = 30
MIN_REF_SPAN_K = 3.0
DENSE_NDVI = 0.6
BARE_NDVI = 0.2
LOW_COVER_NDVI = 0.3
SPREAD_NDVI = 0.4
SPREAD_ALERT_K = 3.0
SMALL_FIELD_HA = 3.0
SENTENCE_MAX_AGE_DAYS = 16
TSI_MID = 0.25
TSI_HIGH = 0.5

EVALSCRIPT = """
//VERSION=3
// Landsat 8/9 L1: LST (K, poprawka emisyjności z NDVI, bez korekcji
// atmosfery) i NDVI dla każdej sceny; NaN = chmura, cień, cirrus, śnieg,
// woda albo brak danych. W userData daty scen.
const MAX_SCENES = __MAX_SCENES__;
function setup() {
  return {
    input: [{ bands: ["B04", "B05", "B10", "BQA", "dataMask"] }],
    output: { id: "default", bands: 2, sampleType: "FLOAT32" },
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
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  const orbits = scenes.orbits || scenes;
  outputMetadata.userData = { dates: orbits.map(o => o.dateFrom || o.date) };
}
function emissivity(ndvi) {
  if (ndvi < 0.2) return 0.97;
  if (ndvi > 0.5) return 0.99;
  const pv = Math.pow((ndvi - 0.2) / 0.3, 2);
  return 0.004 * pv + 0.986;
}
function evaluatePixel(samples) {
  const out = new Array(2 * Math.max(1, samples.length)).fill(NaN);
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.dataMask !== 1 || !(s.B10 > 200)) continue;
    const q = decodeL8C2Qa(s.BQA);
    if (q.fill || !q.clear || q.cloud || q.dilatedCloud || q.cirrus || q.cloudShadow || q.snow || q.water
        || q.cloudConfidence >= 2) continue;
    const ndvi = (s.B05 - s.B04) / (s.B05 + s.B04 + 1e-9);
    const bt = s.B10;
    const lst = bt / (1 + (10.895 * bt / 14388) * Math.log(emissivity(ndvi)));
    out[2 * i] = lst;
    out[2 * i + 1] = ndvi;
  }
  return out;
}
"""


# ---------------------------------------------------------------------------
# Obliczenia (bez I/O)
# ---------------------------------------------------------------------------

def lst_from_bt(bt_k: float, ndvi: float) -> float:
    """To samo co w evalscripcie — do testów i dokumentacji."""
    if ndvi < 0.2:
        eps = 0.97
    elif ndvi > 0.5:
        eps = 0.99
    else:
        eps = 0.004 * ((ndvi - 0.2) / 0.3) ** 2 + 0.986
    return bt_k / (1 + (10.895 * bt_k / 14388.0) * math.log(eps))


def scene_metrics(lst_k, ndvi, inside) -> Optional[dict]:
    """Metryki jednej sceny: pole vs odniesienia z okolicy. None, gdy pole
    zachmurzone. Temperatury w °C."""
    import numpy as np
    lst_k = np.asarray(lst_k, float)
    ndvi = np.asarray(ndvi, float)
    inside = np.asarray(inside, bool)
    n_in = int(inside.sum())
    if n_in == 0:
        return None
    ok = np.isfinite(lst_k) & np.isfinite(ndvi)
    f = inside & ok
    nf = int(f.sum())
    if nf < MIN_FIELD_PX or nf / n_in < MIN_VALID_FRAC:
        return None
    t = lst_k[f] - 273.15
    out = {"t_field": round(float(np.median(t)), 2),
           "t_p10": round(float(np.percentile(t, 10)), 2),
           "t_p90": round(float(np.percentile(t, 90)), 2),
           "ndvi": round(float(np.mean(ndvi[f])), 3),
           "n_px": nf, "valid_frac": round(nf / n_in, 3),
           "t_cold": None, "t_hot": None, "tsi": None}
    reg = ok & ~inside
    cold = lst_k[reg & (ndvi >= DENSE_NDVI)]
    hot = lst_k[reg & (ndvi <= BARE_NDVI)]
    if cold.size >= MIN_REF_PX:
        out["t_cold"] = round(float(np.percentile(cold, 5)) - 273.15, 2)
    if hot.size >= MIN_REF_PX:
        out["t_hot"] = round(float(np.percentile(hot, 95)) - 273.15, 2)
    if out["t_cold"] is not None and out["t_hot"] is not None and \
            out["t_hot"] - out["t_cold"] >= MIN_REF_SPAN_K:
        tsi = (out["t_field"] - out["t_cold"]) / (out["t_hot"] - out["t_cold"])
        out["tsi"] = round(max(0.0, min(1.0, tsi)), 3)
    return out


def stack_metrics(arr, dates: list, inside) -> list:
    """Tablica (2·N, H, W) + daty scen -> [{scene_date, ...metryki}]."""
    import numpy as np
    arr = np.asarray(arr, float)
    n = arr.shape[0] // 2
    rows = []
    for k in range(n):
        if k >= len(dates) or not dates[k]:
            continue
        m = scene_metrics(arr[2 * k], arr[2 * k + 1], inside)
        if m:
            rows.append({"scene_date": str(dates[k])[:10], **m})
    rows.sort(key=lambda r: r["scene_date"])
    # ORBIT = jedna „scena” na dzień; gdyby dzień się powtórzył — ostatni wygrywa
    return list({r["scene_date"]: r for r in rows}.values())


def tsi_class(tsi: float) -> str:
    if tsi < TSI_MID:
        return "rośliny chłodzą się transpiracją, bez oznak stresu"
    if tsi < TSI_HIGH:
        return "pole cieplejsze od dobrze uwodnionej roślinności — możliwy początek stresu"
    return "pole nagrzane prawie jak sucha gleba — prawdopodobny stres wodny albo słaba okrywa"


def describe(row: dict) -> str:
    """Zdanie o jednej scenie (PL, do werdyktu)."""
    msg = f" Termika Landsat ({row['scene_date']}): pole ok. {row['t_field']:.0f} °C"
    if row.get("t_cold") is not None:
        msg += (f"; różnica do najchłodniejszej gęstej roślinności w okolicy: "
                f"{row['t_field'] - row['t_cold']:+.1f} K")
    low_cover = row.get("ndvi") is not None and row["ndvi"] < LOW_COVER_NDVI
    if row.get("tsi") is not None and not low_cover:
        msg += f"; indeks stresu termicznego {row['tsi']:.2f} — {tsi_class(row['tsi'])}"
    elif row.get("tsi") is not None:
        msg += f"; indeks stresu termicznego {row['tsi']:.2f}"
    elif row.get("t_cold") is None:
        msg += " (brak gęstej roślinności w okolicy do porównania)"
    msg += "."
    if low_cover:
        msg += " Słaba okrywa — temperatura i indeks odzwierciedlają głównie nagrzanie gleby, nie stan roślin."
    spread = (row.get("t_p90") or 0) - (row.get("t_p10") or 0)
    if row.get("ndvi", 0) >= SPREAD_NDVI and spread >= SPREAD_ALERT_K:
        msg += f" Różnica temperatur w polu {spread:.1f} K — możliwe nierówne nawodnienie."
    return msg


# ---------------------------------------------------------------------------
# Sentinel Hub
# ---------------------------------------------------------------------------

def _resolution_deg(lat: float) -> tuple:
    resy = PIXEL_M / 111320.0
    resx = PIXEL_M / (111320.0 * max(0.2, math.cos(math.radians(lat))))
    return resx, resy


def parse_tar(content: bytes) -> tuple:
    """Odpowiedź multipart (tar): default.tif + userdata.json ->
    (tablica, transform, daty)."""
    import rasterio
    tif, meta = None, {}
    with tarfile.open(fileobj=io.BytesIO(content)) as tar:
        for m in tar.getmembers():
            data = tar.extractfile(m).read() if m.isfile() else None
            if data is None:
                continue
            if m.name.endswith(".tif") or m.name.endswith(".tiff"):
                tif = data
            elif m.name.endswith(".json"):
                meta = json.loads(data.decode("utf-8"))
    if tif is None:
        raise RuntimeError("Odpowiedź Process API bez rastra.")
    with rasterio.MemoryFile(tif) as mem, mem.open() as ds:
        arr = ds.read()
        transform = ds.transform
    return arr, transform, list(meta.get("dates") or [])


def fetch_stack(region_4326, t_start: date, t_end: date) -> tuple:
    import analytics
    import core
    from shapely.geometry import mapping
    from sentinel1 import _post_with_retry
    from yield_mvi import CDSE_PROCESS_URL

    resx, resy = _resolution_deg(region_4326.centroid.y)
    payload = {
        "input": {
            "bounds": {"geometry": mapping(region_4326),
                       "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": COLLECTION,
                      "dataFilter": {"timeRange": {"from": f"{t_start.isoformat()}T00:00:00Z",
                                                   "to": f"{t_end.isoformat()}T23:59:59Z"},
                                     "maxCloudCoverage": 70}}],
        },
        "output": {"resx": resx, "resy": resy,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}},
                                 {"identifier": "userdata", "format": {"type": "application/json"}}]},
        "evalscript": EVALSCRIPT.replace("__MAX_SCENES__", str(MAX_SCENES)),
    }
    headers = {"Authorization": f"Bearer {core.cdse_token()}", "Content-Type": "application/json",
               "Accept": "application/tar"}
    r = _post_with_retry(CDSE_PROCESS_URL, payload, headers, timeout=300,
                         refresh_token_fn=analytics._get_token)
    if r.status_code != 200:
        raise RuntimeError(f"Sentinel Hub Process API (Landsat) błąd {r.status_code}: {r.text[:300]}")
    return parse_tar(r.content)


# ---------------------------------------------------------------------------
# Pole (z zapisem do bazy)
# ---------------------------------------------------------------------------

def _field_geoms(field: dict):
    from shapely import wkt as shapely_wkt
    import water_stress
    geom = shapely_wkt.loads(field["geometry_wkt"])
    return geom, water_stress.inner_buffer_utm(geom, FIELD_BUFFER_M)


def fetch_field(field: dict, date_from: date, date_to: date) -> list:
    """Pobiera sceny z okresu, liczy metryki i zapisuje. Zwraca nowe wiersze."""
    import db
    from optram import calibration_region
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping
    if not field.get("geometry_wkt"):
        raise ValueError("Pole nie ma geometrii.")
    date_from = max(date_from, DATA_START)
    if date_to <= date_from:
        raise ValueError("Pusty okres (dane Landsat na CDSE są od 2021-01-01).")
    geom, inner = _field_geoms(field)
    region = calibration_region(geom, REGION_BUFFER_M)
    arr, transform, dates = fetch_stack(region, date_from, date_to)
    inside = ~geometry_mask([mapping(inner)], out_shape=arr.shape[1:], transform=transform)
    # Małe pole (test na żywo: 3 ha -> 5 pikseli po buforze 45 m): bez bufora,
    # a w ostateczności każdy piksel dotykający pola. Panel i tak ostrzega
    # o polach < SMALL_FIELD_HA (piksel termiczny 100 m miesza się z otoczeniem).
    for opts in ((geom, False), (geom, True)):
        if int(inside.sum()) >= MIN_FIELD_PX:
            break
        inside = ~geometry_mask([mapping(opts[0])], out_shape=arr.shape[1:], transform=transform,
                                all_touched=opts[1])
    rows = stack_metrics(arr, dates, inside)
    db.upsert_thermal(field["id"], rows)
    db.set_setting(_refresh_marker(field["id"]), date.today().isoformat())
    log.info(f"Termika pole {field['id']}: {arr.shape[0] // 2} scen, {len(rows)} czystych nad polem")
    return rows


REFRESH_MIN_INTERVAL_DAYS = 4   # L8+L9 razem dają nową scenę co ~8 dni
REFRESH_MAX_DAYS = 40


def _refresh_marker(field_id: int) -> str:
    return f"thermal_refresh:{field_id}"


def update_recent(field: dict, days: Optional[int] = None, force: bool = False) -> int:
    """Scheduler / „Sprawdź teraz”: dociąga nowe sceny dla pól, które już mają
    termikę. Audyt 11.2026: wcześniej co 48 h zawsze 40 dni (te same sceny
    opłacane kilka razy) — teraz od ostatniej sceny i nie częściej niż co
    REFRESH_MIN_INTERVAL_DAYS."""
    import db
    series = db.get_thermal_series(field["id"])
    if not series:
        return 0
    end = date.today()
    last_try = db.get_setting(_refresh_marker(field["id"]))
    if not force and last_try and (end - date.fromisoformat(last_try[:10])).days < REFRESH_MIN_INTERVAL_DAYS:
        return 0
    if days is None:
        days = min(REFRESH_MAX_DAYS, max(16, (end - date.fromisoformat(series[-1]["scene_date"])).days + 3))
    return len(fetch_field(field, end - timedelta(days=days), end))


def field_area_ha(field: dict) -> Optional[float]:
    try:
        import water_geometry
        from shapely import wkt as shapely_wkt
        return round(water_geometry.area_ha(shapely_wkt.loads(field["geometry_wkt"])), 2)
    except Exception:
        return None


def get_field_thermal(field: dict) -> dict:
    import db
    series = db.get_thermal_series(field["id"])
    area = field_area_ha(field)
    out = {"has_data": bool(series), "series": series, "area_ha": area,
           "small_field": area is not None and area < SMALL_FIELD_HA,
           "calibration_status": "niekalibrowane / wymaga walidacji terenowej",
           "source": "Landsat 8/9 Collection 2 Level-1 (USGS/NASA) przez Copernicus Data Space"}
    if series:
        last = series[-1]
        out["latest"] = last
        out["message_pl"] = describe(last).strip()
        tsis = [r["tsi"] for r in series if r.get("tsi") is not None]
        out["tsi_mean"] = round(sum(tsis) / len(tsis), 3) if tsis else None
    return out


def thermal_sentence(field: dict, today: Optional[date] = None) -> tuple:
    """(zdanie, ostatni wiersz) albo ('', None), gdy brak świeżej sceny."""
    import db
    today = today or date.today()
    series = db.get_thermal_series(field["id"])
    if not series:
        return "", None
    last = series[-1]
    if (today - date.fromisoformat(last["scene_date"])).days > SENTENCE_MAX_AGE_DAYS:
        return "", None
    return describe(last), last


def attach_thermal(field: dict, verdict: dict) -> dict:
    """Dokleja zdanie o termice i — gdy satelita zgłosił ryzyko — czy termika
    je potwierdza. Nieblokujące, bez zapytań do API."""
    try:
        s, last = thermal_sentence(field)
    except Exception as e:
        log.warning("Termika niedostępna dla pola id=%s: %s", field.get("id"), e)
        return verdict
    if not s:
        return verdict
    tsi = last.get("tsi")
    if verdict.get("risk_flag") and tsi is not None and last.get("ndvi", 0) >= LOW_COVER_NDVI:
        if tsi >= TSI_HIGH:
            s += " Termika potwierdza sygnał stresu."
        elif tsi < TSI_MID:
            s += " Termika nie pokazuje stresu — rozważ inne przyczyny spadku."
    verdict["message_pl"] = (verdict.get("message_pl") or "") + s
    verdict["thermal"] = last
    return verdict
