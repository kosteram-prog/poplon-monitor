"""Wieloletnie strefy plonu + nawożenie zmienną dawką (rozszerzenie yield_mvi).

Wejście: płodozmian z plonami z kilku lat (np. 5) dla jednego pola. Dla
każdego roku liczymy WZGLĘDNĄ mapę plonu (rel = MVI+ / średnia MVI pola,
średnia pola = 1.0) w oknie sezonu danej uprawy. Mapy względne są
porównywalne między uprawami (kukurydza 10 t/ha i rzepak 3,5 t/ha), więc
strefy z wielolecia liczymy jako ŚREDNIĄ map względnych — piksel stale
słabszy przez 5 lat trafia do strefy niskiej niezależnie od uprawy.

Dawki nawozu: operator podaje dawkę ŚREDNIĄ [kg/ha] dla pola; rozdzielamy
ją na strefy jedną z dwóch strategii:
  - "potencjal"   — dawka proporcjonalna do potencjału strefy (więcej tam,
                    gdzie plon większy — nawożenie pod wynos składników),
  - "wyrownawcza" — odwrotnie (więcej w strefach słabych — odbudowa
                    zasobności); dawka = D × (2 − rel).
W obu strategiach średnia ważona powierzchnią stref = dawka podana
(zachowanie masy, jak w yield_mvi), potem zaokrąglenie do pełnych kg
(krok `rounding`, domyślnie 5 kg) i opcjonalne widełki min/max.

Próby glebowe: CSV z punktami (współrzędne + pH, P2O5, K2O, Mg, próchnica).
Punkty przypisujemy do stref (po rastrze stref) i liczymy średnie per
strefa — do raportu; dawek NIE korygujemy automatycznie o zasobność
(decyzja agronoma).

Plik dla siewnika/rozsiewacza: strefy z dawką jako Shapefile (ZIP, WGS84,
atrybut RATE) — format przyjmowany przez większość terminali (Trimble,
Topcon, Müller, John Deere przez konwersję) — oraz GeoJSON.

Pliki w uploads/yield_analyses/<id>/.
"""
import io
import json
import logging
import os
import re
import zipfile
from datetime import date
from typing import Optional

import yield_mvi as ym

log = logging.getLogger("yield_multi")

ANALYSES_DIR = os.path.join("uploads", "yield_analyses")
MIN_YEARS_OK = 2                # minimum lat z udaną mapą względną
STABLE_CV_PCT = 10.0            # międzyroczny CV piksela poniżej — strefa stabilna
STRATEGIES = ("potencjal", "wyrownawcza")

# --------------------------------------------------------------------------
# Mapy względne per rok i ich łączenie
# --------------------------------------------------------------------------

def year_rel(arr, inside):
    """Z rastra [EVI2_med, NDMI_p75, MVI, n] liczy mapę WZGLĘDNĄ plonu
    (średnia pola = 1.0). Progi jakości jak w yield_mvi.disaggregate."""
    import numpy as np
    mvi, nscenes = arr[2], arr[3]
    valid = inside & np.isfinite(mvi) & (nscenes >= ym.MIN_VALID_SCENES)
    n_inside = int(np.count_nonzero(inside))
    n_valid = int(np.count_nonzero(valid))
    if n_inside == 0 or n_valid / n_inside < ym.MIN_VALID_FRAC:
        raise ValueError(f"za mało pikseli z co najmniej {ym.MIN_VALID_SCENES} czystymi scenami "
                         f"({n_valid} z {n_inside})")
    mvi_pos = np.where(valid, np.maximum(mvi, 0.0), np.nan)
    mean_mvi = float(np.nanmean(mvi_pos))
    if mean_mvi < ym.MIN_MEAN_MVI:
        raise ValueError(f"średni MVI bardzo niski ({mean_mvi:.3f}) — w oknie sezonu nie widać "
                         f"rozwiniętego łanu")
    rel = mvi_pos / mean_mvi
    cv = 100.0 * float(np.nanstd(rel[valid]) / np.nanmean(rel[valid]))
    return {"rel": rel, "valid": valid, "cv_pct": round(cv, 1), "mean_mvi": round(mean_mvi, 4),
            "median_scenes": int(np.nanmedian(nscenes[valid]))}


def combine_years(rels):
    """Średnia map względnych z lat + stabilność (międzyroczny CV piksela).
    Piksel ważny, gdy ma dane z >= połowy lat (min. 2)."""
    import numpy as np
    stack = np.stack([r["rel"] for r in rels])                    # [Y, H, W]
    n_ok = np.sum([r["valid"] for r in rels], axis=0)
    need = max(MIN_YEARS_OK, int(np.ceil(len(rels) / 2.0)))
    valid = n_ok >= need
    import warnings as _w
    with np.errstate(invalid="ignore"), _w.catch_warnings():
        # piksele poza polem to same NaN-y -> nanmean/nanstd ostrzegają o pustych
        # wycinkach; wynik i tak maskujemy przez `valid`
        _w.simplefilter("ignore", RuntimeWarning)
        rel_mean = np.where(valid, np.nanmean(stack, axis=0), np.nan)
        rel_std = np.where(valid & (n_ok >= 2), np.nanstd(stack, axis=0), np.nan)
    # renormalizacja: średnia pola dokładnie 1.0 (nanmean po maskowaniu)
    m = float(np.nanmean(rel_mean[valid])) if valid.any() else 0.0
    if m <= 0:
        raise ValueError("Połączona mapa względna jest pusta — żaden rok nie dał danych.")
    rel_mean = rel_mean / m
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_px = 100.0 * rel_std / rel_mean
    return {"rel": rel_mean, "valid": valid, "cv_px": cv_px, "n_ok": n_ok}


def zones_from_rel(rel, valid):
    """Klasy stref 1..5 z mapy względnej (progi i sito jak w yield_mvi)."""
    import numpy as np
    zones = np.full(rel.shape, 0, dtype="uint8")
    zones[valid] = (np.digitize(rel[valid], ym.ZONE_BREAKS) + 1).astype("uint8")
    return ym._sieve(zones, valid)


def zone_stats_multi(rel, zones, valid, cv_px):
    import numpy as np
    px_ha = (ym.PIXEL_M * ym.PIXEL_M) / 10000.0
    n_valid = int(np.count_nonzero(valid))
    out = []
    for k in range(1, len(ym.ZONE_LABELS) + 1):
        m = zones == k
        cnt = int(np.count_nonzero(m))
        stab = float(np.nanmean(cv_px[m])) if cnt and np.isfinite(cv_px[m]).any() else None
        out.append({"zone": k, "label": ym.ZONE_LABELS[k - 1], "color": ym.ZONE_COLORS[k - 1],
                    "area_ha": round(cnt * px_ha, 2),
                    "share_pct": round(100.0 * cnt / n_valid, 1) if n_valid else 0.0,
                    "mean_rel_pct": round(100.0 * float(np.nanmean(rel[m])), 1) if cnt else None,
                    "interyear_cv_pct": round(stab, 1) if stab is not None else None})
    return out


# --------------------------------------------------------------------------
# Pełny przebieg analizy wieloletniej
# --------------------------------------------------------------------------

def build_analysis(geom_4326, years: list, analysis_id: int) -> dict:
    """years: [{year, crop, yield_t_ha, date_from?, date_to?}, ...]
    Liczy mapę względną dla każdego roku (osobne zapytanie Process API),
    łączy, wyznacza strefy i zapisuje pliki. Zwraca statystyki."""
    import numpy as np
    import water_geometry as wg
    import water_stress
    wg.check_field_extent(geom_4326)
    inner = water_stress.inner_buffer_utm(geom_4326, ym.EDGE_BUFFER_M)
    rels, year_infos, transform, bounds, shape = [], [], None, None, None
    inside = None
    for y in sorted(years, key=lambda v: v["year"]):
        yr, crop = int(y["year"]), y.get("crop") or ""
        d0, d1 = ym.crop_window(crop, yr)
        t0 = date.fromisoformat(y["date_from"]) if y.get("date_from") else d0
        t1 = date.fromisoformat(y["date_to"]) if y.get("date_to") else d1
        info = {"year": yr, "crop": crop, "yield_t_ha": y.get("yield_t_ha"),
                "date_from": t0.isoformat(), "date_to": t1.isoformat(), "status": "ok"}
        try:
            arr, tr, bd = ym.fetch_mvi_raster(geom_4326, t0, t1)
            if shape is None:
                shape, transform, bounds = arr.shape[1:], tr, bd
                inside = ym.field_mask(inner, shape, tr)
                full = ym.field_mask(geom_4326, shape, tr)
                if inside.sum() < 0.5 * full.sum():
                    inside = full
            elif arr.shape[1:] != shape:
                raise ValueError(f"raster {arr.shape[1:]} niezgodny z {shape}")
            r = year_rel(arr, inside)
            info.update(cv_pct=r["cv_pct"], mean_mvi=r["mean_mvi"],
                        median_scenes=r["median_scenes"])
            rels.append(r | {"year": yr})
        except Exception as e:          # rok bez danych nie przerywa całej analizy
            log.warning("Analiza %s, rok %s: %s", analysis_id, yr, e)
            info.update(status="error", error=str(e))
        year_infos.append(info)
    if len(rels) < MIN_YEARS_OK:
        errs = "; ".join(f"{i['year']}: {i.get('error', '?')}" for i in year_infos if i["status"] == "error")
        raise ValueError(f"Mapę wieloletnią liczę z co najmniej {MIN_YEARS_OK} udanych lat "
                         f"(udane: {len(rels)}). Błędy — {errs}")
    comb = combine_years(rels)
    zones = zones_from_rel(comb["rel"], comb["valid"])
    zstats = zone_stats_multi(comb["rel"], zones, comb["valid"], comb["cv_px"])
    out_dir = os.path.join(ANALYSES_DIR, str(analysis_id))
    os.makedirs(out_dir, exist_ok=True)
    ym.write_geotiff(os.path.join(out_dir, "rel_sredni.tif"), comb["rel"], transform)
    _write_zones_tif(os.path.join(out_dir, "strefy.tif"), zones, transform)
    for r in rels:
        with open(os.path.join(out_dir, f"rok_{r['year']}.png"), "wb") as f:
            f.write(ym.overlay_png(r["rel"], r["valid"], 1.0))
        ym.write_geotiff(os.path.join(out_dir, f"rel_{r['year']}.tif"), r["rel"], transform)
    with open(os.path.join(out_dir, "nakladka.png"), "wb") as f:
        f.write(ym.overlay_png(comb["rel"], comb["valid"], 1.0))
    gj = ym.zones_geojson(zones, comb["valid"], transform)
    with open(os.path.join(out_dir, "strefy.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f)
    vals = comb["rel"][comb["valid"]]
    px_ha = (ym.PIXEL_M * ym.PIXEL_M) / 10000.0
    stats = {"years": year_infos, "years_ok": len(rels),
             "cv_pct": round(100.0 * float(np.nanstd(vals) / np.nanmean(vals)), 1),
             "p10_rel_pct": round(100.0 * float(np.nanpercentile(vals, 10)), 1),
             "p90_rel_pct": round(100.0 * float(np.nanpercentile(vals, 90)), 1),
             "valid_pixels": int(np.count_nonzero(comb["valid"])),
             "area_ha": round(int(np.count_nonzero(comb["valid"])) * px_ha, 2),
             "field_area_ha": wg.area_ha(geom_4326),
             "stable_share_pct": _stable_share(comb), "zones": zstats,
             "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top]}
    stats["high_variability"] = stats["cv_pct"] > ym.HIGH_CV_PCT
    return stats


def _stable_share(comb):
    """Udział pikseli o międzyrocznym CV < STABLE_CV_PCT (stabilne strefy —
    tam zmienna dawka ma największy sens)."""
    import numpy as np
    ok = np.isfinite(comb["cv_px"])
    if not ok.any():
        return None
    return round(100.0 * float(np.count_nonzero(comb["cv_px"][ok] < STABLE_CV_PCT) / np.count_nonzero(ok)), 1)


def _write_zones_tif(path, zones, transform):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=zones.shape[0], width=zones.shape[1],
                       count=1, dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
                       compress="deflate") as ds:
        ds.write(zones, 1)
        ds.set_band_description(1, "strefa 1-5 (0 = poza polem)")


# --------------------------------------------------------------------------
# Dawki nawozu per strefa
# --------------------------------------------------------------------------

def fertilizer_doses(zones_stats: list, avg_dose: float, strategy: str = "potencjal",
                     rounding: int = 5, dose_min: Optional[float] = None,
                     dose_max: Optional[float] = None) -> dict:
    """Rozdziela dawkę średnią [kg/ha] na strefy. Zwraca listę stref z dawką
    i podsumowanie. Zachowanie masy: średnia ważona powierzchnią = avg_dose
    (przed zaokrągleniem i widełkami; po nich raportujemy rzeczywistą)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Nieznana strategia '{strategy}' (dozwolone: {', '.join(STRATEGIES)})")
    if avg_dose <= 0:
        raise ValueError("Dawka średnia musi być dodatnia.")
    zs = [z for z in zones_stats if z.get("area_ha") and z.get("mean_rel_pct") is not None]
    if not zs:
        raise ValueError("Brak stref z powierzchnią — najpierw policz analizę.")
    area_total = sum(z["area_ha"] for z in zs)
    raw = {}
    for z in zs:
        rel = z["mean_rel_pct"] / 100.0
        raw[z["zone"]] = avg_dose * (rel if strategy == "potencjal" else max(0.0, 2.0 - rel))
    mean_raw = sum(raw[z["zone"]] * z["area_ha"] for z in zs) / area_total
    scale = avg_dose / mean_raw if mean_raw > 0 else 1.0
    out, step = [], max(1, int(rounding or 1))
    for z in zs:
        d = raw[z["zone"]] * scale
        if dose_min is not None:
            d = max(d, float(dose_min))
        if dose_max is not None:
            d = min(d, float(dose_max))
        d = round(d / step) * step
        out.append(dict(z, dose_kg_ha=round(d, 1), dose_total_kg=round(d * z["area_ha"], 0)))
    mean_eff = sum(z["dose_kg_ha"] * z["area_ha"] for z in out) / area_total
    return {"strategy": strategy, "avg_dose_kg_ha": float(avg_dose), "rounding_kg": step,
            "dose_min_kg_ha": dose_min, "dose_max_kg_ha": dose_max,
            "effective_avg_kg_ha": round(mean_eff, 1),
            "total_kg": round(sum(z["dose_total_kg"] for z in out), 0),
            "area_ha": round(area_total, 2), "zones": out}


# --------------------------------------------------------------------------
# Próby glebowe (CSV)
# --------------------------------------------------------------------------

_SOIL_COLS = {
    "id": ("id", "nr", "numer", "probka", "próbka", "sample", "punkt", "nr probki", "nr próbki"),
    "lat": ("lat", "latitude", "szerokosc", "szerokość", "y", "szer"),
    "lon": ("lon", "lng", "longitude", "dlugosc", "długość", "x", "dl", "dł"),
    "ph": ("ph", "ph kcl", "ph_kcl", "odczyn"),
    "p2o5": ("p2o5", "p", "fosfor", "p2o5 mg/100g", "p [mg/100g]"),
    "k2o": ("k2o", "k", "potas", "k2o mg/100g", "k [mg/100g]"),
    "mg": ("mg", "magnez", "mg mg/100g", "mg [mg/100g]"),
    "humus": ("humus", "prochnica", "próchnica", "corg", "c org", "om", "mo"),
}


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip().lower().replace(",", "."))


def parse_soil_csv(data: bytes) -> dict:
    """CSV z wynikami prób glebowych: separator , ; lub tab, przecinek
    dziesiętny OK, nagłówki PL/EN (synonimy w _SOIL_COLS). Współrzędne WGS84
    (lat/lon) — opcjonalne; bez nich próbka trafia tylko do tabeli raportu."""
    import csv
    text = data.decode("utf-8-sig", errors="replace")
    sep = max(",;\t", key=lambda s: text.splitlines()[0].count(s) if text.splitlines() else 0)
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("CSV musi mieć nagłówek i co najmniej jeden wiersz z próbką.")
    hdr = [_norm_header(h) for h in rows[0]]
    col = {}
    for key, names in _SOIL_COLS.items():
        for i, h in enumerate(hdr):
            if h in names and key not in col:
                col[key] = i
    if "ph" not in col and "p2o5" not in col and "k2o" not in col and "mg" not in col:
        raise ValueError("Nie rozpoznałem kolumn z wynikami (szukam: pH, P2O5, K2O, Mg, próchnica). "
                         f"Nagłówek pliku: {', '.join(hdr[:10])}")

    def num(row, key):
        i = col.get(key)
        if i is None or i >= len(row):
            return None
        v = (row[i] or "").strip().replace(",", ".")
        if not v:
            return None
        try:
            return float(re.sub(r"[^0-9.\-]", "", v))
        except ValueError:
            return None

    samples, warn = [], []
    for n, row in enumerate(rows[1:], start=2):
        s = {"id": (row[col["id"]].strip() if "id" in col and col["id"] < len(row) and row[col["id"]].strip()
                    else str(n - 1)),
             "lat": num(row, "lat"), "lon": num(row, "lon"), "ph": num(row, "ph"),
             "p2o5": num(row, "p2o5"), "k2o": num(row, "k2o"), "mg": num(row, "mg"),
             "humus": num(row, "humus")}
        if s["lat"] is not None and not (49.0 <= s["lat"] <= 55.5 and s["lon"] and 14.0 <= s["lon"] <= 24.5):
            # współrzędne spoza Polski — możliwe PUWG 1992 albo zamienione kolumny
            if s["lat"] and s["lon"] and 14.0 <= s["lat"] <= 24.5 and 49.0 <= s["lon"] <= 55.5:
                s["lat"], s["lon"] = s["lon"], s["lat"]
                warn.append(f"wiersz {n}: zamienione lat/lon — poprawiłem")
            else:
                warn.append(f"wiersz {n}: współrzędne poza Polską (lat={s['lat']}, lon={s['lon']}) — "
                            "punkt bez strefy (oczekuję WGS84)")
                s["lat"] = s["lon"] = None
        samples.append(s)
    return {"samples": samples, "warnings": warn}


def assign_sample_zones(samples: list, zones_tif_path: str) -> list:
    """Dopisuje numer strefy do próbek ze współrzędnymi (lookup w strefy.tif)."""
    import rasterio
    with rasterio.open(zones_tif_path) as ds:
        pts = [(s["lon"], s["lat"]) for s in samples if s["lat"] is not None]
        vals = list(ds.sample(pts)) if pts else []
    it = iter(vals)
    for s in samples:
        if s["lat"] is not None:
            z = int(next(it)[0])
            s["zone"] = z if 1 <= z <= len(ym.ZONE_LABELS) else None
        else:
            s["zone"] = None
    return samples


def soil_zone_summary(samples: list) -> list:
    """Średnie wyników per strefa (tylko próbki przypisane do stref)."""
    out = []
    for k in range(1, len(ym.ZONE_LABELS) + 1):
        sub = [s for s in samples if s.get("zone") == k]
        if not sub:
            continue
        row = {"zone": k, "label": ym.ZONE_LABELS[k - 1], "n": len(sub)}
        for key in ("ph", "p2o5", "k2o", "mg", "humus"):
            vals = [s[key] for s in sub if s.get(key) is not None]
            row[key] = round(sum(vals) / len(vals), 2) if vals else None
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Plik dla siewnika / rozsiewacza (mapa aplikacyjna)
# --------------------------------------------------------------------------

def prescription_geojson(analysis_dir: str, fert: dict) -> dict:
    """strefy.geojson + dawki -> GeoJSON aplikacyjny (atrybut rate_kg_ha)."""
    with open(os.path.join(analysis_dir, "strefy.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    dose = {z["zone"]: z["dose_kg_ha"] for z in fert["zones"]}
    feats = []
    for ft in gj["features"]:
        k = ft["properties"]["zone"]
        if k not in dose:
            continue
        ft = {"type": "Feature", "geometry": ft["geometry"],
              "properties": dict(ft["properties"], rate_kg_ha=dose[k],
                                 fert=fert.get("fert_name") or "")}
        feats.append(ft)
    return {"type": "FeatureCollection", "features": feats}


_WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
              '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')


def prescription_shapefile_zip(prescription_gj: dict, out_zip_path: str):
    """Zapisuje mapę aplikacyjną jako Shapefile (ZIP: shp/shx/dbf/prj, WGS84).
    Atrybuty: ZONE, LABEL, RATE (kg/ha) — czytane przez terminale maszyn."""
    import shapefile                      # pyshp
    base = out_zip_path[:-4] if out_zip_path.lower().endswith(".zip") else out_zip_path
    w = shapefile.Writer(base, shapeType=shapefile.POLYGON)
    w.field("ZONE", "N", decimal=0)
    w.field("LABEL", "C", size=32)
    w.field("RATE", "N", decimal=1)
    for ft in prescription_gj["features"]:
        geom, pr = ft["geometry"], ft["properties"]
        polys = geom["coordinates"] if geom["type"] == "Polygon" else [p for mp in geom["coordinates"] for p in mp]
        # shapefile: ringi zewnętrzne CW, dziury CCW — pyshp .poly() przyjmuje listę ringów
        rings = [[list(pt) for pt in ring] for ring in polys]
        w.poly(rings)
        w.record(pr["zone"], pr.get("label") or "", pr["rate_kg_ha"])
    w.close()
    with open(base + ".prj", "w") as f:
        f.write(_WGS84_PRJ)
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            zf.write(base + ext, os.path.basename(base) + ext)
    for ext in (".shp", ".shx", ".dbf", ".prj"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    return out_zip_path
