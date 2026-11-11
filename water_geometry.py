"""Geometria pól modułu wodnego: numery działek ewidencyjnych (ULDK) i import
pól z plików GeoJSON / CSV.

Moduł wodny działa w Polsce i poza nią (Uzbekistan), dlatego geometria pola
jest zapisywana wprost (water_monitored_fields.geometry_wkt). Dla polskich
działek nie trzeba jej jednak wpisywać ręcznie: numer TERYT działki (albo
kilka numerów oddzielonych przecinkiem — pole złożone z kilku działek)
wystarczy, żeby pobrać obrys z ULDK (GUGiK), tak jak w module poplonów.
"""
import csv
import io
import json
import re
from typing import Optional

# Identyfikator działki ewidencyjnej: WWPPGG_R.OOOO.NR, np. 141201_1.0001.6509,
# 026401_1.0001.AR_12.123/4 (arkusz mapy), numery z ukośnikiem.
POLISH_PARCEL_RE = re.compile(r"^\d{6}_\d\.\d{4}\.\S+$")
MAX_PARCELS_PER_FIELD = 30

FIELD_PROPS = ("region", "crop_type", "soil_taw_mm", "sowing_date",
               "farmer_name", "farmer_phone", "notes")


def split_parcel_ids(text: str) -> list:
    return [p for p in re.split(r"[,;\s]+", (text or "").strip()) if p]


def is_polish_parcel_ids(text: str) -> bool:
    ids = split_parcel_ids(text)
    return bool(ids) and len(ids) <= MAX_PARCELS_PER_FIELD and all(
        POLISH_PARCEL_RE.match(i) for i in ids)


def geometry_from_parcel_ids(text: str) -> dict:
    """Obrys pola z ULDK dla jednego albo kilku numerów działek (suma).
    Zwraca {"geometry_wkt", "parcels": [{parcel_id, gmina, ...}], "area_ha"}.
    Rzuca core.AppError, gdy któraś działka nie istnieje / ULDK niedostępny."""
    import core
    from shapely.ops import unary_union
    ids = split_parcel_ids(text)
    if not ids:
        raise core.AppError("Podaj numer działki.")
    if len(ids) > MAX_PARCELS_PER_FIELD:
        raise core.AppError(f"Za dużo działek w jednym polu (maks. {MAX_PARCELS_PER_FIELD}).")
    geoms, parcels = [], []
    for pid in ids:
        geom, meta = core.fetch_parcel_geometry(pid)
        geoms.append(geom)
        parcels.append({"parcel_id": pid, **{k: v for k, v in (meta or {}).items()
                                             if k in ("gmina", "powiat", "wojewodztwo")}})
    union = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    return {"geometry_wkt": union.wkt, "parcels": parcels, "area_ha": area_ha(union)}


# Maksymalna rozpiętość jednego pola. Pole może składać się z kilku działek,
# ale wszystkie muszą leżeć obok siebie: Sentinel Hub odrzuca obszar szerszy
# niż 2500 pikseli (25 km przy 10 m), a działki z różnych gmin to w praktyce
# kilka osobnych pól, które trzeba oceniać osobno.
MAX_FIELD_EXTENT_KM = 5.0


def extent_km(geom_4326) -> float:
    """Dłuższy bok prostokąta otaczającego geometrię [km] (geodezyjnie)."""
    from pyproj import Geod
    minx, miny, maxx, maxy = geom_4326.bounds
    g = Geod(ellps="WGS84")
    midy, midx = (miny + maxy) / 2, (minx + maxx) / 2
    w = g.inv(minx, midy, maxx, midy)[2]
    h = g.inv(midx, miny, midx, maxy)[2]
    return round(max(w, h) / 1000.0, 2)


def check_field_extent(geom_4326, label: str = "Pole") -> None:
    """Rzuca ValueError z czytelnym komunikatem, gdy pole jest za rozległe."""
    ext = extent_km(geom_4326)
    if ext > MAX_FIELD_EXTENT_KM:
        raise ValueError(
            f"{label} ma rozpiętość ok. {ext:.0f} km — części leżą daleko od siebie, więc to "
            f"raczej kilka osobnych pól niż jedno (dopuszczalne do {MAX_FIELD_EXTENT_KM:.0f} km). "
            f"Dodaj je osobno: zaznacz „Każda działka jako osobne pole” albo użyj importu z pliku.")


def area_ha(geom_4326) -> float:
    from pyproj import Geod
    return round(abs(Geod(ellps="WGS84").geometry_area_perimeter(geom_4326)[0]) / 1e4, 2)


# ---------------------------------------------------------------------------
# Import z plików
# ---------------------------------------------------------------------------

def _to_wgs84(geom, crs_name: Optional[str]):
    if not crs_name or crs_name.upper() in ("EPSG:4326", "OGC:CRS84", "CRS84",
                                             "URN:OGC:DEF:CRS:OGC:1.3:CRS84"):
        return geom
    from pyproj import Transformer
    from shapely.ops import transform
    code = crs_name.split("::")[-1] if "::" in crs_name else crs_name
    if code.isdigit():
        code = f"EPSG:{code}"
    tr = Transformer.from_crs(code, "EPSG:4326", always_xy=True).transform
    return transform(tr, geom)


def _field_from_props(props: dict, wkt: Optional[str], i: int) -> dict:
    pid = str(props.get("parcel_id") or props.get("id") or props.get("name")
              or props.get("ID_DZIALKI") or f"pole-{i}").strip()
    mode = str(props.get("irrigation_mode") or "").strip().lower()
    if mode in ("irygacyjny", "nawadniane", "irrigated", "irrigation"):
        mode = "irrigation"
    elif mode in ("deszczowy", "rainfed", "bez nawadniania"):
        mode = "rainfed"
    else:
        mode = None   # wołający uzupełni domyślnym
    f = {"parcel_id": pid, "irrigation_mode": mode, "geometry_wkt": wkt}
    for k in FIELD_PROPS:
        v = props.get(k)
        if v not in (None, ""):
            f[k] = float(str(v).replace(",", ".")) if k == "soil_taw_mm" else str(v).strip()
    return f


def parse_fields_text(filename: str, content: str) -> list:
    """Lista pól z treści pliku GeoJSON albo CSV. Pole bez geometrii ma
    geometry_wkt=None (do uzupełnienia z ULDK, gdy numer jest polski).
    CSV: separator , albo ; ; kolumna z geometrią: wkt / WKT / geometry_wkt."""
    from shapely import wkt as shapely_wkt
    from shapely.geometry import shape
    name = (filename or "").lower()
    out = []
    if name.endswith((".geojson", ".json")) or content.lstrip().startswith("{"):
        gj = json.loads(content)
        crs_name = ((gj.get("crs") or {}).get("properties") or {}).get("name")
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        for i, ft in enumerate(feats, 1):
            props = ft.get("properties") or {}
            wkt = None
            if ft.get("geometry"):
                wkt = _to_wgs84(shape(ft["geometry"]), crs_name).wkt
            out.append(_field_from_props(props, wkt, i))
        return out
    sample = content[:2000]
    delim = ";" if sample.count(";") > sample.count(",") else ","
    for i, row in enumerate(csv.DictReader(io.StringIO(content.lstrip("﻿")), delimiter=delim), 1):
        row = {(k or "").strip(): v for k, v in row.items()}
        wkt = row.get("wkt") or row.get("WKT") or row.get("geometry_wkt") or None
        if wkt:
            shapely_wkt.loads(wkt)   # walidacja — błąd -> wyjątek z numerem wiersza u wołającego
        out.append(_field_from_props(row, wkt, i))
    return out


def load_fields(path: str) -> list:
    with open(path, encoding="utf-8-sig") as f:
        return parse_fields_text(path, f.read())


# ---------------------------------------------------------------------------
# Gotowe obrysy pól spoza Polski: Fields of The World (FTW Global)
# ---------------------------------------------------------------------------
#
# Globalna mapa granic pól z Sentinel-2 (10 m, roczniki 2024 i 2025),
# CC-BY-4.0, bez konta i klucza: https://source.coop/ftw/global-data
# To NIE jest kataster — obrys "pola widocznego z satelity" (model ML) z
# oceną pewności 0-100; zawsze do obejrzenia na mapie przed zatwierdzeniem.
#
# Wariant "przez internet": plik kraju (GeoParquet, setki MB - kilka GB)
# czytamy zdalnie przez DuckDB (httpfs). Filtr na kolumnie bbox pozwala
# pobrać tylko grupy wierszy obejmujące kliknięty punkt — kilka sekund na
# zapytanie, bez pobierania całego pliku. Rozszerzenie httpfs DuckDB
# pobiera przy pierwszym użyciu z extensions.duckdb.org.

FTW_BASE_URL = ("https://data.source.coop/ftw/global-data/predictions/vectors/alpha/"
                "results-by-admin-conf")
FTW_ATTRIBUTION = "Obrysy pól: Fields of The World (FTW Global, CC-BY-4.0)"
# kod: (nazwa pliku, przybliżony bbox kraju lon_min, lat_min, lon_max, lat_max).
# Kolejność = kolejność prób, gdy punkt leży w kilku bboxach (Dolina Fergańska).
FTW_COUNTRIES = {
    "UZ": ("Uzbekistan", 55.9, 37.1, 73.2, 45.6),
    "TJ": ("Tajikistan", 67.3, 36.6, 75.2, 41.1),
    "KG": ("Kyrgyzstan", 69.2, 39.1, 80.3, 43.3),
    "TM": ("Turkmenistan", 52.4, 35.1, 66.7, 42.8),
    "KZ": ("Kazakhstan", 46.4, 40.5, 87.4, 55.5),
    "PL": ("Poland", 14.1, 49.0, 24.2, 54.9),
}
POLAND_BBOX = (14.1, 49.0, 24.2, 54.9)
FTW_SIMPLIFY_M = 6.0    # wygładzenie "schodków" pikseli 10 m (mniej wierzchołków do edycji)
FTW_MIN_HOLE_M2 = 1000.0  # dziury < 0.1 ha to zwykle szum modelu (pojedyncze drzewo, cień), nie wyłączenie z pola


def in_poland_bbox(lon: float, lat: float) -> bool:
    x0, y0, x1, y1 = POLAND_BBOX
    return x0 <= lon <= x1 and y0 <= lat <= y1


def ftw_candidate_countries(lon: float, lat: float) -> list:
    return [cc for cc, (_, x0, y0, x1, y1) in FTW_COUNTRIES.items()
            if x0 <= lon <= x1 and y0 <= lat <= y1]


def ftw_url(country_code: str) -> str:
    name = FTW_COUNTRIES[country_code][0]
    return f"{FTW_BASE_URL}/admin:country_code={country_code}/{name}.parquet"


_ftw_con = None
_ftw_lock = None


def _ftw_connection():
    """Jedno połączenie DuckDB na proces z pamięcią metadanych plików
    (enable_object_cache): stopka pliku Parquet kraju jest czytana raz, kolejne
    kliknięcia pobierają już tylko potrzebne grupy wierszy."""
    global _ftw_con, _ftw_lock
    import threading
    import duckdb
    if _ftw_lock is None:
        _ftw_lock = threading.Lock()
    if _ftw_con is None:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; SET enable_object_cache=true; "
                    "SET enable_progress_bar=false;")
        _ftw_con = con
    return _ftw_con, _ftw_lock


def _ftw_query_point(url: str, lon: float, lat: float) -> list:
    """Kandydaci (id, rok, pewność, WKB) z bboxem obejmującym punkt."""
    con, lock = _ftw_connection()
    with lock:
        return con.execute(
            'SELECT id, year("determination:datetime"), confidence, geometry '
            f"FROM read_parquet('{url}') "
            "WHERE bbox.xmin <= ? AND bbox.xmax >= ? AND bbox.ymin <= ? AND bbox.ymax >= ?",
            [lon, lon, lat, lat]).fetchall()


def _simplify_m(geom_4326, meters: float):
    from pyproj import Transformer
    from shapely.ops import transform
    from sh_utils import utm_epsg
    c = geom_4326.centroid
    epsg = utm_epsg(c.x, c.y)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    g = transform(fwd, geom_4326).simplify(meters, preserve_topology=True)
    g = _drop_small_holes(g, FTW_MIN_HOLE_M2)
    return transform(inv, g)


def _drop_small_holes(g, min_area_m2: float):
    """Usuwa dziury mniejsze niż min_area_m2 (w układzie metrycznym)."""
    from shapely.geometry import MultiPolygon, Polygon
    def fix(p):
        return Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= min_area_m2])
    if isinstance(g, Polygon):
        return fix(g)
    if isinstance(g, MultiPolygon):
        return MultiPolygon([fix(p) for p in g.geoms])
    return g


def ftw_field_at(lon: float, lat: float, countries: Optional[list] = None) -> Optional[dict]:
    """Pole z FTW zawierające punkt — z najnowszego rocznika, a przy remisie
    o najwyższej pewności. None, gdy w żadnym pasującym kraju nie ma pola
    w tym punkcie (np. zabudowa, droga, poza zasięgiem). Wynik w cache."""
    from sh_utils import cached_call
    key = ("ftw_at", round(lon, 6), round(lat, 6), tuple(countries or ()))
    return cached_call(key, lambda: _ftw_field_at_raw(lon, lat, countries), ttl=24 * 3600)


def _ftw_field_at_raw(lon: float, lat: float, countries: Optional[list]) -> Optional[dict]:
    from shapely import wkb
    from shapely.geometry import Point
    pt = Point(lon, lat)
    for cc in (countries or ftw_candidate_countries(lon, lat)):
        if cc not in FTW_COUNTRIES:
            continue
        hits = []
        for fid, year, conf, geom_wkb in _ftw_query_point(ftw_url(cc), lon, lat):
            g = wkb.loads(bytes(geom_wkb))
            if g.contains(pt):
                hits.append((int(year or 0), float(conf or 0.0), str(fid), g))
        if hits:
            year, conf, fid, g = max(hits, key=lambda h: (h[0], h[1]))
            g = _simplify_m(g, FTW_SIMPLIFY_M)
            return {"source": "ftw", "ftw_id": f"{cc}-{year}-{fid}", "country": cc,
                    "year": year, "confidence": round(conf), "geometry_wkt": g.wkt,
                    "area_ha": area_ha(g), "attribution": FTW_ATTRIBUTION}
    return None
