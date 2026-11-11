"""Wspólne narzędzia do zapytań Sentinel Hub (Statistical API).

Powstało przy optymalizacji zużycia Processing Units (listopad 2026 — pomiary
i uzasadnienie w notatce projektu claude/analiza-wydajnosci-2026-11.md).
Limit konta CDSE to 300 PU/min i 10 000 PU/mies., więc tempo analizy działek
wprost zależy od tego, ile PU kosztuje jedna działka.

  * stat_bounds()       — geometria działki w UTM (prawdziwe metry) zamiast
                          EPSG:3857. Na ~52°N piksel "10 m" w 3857 ma ok. 6 m
                          w terenie, więc płaciliśmy za ~2,7x więcej pikseli,
                          niż ma Sentinel (zmierzone: 2,16 PU -> 0,75 PU dla
                          działki 12,9 ha; różnica NDVI średnio 0,005).
  * valid_fraction()    — udział wiarygodnych pikseli liczony względem pikseli
                          DZIAŁKI, nie prostokąta otaczającego. Statistical API
                          liczy sampleCount dla całego bbox, więc dawniej
                          działka zajmująca 40% bbox nigdy nie przekraczała
                          valid_frac 0,4 — nawet w bezchmurny dzień.
  * stats_time_range()  — koniec zakresu = następny dzień 00:00Z. Przy P1D
                          i końcu "23:59:59Z" ostatni (niepełny) interwał był
                          pomijany (lastIntervalBehavior=SKIP), czyli scena
                          z dnia t_end przepadała.
  * cached_call()       — krótka pamięć podręczna z "single-flight": gdy dwa
                          wątki (np. NDVI i analiza wielowskaźnikowa tej samej
                          działki w batchu) potrzebują tego samego szeregu,
                          do Sentinel Hub idzie jedno zapytanie, a drugi wątek
                          czeka na jego wynik.
"""

import hashlib
import threading
import time
from datetime import timedelta

from pyproj import Transformer
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

STAT_RES_M = 10.0          # natywna rozdzielczość pasm 10 m Sentinel-2
CACHE_TTL_SECONDS = 600    # jak cache geometrii ULDK w core.py


def utm_epsg(lon: float, lat: float) -> int:
    """Kod EPSG strefy UTM dla punktu (Polska: 33N/34N, skrawek 35N)."""
    zone = int((lon + 180.0) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def stat_bounds(geom_4326, res_m: float = STAT_RES_M):
    """Zwraca (bounds_dict, n_poly_px) dla Statistical API.

    bounds_dict — gotowy blok input.bounds (geometria w UTM + CRS),
    n_poly_px   — przybliżona liczba pikseli res_m x res_m wewnątrz działki
                  (pole w metrach / pole piksela) — mianownik valid_frac.
    Transformer tworzony per wywołanie (obiekty pyproj nie są bezpieczne
    wątkowo, a koszt jest pomijalny wobec zapytania sieciowego)."""
    c = geom_4326.centroid
    epsg = utm_epsg(c.x, c.y)
    tr = Transformer.from_crs(4326, epsg, always_xy=True).transform
    g = shp_transform(tr, geom_4326)
    bounds = {
        "geometry": mapping(g),
        "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
    }
    return bounds, g.area / (res_m * res_m)


def stats_time_range(t_start, t_end) -> dict:
    """Zakres czasu dla aggregation.timeRange obejmujący CAŁY dzień t_end."""
    return {"from": f"{t_start.isoformat()}T00:00:00Z",
            "to": f"{(t_end + timedelta(days=1)).isoformat()}T00:00:00Z"}


def valid_fraction(stats: dict, n_poly_px):
    """Udział pikseli z danymi (po masce chmur) w pikselach działki, 0..1.

    sampleCount - noDataCount = piksele wewnątrz działki, które przeszły maskę.
    Mianownik to liczba pikseli działki z geometrii; przybliżenie na brzegach
    działki może dać >1 dla bardzo małych działek, więc przycinamy do 1.
    Bez n_poly_px (np. w testach) — dawny wzór względem całego bbox."""
    n_all = stats.get("sampleCount") or 0
    n_bad = stats.get("noDataCount") or 0
    if not n_all:
        return None
    valid = max(0, n_all - n_bad)
    if n_poly_px and n_poly_px >= 1:
        return round(min(1.0, valid / n_poly_px), 3)
    return round(valid / n_all, 3)


def geom_key(geom) -> str:
    """Stabilny klucz geometrii do pamięci podręcznej."""
    wkb = getattr(geom, "wkb", None)
    if wkb:
        return hashlib.sha1(wkb).hexdigest()
    return repr(geom)


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


_cache: dict = {}
_key_locks: dict = {}
_cache_lock = threading.Lock()


def _copy(value):
    if isinstance(value, list):
        return [dict(x) if isinstance(x, dict) else x for x in value]
    return value


def cached_call(key, fn, ttl: float = CACHE_TTL_SECONDS):
    """Wynik fn() z pamięci (ważny ttl s); równoczesne wywołania z tym samym
    kluczem czekają na jedno wspólne zapytanie. Wyjątki nie są cache'owane."""
    with _cache_lock:
        ent = _cache.get(key)
        if ent and time.time() - ent[0] < ttl:
            return _copy(ent[1])
        klock = _key_locks.setdefault(key, threading.Lock())
    with klock:
        with _cache_lock:
            ent = _cache.get(key)
            if ent and time.time() - ent[0] < ttl:
                return _copy(ent[1])
        value = fn()
        with _cache_lock:
            now = time.time()
            _cache[key] = (now, value)
            for k in [k for k, e in _cache.items() if now - e[0] >= ttl]:
                _cache.pop(k, None)
                _key_locks.pop(k, None)
        return _copy(value)


def clear_caches():
    """Czyści pamięć podręczną (używane w testach)."""
    with _cache_lock:
        _cache.clear()
        _key_locks.clear()
