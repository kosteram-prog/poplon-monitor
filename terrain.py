"""
terrain.py — Kontekst rzeźby terenu (Open-Meteo Elevation API, bezpłatne,
BEZ klucza API) dla działki: nachylenie, ekspozycja, szacowane ryzyko erozji.

Celowo TYLKO informacyjny kontekst — NIE wchodzi do Confidence Score (patrz
analytics.compute_confidence_score) i nie zmienia fused_status w app.py. To
inny typ dowodu niż reszta systemu: nie "czy jest okrywa na tej scenie", tylko
"jak ryzykowna jest ta działka niezależnie od wyniku" — analogicznie do tego,
dlaczego NDTI nie wchodzi do Confidence Score, tylko do detect_agro_events_fused
(inny cel = inne miejsce w architekturze, nie dokładanie kolejnego ważonego
składnika do już i tak wieloczynnikowego wyniku).

Estymacja nachylenia/ekspozycji metodą różnic skończonych na 5 punktach
(centroid + 4 przesunięte o SAMPLE_OFFSET_M w N/S/E/W) — nie pełny raster DEM,
tylko punktowe próbki z Open-Meteo Elevation API (który sam korzysta z
Copernicus DEM GLO-90 / SRTM w zależności od regionu).
"""

import math
from typing import Optional

import requests

OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Odległość próbek od centroidu [m] do estymacji nachylenia. Wystarczająco
# mała, by nie wyjść daleko poza typową działkę rolną, wystarczająco duża,
# by uśrednić lokalny szum pojedynczego piksela DEM (rozdzielczość natywna
# rzędu 30-90m w zależności od źródła).
SAMPLE_OFFSET_M = 90.0

# Progi kategoryzacji ryzyka erozji wg nachylenia — ROBOCZY szacunek z
# ogólnej literatury erozji gleby (np. wytyczne dot. dobrej kultury rolnej —
# działki >6° uznaje się powszechnie za wymagające szczególnej ochrony
# przeciwerozyjnej), NIE kalibrowany na polskich poplonach — ta sama
# kategoria zastrzeżenia co REIP_NORM_BOUNDS/TILLAGE_NDTI_DROP w analytics.py.
SLOPE_LOW_MAX_DEG = 3.0
SLOPE_MEDIUM_MAX_DEG = 6.0


def _offset_point(lat: float, lon: float, dx_m: float, dy_m: float) -> tuple:
    """Przesuwa punkt (lat,lon) o (dx_m wschód, dy_m północ) metrów —
    przybliżenie płaskiej Ziemi (równoodległościowe), wystarczające dla
    przesunięć rzędu dziesiątek-set metrów używanych tutaj."""
    dlat = dy_m / 111_320.0
    dlon = dx_m / (111_320.0 * math.cos(math.radians(lat)) + 1e-9)
    return lat + dlat, lon + dlon


def _slope_aspect_from_elevations(elev_n, elev_s, elev_e, elev_w, offset_m) -> tuple:
    """Zwraca (slope_deg, aspect_deg) metodą różnic centralnych.

    aspect_deg to kierunek NAJSTROMSZEGO SPADKU (nie wzniesienia), w
    stopniach od północy zgodnie z ruchem wskazówek zegara (0°=N, 90°=E,
    180°=S, 270°=W) — standardowa konwencja meteorologiczna/geograficzna
    dla ekspozycji stoku."""
    dz_dx = (elev_e - elev_w) / (2 * offset_m)   # gradient wysokości w kierunku wschodnim
    dz_dy = (elev_n - elev_s) / (2 * offset_m)   # gradient wysokości w kierunku północnym

    slope_deg = round(math.degrees(math.atan(math.hypot(dz_dx, dz_dy))), 2)

    # Kierunek spadku = przeciwny do gradientu wzrostu wysokości.
    downhill_east  = -dz_dx
    downhill_north = -dz_dy
    aspect_deg = round(math.degrees(math.atan2(downhill_east, downhill_north)) % 360.0, 1)

    return slope_deg, aspect_deg


def _erosion_risk(slope_deg: float) -> str:
    if slope_deg <= SLOPE_LOW_MAX_DEG:
        return "niskie"
    if slope_deg <= SLOPE_MEDIUM_MAX_DEG:
        return "srednie"
    return "wysokie"


def fetch_terrain_context(geom_4326, offset_m: float = SAMPLE_OFFSET_M) -> Optional[dict]:
    """
    Pobiera wysokość w centroidzie działki + 4 punktach przesuniętych o
    `offset_m` metrów (N/S/E/W) z Open-Meteo Elevation API (bezpłatne, bez
    klucza) i estymuje nachylenie/ekspozycję.

    Zwraca None przy błędzie API lub niekompletnych danych — kontekst czysto
    informacyjny, więc brak danych NIE powinien blokować ani zmieniać oceny
    (wywołujący, patrz app.py check_multi, traktuje to nieblokująco,
    analogicznie do s1_error).

    Zwraca:
    {
        "elevation_m":  float,  # wysokość w centroidzie [m n.p.m.]
        "slope_deg":    float,  # nachylenie terenu [stopnie]
        "aspect_deg":   float,  # ekspozycja stoku (kierunek spadku) [0-360°, 0=N]
        "erosion_risk": str,    # "niskie" | "srednie" | "wysokie" — roboczy szacunek
    }
    """
    centroid = geom_4326.centroid
    lat, lon = centroid.y, centroid.x

    north = _offset_point(lat, lon, 0.0, offset_m)
    south = _offset_point(lat, lon, 0.0, -offset_m)
    east  = _offset_point(lat, lon, offset_m, 0.0)
    west  = _offset_point(lat, lon, -offset_m, 0.0)
    points = [(lat, lon), north, south, east, west]

    lats = ",".join(str(p[0]) for p in points)
    lons = ",".join(str(p[1]) for p in points)

    try:
        r = requests.get(
            OPEN_METEO_ELEVATION_URL,
            params={"latitude": lats, "longitude": lons},
            timeout=20,
        )
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    elevations = r.json().get("elevation")
    if not elevations or len(elevations) != 5 or any(e is None for e in elevations):
        return None

    elev_c, elev_n, elev_s, elev_e, elev_w = elevations
    slope_deg, aspect_deg = _slope_aspect_from_elevations(elev_n, elev_s, elev_e, elev_w, offset_m)

    return {
        "elevation_m": round(float(elev_c), 1),
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "erosion_risk": _erosion_risk(slope_deg),
    }
