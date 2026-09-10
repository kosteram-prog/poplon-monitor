"""
insar.py — Koherencja InSAR (Sentinel-1 SLC, CDSE On-Demand Production API)
jako DODATKOWA, RĘCZNIE ZLECANA analiza dla niejednoznacznych zdarzeń
agrotechnicznych.

Dlaczego osobny moduł i osobny, ręcznie klikany przepływ (nie automatyczna
część check_multi, jak weather.py/terrain.py):

  - Wszystkie pozostałe pomiary w tej appce (NDVI/EVI/SAVI/NDRE/NDWI/REIP/
    NDTI z analytics.py, VV/VH/RVI z sentinel1.py) idą przez Sentinel Hub
    Statistical API — SYNCHRONICZNE zapytanie, serwer sam liczy średnią po
    geometrii działki, odpowiedź w kilkanaście-kilkadziesiąt sekund.
  - Koherencja InSAR wymaga PRAWDZIWEGO przetwarzania interferometrycznego
    (koregistracja dwóch scen SLC, wyznaczenie fazy, wygenerowanie
    interferogramu) — to inny system w CDSE: On-Demand Production API
    (https://odp.dataspace.copernicus.eu/odata/v1), oparty o ASYNCHRONICZNE
    zlecenia (ProductionOrder) bez gwarantowanego czasu ukończenia ("brak
    gwarancji, że przetwarzanie zakończy się w określonym czasie" — dok.
    CDSE). Wynik to plik rastrowy (zip z GeoTIFF), nie gotowa statystyka —
    trzeba go pobrać i samodzielnie wyciąć po geometrii działki.
  - To wolniejsze, cięższe i inaczej limitowane (wspólna, darmowa pula
    zasobów CDSE, bez SLA) niż Sentinel Hub — odpalanie tego automatycznie
    dla każdej działki w każdym batchu byłoby marnotrawstwem. Sensowne
    tylko tam, gdzie operator patrzy na niejednoznaczny werdykt i chce
    dodatkowego, niezależnego fizycznego dowodu.

Przepływ:
  1. find_slc_product()      — znajdź scenę SLC pokrywającą działkę najbliżej
                                daty zdarzenia (Catalogue API).
  2. submit_coherence_order()— zleć zadanie COH12 (workflow "card_coh12_public")
                                dla tej sceny (procesor JRC sam dobiera scenę
                                PARUJĄCĄ sprzed 12 dni — nie podajemy dwóch
                                dat ręcznie, patrz UWAGA w submit_coherence_order).
  3. get_order_status()      — sprawdź status (queued/processing/completed/failed).
  4. fetch_coherence_result()— po ukończeniu: pobierz zip, wytnij średnią
                                koherencję po geometrii działki, zinterpretuj.

UWAGA (znana, jawna luka — w odróżnieniu od reszty modułów w tym projekcie,
TA integracja nie została nigdy przetestowana na żywym koncie CDSE):
dokumentacja On-Demand Production API nie podaje wprost, jak wskazać PARĘ
scen do koherencji — opis produktu COH12 ("koherencja pary obrazów 12 dni
od siebie") sugeruje, że wystarczy jeden `InputProductReference.Reference`
(scena "późniejsza"), a procesor JRC sam znajduje sparowaną scenę sprzed 12
dni na tej samej orbicie. Jeśli w praktyce API wymaga innego mechanizmu,
submit_coherence_order() będzie trzeba skorygować po pierwszym realnym
wywołaniu — kod tutaj rzuca jawny, czytelny błąd zamiast cicho zwracać złe
dane, gdyby założenie się nie potwierdziło.
"""

import io
import time
import zipfile
from datetime import date, datetime, timedelta
from typing import Optional

import requests

CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ODP_API_URL = "https://odp.dataspace.copernicus.eu/odata/v1"

# Jedyny workflow potwierdzony wprost w dokumentacji CDSE (przykład
# odpowiedzi /Workflows). COH6 (para 6 dni) istnieje w ofercie CDSE, ale
# jego dokładny identyfikator "Name" nie był podany w dokumentacji w
# momencie pisania tego modułu — do odkrycia w razie potrzeby przez
# GET {ODP_API_URL}/Workflows?$filter=contains(Name,'coh6').
COHERENCE_WORKFLOW_NAME = "card_coh12_public"
COHERENCE_PAIR_DAYS = 12

# Tolerancja wyszukiwania sceny SLC wokół daty zdarzenia — S1 rewizyta to
# ~6-12 dni, więc kilka dni tolerancji zwiększa szansę trafienia bez
# odbiegania zbytnio od interesującej nas daty.
SLC_SEARCH_TOLERANCE_DAYS = 4

# Próg interpretacji: koherencja poniżej tej wartości traktowana jako silny
# dowód zaburzenia powierzchni gleby (orka niszczy drobną strukturę
# powierzchni -> gwałtowny spadek koherencji fazowej między akwizycjami).
#
# UWAGA (roboczy szacunek, NIEKALIBROWANY na danych poplonu — ta sama
# kategoria zastrzeżenia co TILLAGE_NDTI_DROP/REIP_NORM_BOUNDS w
# analytics.py): typowe wartości z literatury InSAR dla powierzchni rolnych
# to koherencja ~0.6-0.8+ dla stabilnej powierzchni (w tym gęstej okrywy) i
# <0.3 dla świeżo zaoranej gleby, ale dokładny rozkład zależy od typu gleby,
# wilgotności i sezonu. Do czasu zebrania realnych par (potwierdzona orka
# vs brak orki) z tego projektu traktuj jako wartość roboczą.
COHERENCE_DISTURBANCE_MAX = 0.30


def _odp_token() -> str:
    """Token OAuth do ODP API — ten sam serwer tożsamości CDSE (Keycloak,
    realm 'CDSE') co Sentinel Hub, więc reużywamy core.cdse_token() zamiast
    duplikować logikę client_credentials (patrz analytics._get_token dla
    analogicznego wzorca reużycia)."""
    import core as _core
    return _core.cdse_token()


def find_slc_product(geom_4326, target_date: date,
                      tolerance_days: int = SLC_SEARCH_TOLERANCE_DAYS) -> Optional[dict]:
    """Szuka sceny Sentinel-1 SLC pokrywającej geometrię działki, najbliższej
    `target_date` (w oknie ±tolerance_days), przez Catalogue API CDSE.

    Zwraca {"id": str, "name": str, "content_date": date} albo None, gdy nic
    nie znaleziono w oknie."""
    from shapely.geometry import mapping as shp_mapping

    wkt = geom_4326.wkt
    d_from = (target_date - timedelta(days=tolerance_days)).isoformat()
    d_to = (target_date + timedelta(days=tolerance_days + 1)).isoformat()

    filter_expr = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'SLC') and "
        f"ContentDate/Start gt {d_from}T00:00:00.000Z and "
        f"ContentDate/Start lt {d_to}T00:00:00.000Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    )
    params = {"$filter": filter_expr, "$top": 20, "$orderby": "ContentDate/Start"}

    try:
        r = requests.get(CDSE_CATALOGUE_URL, params=params, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"Brak połączenia z CDSE Catalogue API: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"CDSE Catalogue API błąd {r.status_code}: {r.text[:300]}")

    products = r.json().get("value", [])
    if not products:
        return None

    def _content_date(p):
        try:
            return datetime.fromisoformat(
                p["ContentDate"]["Start"].replace("Z", "+00:00")).date()
        except (KeyError, TypeError, ValueError):
            return None

    # Najbliższa target_date spośród znalezionych scen.
    best = min(products, key=lambda p: abs(((_content_date(p) or target_date) - target_date).days))
    best_date = _content_date(best)
    return {"id": best.get("Id"), "name": best.get("Name"), "content_date": best_date}


def submit_coherence_order(product_name: str, priority: int = 1) -> str:
    """Zleca zadanie koherencji COH12 dla sceny `product_name` (patrz
    find_slc_product). Zwraca identyfikator zamówienia (order_id) — zadanie
    jest ASYNCHRONICZNE, ten wywołanie NIE czeka na wynik (patrz docstring
    modułu). Wywołujący musi odpytywać get_order_status()."""
    payload = {
        "WorkflowName": COHERENCE_WORKFLOW_NAME,
        "InputProductReference": {"Reference": product_name},
        "Priority": priority,
        "Name": f"poplon_coh_{product_name[:40]}",
    }
    headers = {"Authorization": f"Bearer {_odp_token()}", "Content-Type": "application/json"}

    try:
        r = requests.post(f"{ODP_API_URL}/ProductionOrder/OData.CSC.Order",
                          json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"Brak połączenia z CDSE On-Demand Production API: {e}")

    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Zlecenie koherencji InSAR nie powiodło się ({r.status_code}): {r.text[:400]}")

    data = r.json()
    order_id = data.get("Id") or data.get("id")
    if not order_id:
        raise RuntimeError(f"Odpowiedź CDSE bez identyfikatora zamówienia: {data}")
    return str(order_id)


def get_order_status(order_id: str) -> dict:
    """Sprawdza status zamówienia. Zwraca {"status": str, "estimated_date": str|None,
    "raw": dict} — `status` to surowa wartość z API (np. "queued", "in_progress",
    "completed", "failed"); wywołujący normalizuje wg potrzeb."""
    headers = {"Authorization": f"Bearer {_odp_token()}"}
    try:
        r = requests.get(f"{ODP_API_URL}/ProductionOrders({order_id})",
                         headers=headers, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Brak połączenia z CDSE On-Demand Production API: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"Sprawdzenie statusu zamówienia {order_id} nie powiodło "
                           f"się ({r.status_code}): {r.text[:300]}")

    data = r.json()
    return {
        "status": (data.get("Status") or data.get("status") or "unknown"),
        "estimated_date": data.get("EstimatedDate"),
        "raw": data,
    }


def download_order_result(order_id: str) -> bytes:
    """Pobiera wynikowy plik (zip z GeoTIFF) dla ukończonego zamówienia."""
    headers = {"Authorization": f"Bearer {_odp_token()}"}
    try:
        r = requests.get(f"{ODP_API_URL}/ProductionOrder({order_id})/Product/$value",
                         headers=headers, timeout=180)
    except requests.RequestException as e:
        raise RuntimeError(f"Brak połączenia z CDSE On-Demand Production API: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"Pobranie wyniku zamówienia {order_id} nie powiodło się "
                           f"({r.status_code}): {r.text[:300]}")
    return r.content


def extract_mean_coherence(zip_bytes: bytes, geom_4326) -> Optional[float]:
    """Rozpakowuje zip, znajduje pierwszy plik rastrowy (.tif/.tiff/.img) i
    liczy średnią wartość piksela w obrębie geometrii działki (rasterio,
    maskowanie po poligonie w układzie WGS84 — reprojekcja geometrii do CRS
    rastra, jeśli inny niż 4326).

    Zwraca None, gdy w archiwum nie ma rastra albo maska nie objęła żadnego
    piksela (np. geometria poza zasięgiem sceny)."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping as shp_mapping

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        raster_names = [n for n in zf.namelist()
                        if n.lower().endswith((".tif", ".tiff", ".img"))]
        if not raster_names:
            return None
        raster_bytes = zf.read(raster_names[0])

    with rasterio.MemoryFile(raster_bytes) as memfile:
        with memfile.open() as src:
            geom_json = shp_mapping(geom_4326)
            if src.crs is not None and str(src.crs) != "EPSG:4326":
                geom_json = transform_geom("EPSG:4326", src.crs, geom_json)
            try:
                out_image, _ = rio_mask(src, [geom_json], crop=True, nodata=src.nodata)
            except ValueError:
                # "Input shapes do not overlap raster" — geometria poza sceną.
                return None

            band = out_image[0]
            if src.nodata is not None:
                valid = band[band != src.nodata]
            else:
                valid = band.flatten()
            valid = valid[~_is_nan(valid)]
            if valid.size == 0:
                return None
            return round(float(valid.mean()), 4)


def _is_nan(arr):
    import numpy as np
    return np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.zeros_like(arr, dtype=bool)


def interpret_coherence(mean_coherence: Optional[float]) -> dict:
    """Interpretacja progowa (patrz UWAGA przy COHERENCE_DISTURBANCE_MAX) —
    CELOWO tylko informacyjna, nie zwraca statusu/kategorii zgodności."""
    if mean_coherence is None:
        return {"disturbance_likely": None,
                "message": "Nie udało się wyznaczyć koherencji dla tej działki "
                          "(geometria poza zasięgiem sceny lub brak danych)."}
    disturbed = mean_coherence <= COHERENCE_DISTURBANCE_MAX
    return {
        "disturbance_likely": disturbed,
        "mean_coherence": mean_coherence,
        "message": (
            f"Średnia koherencja {mean_coherence:.2f} — "
            + ("niska, silny niezależny dowód zaburzenia powierzchni gleby "
               "(spójny z orką/likwidacją). Roboczy próg, niekalibrowany na "
               "danych poplonu — traktuj jako dodatkowy argument, nie "
               "rozstrzygający dowód."
               if disturbed else
               "brak silnego sygnału zaburzenia gleby w tym oknie.")
        ),
    }
