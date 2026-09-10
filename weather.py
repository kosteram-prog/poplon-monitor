"""
weather.py — Dane pogodowe (Open-Meteo Historical Weather API, bezpłatne,
BEZ klucza API, oparte o reanalizę ERA5) wspierające ocenę poplonu.

Cztery konkretne zastosowania (patrz analytics.py i app.py):
  1. detect_agro_events_fused — odróżnienie spadku NDVI spowodowanego
     przymrozkiem/przedłużającą się posuchą (naturalna reakcja rośliny) od
     realnej orki/likwidacji, gdy brak niezależnego potwierdzenia SAR/NDTI.
  2. get_season_phase — realny (nie tylko kalendarzowy: grudzień-luty) start
     fazy fenologicznej "winter" dla ozimych, na podstawie faktycznej daty
     pierwszego przymrozku w danym roku/regionie.
  3. get_season_phase — fenologia licząca się w stopniodniach wzrostu (GDD,
     patrz cumulative_gdd/GDD_BASE_TEMP_C) zamiast czystych tygodni od
     siewu — chłodny i ciepły sezon dają identyczną liczbę "tygodni od
     siewu", ale roślina jest w zupełnie innym stadium rozwoju.
  4. Kontekst informacyjny "warunki wzrostu" (growing_conditions_summary) —
     CELOWO tylko do wglądu operatora, NIGDY nie zmienia Confidence
     Score/statusu. Rozdział świadomy: Confidence Score mierzy wyłącznie to,
     co widać na zdjęciu (czy jest okrywa), a nie czy słaby wynik da się
     "usprawiedliwić" złymi warunkami — to pytanie polityki programu, nie
     techniki, i nie jest tu rozstrzygane automatycznie.

Odczyt punktowy dla centroidu działki (nie pełnej geometrii jak Sentinel-2/1)
— w odróżnieniu od wskaźników optycznych/SAR, gdzie liczy się precyzyjny
pomiar na powierzchni działki, tu chodzi o pogodę w okolicy (siatka ERA5 ma
rozdzielczość ~9-25km), więc dokładna geometria działki nie ma znaczenia.

Integruje się z core.py/analytics.py bez ich modyfikacji.
"""

from datetime import date, timedelta
from typing import Optional

import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Próg przymrozka istotnego agronomicznie (nie zwykłe nocne ochłodzenie) —
# poniżej tej temperatury minimalnej dobowej tkanka roślinna typowych
# poplonów (gorczyca, facelia, owies...) zaczyna wykazywać realne
# uszkodzenia/przebarwienia widoczne też w NDVI. Roboczy szacunek z ogólnej
# literatury fizjologii roślin, nie kalibrowany na polskich poplonach — ta
# sama kategoria zastrzeżenia co REIP_NORM_BOUNDS/TILLAGE_NDTI_DROP w
# analytics.py.
FROST_TMIN_C = -2.0

# Okno i próg do wykrywania posuchy: suma opadów w POPRZEDZAJĄCYCH
# DRY_SPELL_WINDOW_DAYS dniach poniżej DRY_SPELL_MAX_PRECIP_MM traktujemy
# jako "posucha kończąca się tego dnia" — wystarczająco długi okres bez
# realnych opadów, by roślinność zareagowała spadkiem NDVI bez utraty
# pokrywy (więdnięcie, ograniczenie wzrostu), nie od razu wskutek orki.
DRY_SPELL_WINDOW_DAYS = 14
DRY_SPELL_MAX_PRECIP_MM = 5.0

# Temperatura bazowa dla stopniodni wzrostu (Growing Degree Days) —
# poniżej tej temperatury średniej dobowej przyjmujemy, że typowe gatunki
# poplonu (gorczyca, facelia, owies, wyka, rzodkiew...) praktycznie nie
# akumulują wzrostu. GDD_dzień = max(0, T_średnia - GDD_BASE_TEMP_C).
#
# UWAGA (roboczy szacunek, ta sama kategoria zastrzeżenia co FROST_TMIN_C
# powyżej i REIP_NORM_BOUNDS/TILLAGE_NDTI_DROP w analytics.py): 5°C to
# powszechnie cytowana baza dla zbóż jarych i roślin chłodnolubnych w
# klimacie umiarkowanym — NIE jest to wartość niezależnie wykalibrowana
# dla konkretnej mieszanki gatunków poplonu w tym projekcie (różne gatunki
# w mieszance mają różne bazy, np. facelia/gorczyca nieco wyżej, owies
# nieco niżej). Traktuj jako przybliżenie do czasu ew. kalibracji.
GDD_BASE_TEMP_C = 5.0

# Progi GDD używane przez analytics.get_season_phase jako odpowiednik
# dzisiejszych progów tygodniowych (early < 6 tyg., peak < 16 tyg.) —
# wyprowadzone z założenia ŚREDNIEGO efektywnego tempa akumulacji ok. 10
# GDD/dzień (odpowiadające średniej dobowej ok. 15°C, typowej dla okresu
# wegetacji poplonu ścierniskowego w Polsce) — czyli w "typowym" roku GDD
# i tygodnie dają PRAKTYCZNIE TEN SAM wynik fazy; różnica ujawnia się
# tylko wtedy, gdy sezon jest wyraźnie cieplejszy/chłodniejszy niż
# przeciętnie. Podobnie jak GDD_BASE_TEMP_C — roboczy szacunek, nie
# niezależna kalibracja.
GDD_EARLY_TO_PEAK = 420.0    # ~6 tyg. * 7 dni * 10 GDD/dzień
GDD_PEAK_TO_LATE  = 1120.0   # ~16 tyg. * 7 dni * 10 GDD/dzień


def fetch_weather_series(geom_4326, t_start: date, t_end: date) -> list[dict]:
    """
    Pobiera dzienny szereg (temperatury min/średnia/max, suma opadów,
    suma promieniowania krótkofalowego) dla centroidu działki z Open-Meteo
    Historical Weather API — reanaliza ERA5, bezpłatna, bez klucza, dobre
    pokrycie nawet dla działek daleko od najbliższej stacji IMGW (w
    odróżnieniu od punktowych pomiarów stacyjnych).

    Zwraca listę słowników:
    {
        "date":         date,
        "tmin_c":       float | None,   # minimalna temperatura dobowa [°C]
        "tmax_c":       float | None,   # maksymalna temperatura dobowa [°C]
        "tmean_c":      float | None,   # średnia temperatura dobowa [°C]
        "precip_mm":    float | None,   # suma opadów dobowych [mm]
        "radiation_mj": float | None,   # suma promieniowania krótkofalowego [MJ/m²]
    }

    Rzuca RuntimeError przy błędzie API — wywołujący (app.py check_multi)
    powinien to traktować jako NIEBLOKUJĄCE (analogicznie do S1: brak
    kontekstu pogodowego nie powinien uniemożliwiać oceny, tylko ją
    zubażać o jeden, pomocniczy sygnał).
    """
    centroid = geom_4326.centroid
    lat, lon = centroid.y, centroid.x

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": t_start.isoformat(),
        "end_date": t_end.isoformat(),
        "daily": ",".join([
            "temperature_2m_min", "temperature_2m_max", "temperature_2m_mean",
            "precipitation_sum", "shortwave_radiation_sum",
        ]),
        "timezone": "Europe/Warsaw",
    }
    try:
        r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Brak połączenia z Open-Meteo: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo błąd {r.status_code}: {r.text[:300]}")

    daily = r.json().get("daily", {})
    dates  = daily.get("time") or []
    tmins  = daily.get("temperature_2m_min") or []
    tmaxs  = daily.get("temperature_2m_max") or []
    tmeans = daily.get("temperature_2m_mean") or []
    precs  = daily.get("precipitation_sum") or []
    rads   = daily.get("shortwave_radiation_sum") or []

    def _at(lst, i):
        return lst[i] if i < len(lst) else None

    out = []
    for i, d_str in enumerate(dates):
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        out.append({
            "date": d,
            "tmin_c": _at(tmins, i),
            "tmax_c": _at(tmaxs, i),
            "tmean_c": _at(tmeans, i),
            "precip_mm": _at(precs, i),
            "radiation_mj": _at(rads, i),
        })
    return out


def detect_frost_dates(weather_series: list[dict], threshold_c: float = FROST_TMIN_C) -> set:
    """Zbiór dat z przymrozkiem (tmin <= threshold_c)."""
    return {
        r["date"] for r in weather_series
        if r.get("tmin_c") is not None and r["tmin_c"] <= threshold_c
    }


def detect_dry_spell_dates(
    weather_series: list[dict],
    window_days: int = DRY_SPELL_WINDOW_DAYS,
    max_precip_mm: float = DRY_SPELL_MAX_PRECIP_MM,
) -> set:
    """Zbiór dat, dla których suma opadów w poprzedzających `window_days`
    dniach (włącznie z danym dniem) nie przekroczyła `max_precip_mm` —
    "posucha kończąca się tego dnia". O(n * window) — wystarczające dla
    typowych serii kilkudziesięciu-kilkuset dni jednego okresu wegetacji."""
    by_date = {r["date"]: (r.get("precip_mm") or 0.0) for r in weather_series}
    dry_dates = set()
    for d in by_date:
        window_start = d - timedelta(days=window_days)
        total = sum(v for dd, v in by_date.items() if window_start < dd <= d)
        if total <= max_precip_mm:
            dry_dates.add(d)
    return dry_dates


def first_frost_date(
    weather_series: list[dict],
    threshold_c: float = FROST_TMIN_C,
    after: Optional[date] = None,
) -> Optional[date]:
    """Najwcześniejsza data przymrozku w serii, opcjonalnie ograniczona do
    dat >= `after` (np. początek jesieni — żeby nie złapać przypadkowego
    zimnego poranka we wrześniu jako "pierwszy przymrozek zimy"). Używane
    przez get_season_phase do wyznaczenia REALNEGO, nie tylko
    kalendarzowego, początku fazy 'winter' dla ozimych."""
    frost_days = sorted(detect_frost_dates(weather_series, threshold_c))
    if after is not None:
        frost_days = [d for d in frost_days if d >= after]
    return frost_days[0] if frost_days else None


def _daily_gdd(rec: dict, base_temp: float = GDD_BASE_TEMP_C) -> Optional[float]:
    """GDD dla pojedynczego dnia. Preferuje tmean_c (wprost z Open-Meteo);
    gdy niedostępne, szacuje z (tmin+tmax)/2 — standardowe przybliżenie w
    agrometeorologii, gdy brak prawdziwej średniej dobowej."""
    tmean = rec.get("tmean_c")
    if tmean is None:
        tmin, tmax = rec.get("tmin_c"), rec.get("tmax_c")
        if tmin is None or tmax is None:
            return None
        tmean = (tmin + tmax) / 2.0
    return max(0.0, tmean - base_temp)


def cumulative_gdd(
    weather_series: list[dict],
    start_date: date,
    end_date: date,
    base_temp: float = GDD_BASE_TEMP_C,
) -> Optional[float]:
    """Suma stopniodni wzrostu (GDD) w oknie [start_date, end_date]
    (włącznie). Zwraca None, gdy w oknie nie ma ŻADNEGO dnia z policzalną
    temperaturą (odróżnienie od 0.0, co oznaczałoby realnie zerowy wzrost —
    ważne, żeby get_season_phase nie pomyliło "brak danych" z "zimno")."""
    values = [
        _daily_gdd(r, base_temp) for r in weather_series
        if start_date <= r["date"] <= end_date
    ]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values), 1)


def growing_conditions_summary(
    weather_series: list[dict],
    start_date: date,
    end_date: date,
    base_temp: float = GDD_BASE_TEMP_C,
) -> dict:
    """Podsumowanie warunków wzrostu w oknie [start_date, end_date] —
    CELOWO tylko informacyjne (patrz UWAGA w docstringu modułu): suma GDD,
    suma opadów, suma promieniowania. Do wyświetlenia operatorowi obok
    wyniku, NIGDY do automatycznej korekty score'u/statusu."""
    window = [r for r in weather_series if start_date <= r["date"] <= end_date]
    gdd = cumulative_gdd(weather_series, start_date, end_date, base_temp)

    precip_vals = [r["precip_mm"] for r in window if r.get("precip_mm") is not None]
    total_precip = round(sum(precip_vals), 1) if precip_vals else None

    rad_vals = [r["radiation_mj"] for r in window if r.get("radiation_mj") is not None]
    total_radiation = round(sum(rad_vals), 1) if rad_vals else None

    return {
        "window_days": len(window),
        "cumulative_gdd": gdd,
        "gdd_base_temp_c": base_temp,
        "total_precip_mm": total_precip,
        "total_radiation_mj": total_radiation,
    }
