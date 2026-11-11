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

import os
from datetime import date, timedelta
from typing import Optional

import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_ATTRIBUTION = "Dane pogodowe: Open-Meteo.com (CC-BY 4.0), reanaliza ERA5 / ERA5-Land (Copernicus)"

# ---------------------------------------------------------------------------
# Open-Meteo: licencja, klucz, pamięć podręczna
# ---------------------------------------------------------------------------
#
# Darmowe API Open-Meteo jest WYŁĄCZNIE do użytku niekomercyjnego, z limitem
# < 10 000 zapytań/dzień, 5 000/h, 600/min na adres IP
# (https://open-meteo.com/en/terms). Do użytku komercyjnego potrzebny jest
# plan płatny: zapytania idą wtedy na serwery z przedrostkiem "customer-"
# z parametrem apikey. Ustaw OPEN_METEO_API_KEY w .env, a aplikacja sama
# przełączy się na te serwery. Dane w obu przypadkach: CC-BY 4.0 (wymagany
# podpis źródła — OPEN_METEO_ATTRIBUTION).
#
# Pamięć podręczna w SQLite (tabela weather_cache) oszczędza limit: ten sam
# okres archiwalny nie jest pobierany drugi raz, klimatologia 1991–2020
# pobierana jest raz na punkt na rok, prognoza — co najwyżej co 3 h.
WEATHER_CACHE_ENABLED = os.environ.get("WEATHER_CACHE", "1") != "0"
CACHE_TTL_FORECAST_S = 3 * 3600
CACHE_TTL_RECENT_S = 6 * 3600          # archiwum obejmujące ostatnie dni (ERA5 jeszcze się uzupełnia)
CACHE_TTL_ARCHIVE_S = 30 * 24 * 3600   # archiwum starsze niż ~10 dni
CACHE_TTL_CLIMATE_S = 365 * 24 * 3600
CACHE_TTL_ELEVATION_S = 365 * 24 * 3600


def om_url(kind: str) -> str:
    """Adres usługi Open-Meteo; z kluczem komercyjnym — serwery customer-."""
    base = {"archive": OPEN_METEO_ARCHIVE_URL, "forecast": OPEN_METEO_FORECAST_URL,
            "elevation": OPEN_METEO_ELEVATION_URL}[kind]
    if os.environ.get("OPEN_METEO_API_KEY"):
        base = base.replace("https://", "https://customer-", 1)
    return base


def om_get(kind: str, params: dict, ttl_s: float, timeout: int = 30) -> dict:
    """Zapytanie do Open-Meteo z pamięcią podręczną. Rzuca RuntimeError przy
    błędzie (w tym 429 — przekroczony limit darmowego API)."""
    import json as _json
    params = dict(params)
    key = kind + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    if WEATHER_CACHE_ENABLED:
        cached = _cache_get(key, ttl_s)
        if cached is not None:
            return cached
    api_key = os.environ.get("OPEN_METEO_API_KEY")
    if api_key:
        params["apikey"] = api_key
    try:
        r = requests.get(om_url(kind), params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Brak połączenia z Open-Meteo: {e}")
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo błąd {r.status_code}: {r.text[:300]}")
    data = r.json()
    if WEATHER_CACHE_ENABLED:
        _cache_set(key, _json.dumps(data))
    return data


def _cache_get(key: str, ttl_s: float):
    import json as _json
    import time as _time
    try:
        import db
        row = db.get_weather_cache(key)
    except Exception:
        return None
    if not row or _time.time() - row[1] > ttl_s:
        return None
    try:
        return _json.loads(row[0])
    except ValueError:
        return None


def _cache_set(key: str, payload: str):
    try:
        import db
        db.set_weather_cache(key, payload)
    except Exception:
        pass   # brak tabeli (np. przed init_db) — po prostu bez pamięci podręcznej


def _round_coord(v: float) -> float:
    # ~1 km — siatka ERA5/ERA5-Land ma 9-25 km, więc sąsiednie pola mogą
    # dzielić jedno zapytanie (i jedną pozycję w pamięci podręcznej).
    return round(float(v), 2)

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


ARCHIVE_DAILY_VARS = [
    "temperature_2m_min", "temperature_2m_max", "temperature_2m_mean",
    "precipitation_sum", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    # zapotrzebowanie atmosfery i wilgotność gleby z modelu (ERA5-Land, m³/m³)
    "vapour_pressure_deficit_max",
    "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
]


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
        "et0_mm":       float | None,   # ewapotranspiracja wskaźnikowa FAO-56
                                        #   Penman-Monteith (Open-Meteo, ERA5) [mm]
        "vpd_max_kpa":  float | None,   # maks. dobowy deficyt pary wodnej [kPa]
        "sm_0_7", "sm_7_28", "sm_28_100": float | None,  # wilgotność gleby
                                        #   ERA5-Land, średnia dobowa [m³/m³]
    }

    Rzuca RuntimeError przy błędzie API — wywołujący (app.py check_multi)
    powinien to traktować jako NIEBLOKUJĄCE (analogicznie do S1: brak
    kontekstu pogodowego nie powinien uniemożliwiać oceny, tylko ją
    zubażać o jeden, pomocniczy sygnał).
    """
    centroid = geom_4326.centroid
    lat, lon = _round_coord(centroid.y), _round_coord(centroid.x)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": t_start.isoformat(),
        "end_date": t_end.isoformat(),
        "daily": ",".join(ARCHIVE_DAILY_VARS),
        # Doby liczone w lokalnej strefie czasowej PUNKTU (dla Polski to
        # i tak Europe/Warsaw) — dawniej na sztywno Europe/Warsaw, co dla
        # pól w Uzbekistanie przesuwało granicę doby o 4 godziny.
        "timezone": "auto",
    }
    ttl = CACHE_TTL_ARCHIVE_S if t_end < date.today() - timedelta(days=10) else CACHE_TTL_RECENT_S
    daily = om_get("archive", params, ttl).get("daily", {})
    dates  = daily.get("time") or []
    tmins  = daily.get("temperature_2m_min") or []
    tmaxs  = daily.get("temperature_2m_max") or []
    tmeans = daily.get("temperature_2m_mean") or []
    precs  = daily.get("precipitation_sum") or []
    rads   = daily.get("shortwave_radiation_sum") or []
    et0s   = daily.get("et0_fao_evapotranspiration") or []

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
            "et0_mm": _at(et0s, i),
            "vpd_max_kpa": _at(daily.get("vapour_pressure_deficit_max") or [], i),
            "sm_0_7": _at(daily.get("soil_moisture_0_to_7cm_mean") or [], i),
            "sm_7_28": _at(daily.get("soil_moisture_7_to_28cm_mean") or [], i),
            "sm_28_100": _at(daily.get("soil_moisture_28_to_100cm_mean") or [], i),
        })
    return out


# Klimatyczny bilans wodny (CWB = suma opadu minus suma ET0 w oknie) —
# zastępuje w module wodnym samą flagę "posucha" (dry_spell), która patrzy
# tylko na opad: 14 dni bez deszczu w kwietniu (ET0 ~1.5 mm/d) i w lipcu
# (ET0 ~4-5 mm/d) to zupełnie inny deficyt wody.
CWB_MIN_DATA_FRAC = 0.8   # min. udział dni z kompletnym P i ET0 w oknie

# Próg "deficytu atmosferycznego" w 14-dniowym oknie [mm]. Roboczy szacunek:
# ok. 2 tygodnie przy ET0 ~3 mm/d i kilku mm deszczu. Używany tylko do
# opisu kontekstu w werdykcie modułu wodnego, nie zmienia flagi ryzyka.
CWB_DEFICIT_14D_MM = -30.0
# Odpowiednio "wyraźnie mokro" — dodatni bilans, przy którym spadek NDMI
# raczej nie wynika z pogody.
CWB_SURPLUS_14D_MM = 10.0


def climatic_water_balance(weather_series: list[dict], end_date: date,
                           days: int) -> Optional[float]:
    """Suma (opad − ET0) [mm] z `days` dni kończących się w `end_date`
    (włącznie). None, gdy w oknie brakuje więcej niż (1−CWB_MIN_DATA_FRAC)
    dni z kompletnymi danymi (np. opóźnienie ERA5 o kilka dni na końcu
    okna) — lepiej nie podawać bilansu niż podać zaniżony brakami."""
    start = end_date - timedelta(days=days - 1)
    vals = [(r["precip_mm"] - r["et0_mm"]) for r in weather_series
            if start <= r["date"] <= end_date
            and r.get("precip_mm") is not None and r.get("et0_mm") is not None]
    if len(vals) < CWB_MIN_DATA_FRAC * days:
        return None
    # Brakujące dni uzupełniamy średnią z dostępnych, żeby bilans dotyczył
    # pełnego okna (inaczej 12 dni z 14 dawałoby systematycznie mniejszą sumę).
    mean = sum(vals) / len(vals)
    return round(mean * days, 1)


def fetch_recent_and_forecast(geom_4326, past_days: int = 92,
                              forecast_days: int = 10) -> list[dict]:
    """Opad i ET0 (FAO-56) z ostatnich `past_days` dni + prognoza na
    `forecast_days` dni — jedno zapytanie do API prognozy Open-Meteo.
    W odróżnieniu od archiwum ERA5 (opóźnienie ok. 5 dni) nie ma tu dziury
    między historią a dniem dzisiejszym, dlatego to źródło bilansu wodnego
    (water_balance.py). Zwraca [{"date", "precip_mm", "et0_mm", "is_forecast"}]."""
    c = geom_4326.centroid
    params = {
        "latitude": _round_coord(c.y), "longitude": _round_coord(c.x),
        "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,vapour_pressure_deficit_max",
        "past_days": int(past_days), "forecast_days": int(forecast_days),
        "timezone": "auto",
    }
    daily = om_get("forecast", params, CACHE_TTL_FORECAST_S).get("daily", {})
    dates = daily.get("time") or []
    precs = daily.get("precipitation_sum") or []
    et0s = daily.get("et0_fao_evapotranspiration") or []
    tmaxs = daily.get("temperature_2m_max") or []
    vpds = daily.get("vapour_pressure_deficit_max") or []
    today = date.today()
    out = []
    for i, d_str in enumerate(dates):
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        out.append({
            "date": d,
            "precip_mm": precs[i] if i < len(precs) else None,
            "et0_mm": et0s[i] if i < len(et0s) else None,
            "tmax_c": tmaxs[i] if i < len(tmaxs) else None,
            "vpd_max_kpa": vpds[i] if i < len(vpds) else None,
            "is_forecast": d > today,
        })
    return out


# ---------------------------------------------------------------------------
# Norma wieloletnia (1991–2020) — czy bieżący bilans P−ET0 jest nietypowy
# ---------------------------------------------------------------------------
#
# Samo "−35 mm w 30 dni" nic nie mówi bez kontekstu: w lipcu to norma, w
# kwietniu — susza. Porównujemy bieżący 30-dniowy bilans z rozkładem
# bilansów z tego samego okresu roku w latach 1991–2020 (ERA5, ten sam
# punkt siatki). Wynik: percentyl empiryczny (idea jak w SPEI, bez
# dopasowania rozkładu log-logistycznego — przy 90 próbkach empiryczny
# percentyl jest wystarczający i łatwy do wyjaśnienia rolnikowi).
CLIMATE_START = date(1991, 1, 1)
CLIMATE_END = date(2020, 12, 31)
CLIMATE_SHIFTS_DAYS = (-7, 0, 7)   # 3 okna na rok -> 90 próbek
CLIMATE_MIN_SAMPLES = 30
VPD_HIGH_KPA = 3.0                 # wysokie zapotrzebowanie atmosfery (roboczo)
HEAT_TMAX_C = 35.0                 # dzień upalny — próg stresu cieplnego m.in. bawełny (roboczo)


def fetch_climatology(lat: float, lon: float) -> dict:
    """{data: P − ET0 [mm]} dla 1991–2020 w punkcie (zaokrąglonym do 0.1°,
    czyli rozdzielczości ERA5-Land). Jedno zapytanie, w pamięci podręcznej rok."""
    params = {"latitude": round(float(lat), 1), "longitude": round(float(lon), 1),
              "start_date": CLIMATE_START.isoformat(), "end_date": CLIMATE_END.isoformat(),
              "daily": "precipitation_sum,et0_fao_evapotranspiration", "timezone": "auto"}
    daily = om_get("archive", params, CACHE_TTL_CLIMATE_S, timeout=90).get("daily", {})
    out = {}
    for d_str, p, e in zip(daily.get("time") or [], daily.get("precipitation_sum") or [],
                           daily.get("et0_fao_evapotranspiration") or []):
        if p is None or e is None:
            continue
        out[date.fromisoformat(d_str)] = p - e
    return out


def _same_day_in_year(d: date, year: int) -> date:
    try:
        return d.replace(year=year)
    except ValueError:          # 29 lutego w roku nieprzestępnym
        return d.replace(year=year, day=28)


def cwb_percentile(value: Optional[float], end_date: date, days: int, clim: dict) -> Optional[float]:
    """Percentyl (0-100) bilansu `value` z `days` dni kończących się w
    `end_date` na tle tego samego okresu roku w normie wieloletniej."""
    if value is None or not clim:
        return None
    samples = []
    for year in range(CLIMATE_START.year, CLIMATE_END.year + 1):
        for shift in CLIMATE_SHIFTS_DAYS:
            e = _same_day_in_year(end_date, year) + timedelta(days=shift)
            vals = [clim.get(e - timedelta(days=k)) for k in range(days)]
            if all(v is not None for v in vals):
                samples.append(sum(vals))
    if len(samples) < CLIMATE_MIN_SAMPLES:
        return None
    below = sum(1 for x in samples if x < value)
    equal = sum(1 for x in samples if x == value)
    return round(100.0 * (below + 0.5 * equal) / len(samples), 1)


def cwb_class(pct: Optional[float]) -> Optional[str]:
    """Słowny opis percentyla bilansu (jak klasy suszy w SPEI)."""
    if pct is None:
        return None
    if pct <= 10:
        return "wyjątkowo sucho"
    if pct <= 20:
        return "sucho"
    if pct >= 90:
        return "wyjątkowo mokro"
    if pct >= 80:
        return "mokro"
    return "w normie"


def forecast_summary(series: list, today: date, days: int = 7) -> Optional[dict]:
    """Podsumowanie prognozy na `days` dni od jutra (z fetch_recent_and_forecast):
    suma opadu i ET0, maks. temperatura i VPD, liczba dni upalnych oraz
    flaga 'sucho i gorąco' (mało deszczu + upał albo wysokie VPD)."""
    fut = [r for r in series if today < r["date"] <= today + timedelta(days=days)]
    if not fut:
        return None
    p = sum(r.get("precip_mm") or 0.0 for r in fut)
    e = sum(r.get("et0_mm") or 0.0 for r in fut)
    tmax = max((r["tmax_c"] for r in fut if r.get("tmax_c") is not None), default=None)
    vpd = max((r["vpd_max_kpa"] for r in fut if r.get("vpd_max_kpa") is not None), default=None)
    heat_days = sum(1 for r in fut if (r.get("tmax_c") or -99) >= HEAT_TMAX_C)
    dry_hot = p < 5.0 and (heat_days > 0 or (vpd is not None and vpd >= VPD_HIGH_KPA))
    return {"days": len(fut), "precip_mm": round(p, 1), "et0_mm": round(e, 1),
            "cwb_mm": round(p - e, 1), "tmax_c": tmax, "vpd_max_kpa": vpd,
            "heat_days": heat_days, "dry_hot": dry_hot}


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
