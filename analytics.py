"""
analytics.py — Wielowskaźnikowa analityka poplonów (Sentinel-2 + Sentinel-1)

Moduł rozszerza istniejący core.py o:
  - Evalscript pobierający 7 wskaźników optycznych jednym zapytaniem
  - Obliczanie EVI, SAVI, NDRE, NDWI, REIP, NDTI po stronie serwera
  - Syntetyczny Confidence Score (0-100%)
  - Fuzję z danymi SAR (RVI z Sentinel-1)
  - Detekcję zdarzeń agrotechnicznych łączącą NDVI + VV
  - Schemat bazy danych dla szeregów wielowskaźnikowych
  - Estymację biomasy (sucha masa t/ha)

Integruje się z core.py bez jego modyfikacji — importuje cdse_token() i fetch_parcel_geometry().
"""

import math
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests

from sentinel1 import _post_with_retry

# ============================================================
# EVALSCRIPT — Sentinel-2 L2A, 7 wskaźników jednym zapytaniem
# Pasma: B02(Blue), B04(Red), B05/B06/B07(RedEdge x3 -> REIP), B08(NIR),
#        B11/B12(SWIR x2 -> NDTI), SCL
# ============================================================

EVALSCRIPT_MULTI = """
//VERSION=3
// Sentinel-2 L2A — EVI, SAVI, NDRE, NDWI, NDVI, REIP, NDTI + maska SCL
//
// UWAGA (naprawiony błąd): pasma NIE są tu dzielone przez 10000. Sentinel Hub
// zwraca pasma optyczne domyślnie już jako REFLEKTANCJĘ [0-1] (tak samo
// zakłada działający EVALSCRIPT w core.py, który liczy NDVI wprost z s.B08/
// s.B04). Dawniej ten evalscript dzielił każde pasmo przez 10000 "z DN na
// reflektancję" — czego skutkiem było:
//   - NDVI/NDRE/NDWI: BEZ ZMIAN (to czyste ilorazy, skala się skraca),
//   - EVI i SAVI: ZAWSZE ~0.0001, czyli po zaokrągleniu 0.000.
// Bo tylko EVI i SAVI mają w mianowniku stałe addytywne (+1.0 oraz L=0.428),
// które nie skalują się razem z pasmami — przy pasmach rzędu 3e-5 stała
// całkowicie dominuje mianownik i wynik dąży do zera. Jeśli kiedyś zajdzie
// potrzeba pracy na DN, trzeba jawnie zadeklarować units:"DN" w input i
// skalować WSZYSTKIE wskaźniki spójnie.
//
// REIP (Red Edge Inflection Point, Guyot & Baret 1988) łączy 4 pasma na
// zboczu red-edge (B04+B05+B06+B07) w jedną fizyczną wielkość — długość
// fali [nm] maksymalnego nachylenia krzywej odbicia, silnie skorelowaną
// z chlorofilem/LAI. Celowo NIE jest to osobny, dodatkowy składnik
// Confidence Score obok NDRE — docelowo zostanie połączony z NDRE w jeden
// komponent "red edge" (patrz plan: krok 3), żeby nie dublować sygnału
// z tego samego obszaru widma i nie zawyżać sztucznie liczby "niezależnych"
// wskaźników w sparse_penalty.
//
// NDTI (Normalized Difference Tillage Index, Van Deventer et al. 1997) —
// B11 vs B12 (SWIR1 vs SWIR2). Rozróżnia glebę gołą / świeżo zaoraną od
// resztek pożniwnych i gęstej biomasy — inny cel niż pozostałe wskaźniki
// (nie "czy jest roślinność", tylko "co jest na powierzchni gleby"). Celowo
// NIE wchodzi do Confidence Score — trafi (krok 3 planu) jako dodatkowy
// dowód optyczny do detect_agro_events_fused (wykrywanie orki), analogicznie
// do dzisiejszej roli RVI z Sentinel-1.

function setup() {
  return {
    input: [{
      bands: ["B02", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL", "dataMask"]
    }],
    output: [
      { id: "ndvi",     bands: 1, sampleType: "FLOAT32" },
      { id: "evi",      bands: 1, sampleType: "FLOAT32" },
      { id: "savi",     bands: 1, sampleType: "FLOAT32" },
      { id: "ndre",     bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi",     bands: 1, sampleType: "FLOAT32" },
      { id: "reip",     bands: 1, sampleType: "FLOAT32" },
      { id: "ndti",     bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8"   }
    ]
  };
}

function evaluatePixel(s) {
  // Maska SCL: odrzuć chmury, cienie, śnieg, no-data
  const bad = [0, 1, 3, 8, 9, 10, 11];
  let valid = s.dataMask === 1 ? 1 : 0;
  if (bad.indexOf(s.SCL) !== -1) valid = 0;

  if (!valid) {
    return {
      ndvi: [NaN], evi: [NaN], savi: [NaN],
      ndre: [NaN], ndwi: [NaN], reip: [NaN], ndti: [NaN], dataMask: [0]
    };
  }

  // Pasma już w reflektancji [0-1] — bez skalowania (patrz komentarz u góry)
  const NIR   = s.B08;
  const RED   = s.B04;
  const BLUE  = s.B02;
  const RE1   = s.B05;   // Red Edge 1 — 705nm
  const RE2   = s.B06;   // Red Edge 2 — 740nm
  const RE3   = s.B07;   // Red Edge 3 — 783nm
  const SWIR1 = s.B11;   // 1610nm
  const SWIR2 = s.B12;   // 2190nm

  // NDVI — baseline
  const ndvi = (NIR - RED) / (NIR + RED + 1e-9);

  // EVI — Enhanced Vegetation Index (Huete et al. 2002)
  // Mniej saturuje niż NDVI przy gęstej biomasie, koryguje atmosferę
  const evi = 2.5 * (NIR - RED) / (NIR + 6.0*RED - 7.5*BLUE + 1.0 + 1e-9);

  // SAVI — Soil Adjusted Vegetation Index (L=0.428 optymalne dla upraw polowych)
  // Lepszy niż NDVI we wczesnych fazach wzrostu i przy rzadkiej okrywie
  const L = 0.428;
  const savi = ((NIR - RED) / (NIR + RED + L + 1e-9)) * (1.0 + L);

  // NDRE — Normalized Difference Red Edge
  // Wrażliwy na zawartość chlorofilu — sygnał wigoru roślin przed widocznym stresem
  const ndre = (NIR - RE1) / (NIR + RE1 + 1e-9);

  // NDWI — Normalized Difference Water Index (Gao 1996)
  // NIR vs SWIR: wysoki = wilgotna biomasa, niski = stres wodny lub schnięcie
  const ndwi = (NIR - SWIR1) / (NIR + SWIR1 + 1e-9);

  // REIP — Red Edge Inflection Point (Guyot & Baret 1988), w nanometrach.
  // reip = 700 + 40 * ((RED+RE3)/2 - RE1) / (RE2 - RE1)
  const reip = 700.0 + 40.0 * (((RED + RE3) / 2.0) - RE1) / (RE2 - RE1 + 1e-9);

  // NDTI — Normalized Difference Tillage Index (Van Deventer et al. 1997)
  // niski = gleba goła/świeżo zaorana, wysoki = resztki pożniwne/biomasa
  const ndti = (SWIR1 - SWIR2) / (SWIR1 + SWIR2 + 1e-9);

  return {
    ndvi:     [Math.max(-1, Math.min(1, ndvi))],
    evi:      [Math.max(-1, Math.min(2.5, evi))],
    savi:     [Math.max(-1, Math.min(1.5, savi))],
    ndre:     [Math.max(-1, Math.min(1, ndre))],
    ndwi:     [Math.max(-1, Math.min(1, ndwi))],
    reip:     [Math.max(680, Math.min(750, reip))],
    ndti:     [Math.max(-1, Math.min(1, ndti))],
    dataMask: [1]
  };
}
"""

# Fallback: łagodniejsze filtrowanie (bez SCL), gdy dla danego AOI/okresu
# maskowanie po SCL wytnie wszystkie piksele i nie zwróci żadnej sceny.
EVALSCRIPT_MULTI_RELAXED = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "dataMask"]
    }],
    output: [
      { id: "ndvi",     bands: 1, sampleType: "FLOAT32" },
      { id: "evi",      bands: 1, sampleType: "FLOAT32" },
      { id: "savi",     bands: 1, sampleType: "FLOAT32" },
      { id: "ndre",     bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi",     bands: 1, sampleType: "FLOAT32" },
      { id: "reip",     bands: 1, sampleType: "FLOAT32" },
      { id: "ndti",     bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8"   }
    ]
  };
}

function evaluatePixel(s) {
  const valid = s.dataMask === 1 ? 1 : 0;
  if (!valid) {
    return {
      ndvi: [NaN], evi: [NaN], savi: [NaN],
      ndre: [NaN], ndwi: [NaN], reip: [NaN], ndti: [NaN], dataMask: [0]
    };
  }

  // Pasma już w reflektancji [0-1] — bez skalowania (patrz EVALSCRIPT_MULTI)
  const NIR   = s.B08;
  const RED   = s.B04;
  const BLUE  = s.B02;
  const RE1   = s.B05;
  const RE2   = s.B06;
  const RE3   = s.B07;
  const SWIR1 = s.B11;
  const SWIR2 = s.B12;

  const ndvi = (NIR - RED) / (NIR + RED + 1e-9);
  const evi = 2.5 * (NIR - RED) / (NIR + 6.0*RED - 7.5*BLUE + 1.0 + 1e-9);
  const L = 0.428;
  const savi = ((NIR - RED) / (NIR + RED + L + 1e-9)) * (1.0 + L);
  const ndre = (NIR - RE1) / (NIR + RE1 + 1e-9);
  const ndwi = (NIR - SWIR1) / (NIR + SWIR1 + 1e-9);
  const reip = 700.0 + 40.0 * (((RED + RE3) / 2.0) - RE1) / (RE2 - RE1 + 1e-9);
  const ndti = (SWIR1 - SWIR2) / (SWIR1 + SWIR2 + 1e-9);

  return {
    ndvi:     [Math.max(-1, Math.min(1, ndvi))],
    evi:      [Math.max(-1, Math.min(2.5, evi))],
    savi:     [Math.max(-1, Math.min(1.5, savi))],
    ndre:     [Math.max(-1, Math.min(1, ndre))],
    ndwi:     [Math.max(-1, Math.min(1, ndwi))],
    reip:     [Math.max(680, Math.min(750, reip))],
    ndti:     [Math.max(-1, Math.min(1, ndti))],
    dataMask: [1]
  };
}
"""

# ============================================================
# STAŁE — progi wskaźników (kalibrowane empirycznie)
# ============================================================

CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/"
                  "CDSE/protocol/openid-connect/token")

# Progi wskaźników optycznych — literatura + kalibracja polska
#
# UWAGA (znana luka z audytu, naprawiona dla NDVI — patrz ndvi_norm_bounds()
# niżej): "high"=0.42 i "low"=0.30 dla ndvi NIE są tu przypadkowe — to
# dokładnie te same wartości co core.DEFAULT_NDVI_COVER_HIGH/MID. Dawniej
# były zwykłą zamrożoną kopią: core.NDVI_COVER_HIGH/MID są kalibrowalne na
# żywo przez /api/calibration/apply -> core.reload_thresholds(), ale ten
# słownik nie miał żadnego mechanizmu przeładowania (w odróżnieniu od wag,
# patrz reload_weights()). Po zatwierdzeniu nowej kalibracji NDVI normalizacja
# w compute_confidence_score cicho używałaby STAREGO progu, podczas gdy
# core.assess() i kontrola rozbieżności NDVI w app.py już liczą względem
# NOWEGO — dokładnie ta sama kategoria rozjazdu co naprawiony wcześniej
# SCORE_THRESHOLDS. Dla ndvi "high" jest teraz czytane NA ŻYWO z
# core.NDVI_COVER_HIGH (patrz ndvi_norm_bounds()); wartość 0.42 poniżej
# zostaje jako fallback/dokumentacja domyślnej wartości, ale NIE jest już
# używana bezpośrednio do normalizacji NDVI w compute_confidence_score.
# "bare"=0.15 pozostaje OSOBNĄ stałą, celowo różną od core.NDVI_BARE_MAX
# (0.28) — to dolna granica normalizacji [0,1], nie próg "jednoznacznie
# gołej gleby" z core.assess().
THRESHOLDS = {
    "ndvi": {"bare": 0.15, "low": 0.30, "high": 0.42},
    "evi":  {"bare": 0.10, "low": 0.25, "high": 0.50},
    "savi": {"bare": 0.08, "low": 0.20, "high": 0.40},
    "ndre": {"bare": 0.08, "low": 0.20, "high": 0.45},
    "ndwi": {"bare": 0.00, "low": 0.15, "high": 0.40},
}


def ndvi_norm_bounds() -> tuple:
    """Zwraca (bare, high) granice normalizacji NDVI dla Confidence Score —
    'high' czytane NA ŻYWO z core.NDVI_COVER_HIGH, nie z zamrożonej
    THRESHOLDS["ndvi"]["high"] (patrz UWAGA przy THRESHOLDS powyżej). Dzięki
    temu compute_confidence_score i calibration.compute_suggested_weights
    zawsze oceniają NDVI względem TEGO SAMEGO, aktualnie obowiązującego
    progu co core.assess() i kontrola rozbieżności NDVI w app.py — również
    po zatwierdzeniu kalibracji przez operatora."""
    import core
    return THRESHOLDS["ndvi"]["bare"], core.NDVI_COVER_HIGH

# RVI (z Sentinel-1) nie ma miejsca w THRESHOLDS (to wskaźnik SAR, nie
# optyczny), ale potrzebuje tych samych granic normalizacji [bare, high] w
# dwóch miejscach: compute_confidence_score() tutaj i
# calibration.compute_suggested_weights() — stała współdzielona, żeby obie
# strony nie mogły się rozjechać przy przyszłej zmianie.
RVI_NORM_BOUNDS = (0.05, 0.40)

# Przynależność każdego komponentu Confidence Score do "rodziny" niezależnej
# fizycznie/spektralnie — używane przez sparse_penalty (patrz
# compute_confidence_score), żeby liczyć NIEZALEŻNE dowody, nie surowe
# komponenty. NDVI, EVI i SAVI to praktycznie trzy przekształcenia tego
# samego kontrastu NIR-RED (te same natywnie 10m pasma B02/B04/B08) — trzy
# widoki jednej osi informacji, nie trzy niezależne pomiary. Licząc je jako
# 3 osobne "dostępne wskaźniki" (jak dotąd), scena z samym NDVI+EVI+SAVI
# (np. gdy natywnie 20m pasma B05/B06/B07/B11/B12 akurat nie zwróciły danych
# przy granicy działki, a 10m pasma tak) dostawała pełne zaufanie mimo braku
# realnie niezależnych dowodów — dokładnie ten sam problem, przeciwko
# któremu sparse_penalty miało chronić.
INDICATOR_FAMILY = {
    "ndvi": "nir_red",
    "evi":  "nir_red",
    "savi": "nir_red",
    "ndre": "red_edge",  # scalony komponent NDRE+REIP (patrz blend_red_edge_norm)
    "ndwi": "ndwi",
    "rvi":  "radar",
}

# Progi Confidence Score -> grade/status — JEDYNE źródło prawdy.
#
# UWAGA (znana luka z audytu): dotąd te same liczby (70/50/30/15) były
# zakodowane NIEZALEŻNIE w trzech miejscach — tu (grade A-F), w app.py przy
# wyliczaniu confidence_label ("wysoka"/"średnia"/"niska") i drugi raz przy
# wyliczaniu fused_status (OBECNY/PRAWDOPODOBNY/BRAK). Trzy osobne kopie tych
# samych magicznych liczb w dwóch plikach to gotowy przepis na rozjazd: gdyby
# kiedyś (patrz UWAGA o rekalibracji niżej) ktoś zmienił próg tylko w jednym
# miejscu, parcela mogłaby dostać np. grade="A" (potwierdzony) a jednocześnie
# fused_status="PRAWDOPODOBNY" — samosprzeczny wynik widoczny dla operatora.
# Teraz wszystkie trzy miejsca odwołują się do tej jednej stałej.
SCORE_THRESHOLDS = {"A": 70, "B": 50, "C": 30, "D": 15}

# Bazowy margines (w punktach score) dla ORIENTACYJNEGO zakresu wokół score,
# używany do score_low/score_high w compute_confidence_score. To NIE jest
# statystyczny przedział ufności (nie mamy rozkładu błędu pomiaru) — to
# jawnie opisana heurystyka niepewności wynikającej z JAKOŚCI i
# KOMPLETNOŚCI danych (valid_frac, liczba niezależnych rodzin wskaźników),
# które i tak już wpływają na sam score przez quality_penalty/sparse_penalty
# — tu tylko robimy tę niepewność WIDOCZNĄ zamiast ukrytej w jednej liczbie.
# Przy pełnej jakości i >=3 rodzinach dowodów margines = SCORE_UNCERTAINTY_BASE_PP;
# rośnie, gdy dane są niepełne/gorszej jakości (patrz compute_confidence_score).
SCORE_UNCERTAINTY_BASE_PP = 5.0

# REIP (Red Edge Inflection Point, w nanometrach) — granice normalizacji
# [bare, high] analogiczne do THRESHOLDS, ale osobne, bo REIP jest w zupełnie
# innej skali (nm, ~700-728) niż pozostałe wskaźniki (ilorazy [-1,1]).
# Gleba goła / brak realnego nachylenia red-edge -> blisko dolnej granicy
# formuły (~700-705nm); gęsta, żywotna okrywa -> ~720-728nm (Guyot & Baret;
# literatura precyzyjnego rolnictwa dla upraw polowych).
#
# REIP NIE jest osobnym, dodatkowo ważonym składnikiem Confidence Score —
# patrz compute_confidence_score: łączony jest z NDRE w jeden komponent
# "red edge" pod tym samym kluczem wag co dotychczasowe NDRE, żeby nie
# dublować sygnału z tego samego obszaru widma i nie zawyżać sztucznie
# n_available_indicators (a przez to omijać sparse_penalty).
#
# UWAGA (znana luka, w odróżnieniu od THRESHOLDS powyżej): granice [705,722]
# NIE są kalibracją polską jak progi NDVI/EVI/SAVI/NDRE/NDWI — to szacunek z
# ogólnej literatury red-edge (Guyot & Baret) dla upraw polowych w ogóle, bez
# empirycznej walidacji na danych poplonu. Blend 50/50 z dobrze umocowanym
# NDRE (patrz blend_red_edge_norm) ryzykuje więc wniesienie mniej pewnego
# sygnału do lepiej skalibrowanego — ta sama kategoria ryzyka co transfer
# współczynników modelu biomasy z innej uprawy (patrz BIOMASS_COEFFICIENTS).
# Do czasu zebrania próbki przez calibration.py (patrz blend_red_edge_norm
# używane też w calibration.compute_suggested_weights) traktuj to jako
# roboczy szacunek, nie ostateczną kalibrację.
REIP_NORM_BOUNDS = (705.0, 722.0)


def _normalize_indicator(val: Optional[float], low_bare: float, high_veg: float) -> Optional[float]:
    """Normalizuje pojedynczy wskaźnik do [0,1] na bazie progów [bare, high].
    Wydzielone na poziom modułu (nie tylko jako domknięcie wewnątrz
    compute_confidence_score), żeby blend_red_edge_norm — używane też przez
    calibration.py — mogło korzystać z DOKŁADNIE tej samej funkcji, zamiast
    trzymać drugą, potencjalnie rozjeżdżającą się kopię."""
    if val is None:
        return None
    return min(1.0, max(0.0, (val - low_bare) / (high_veg - low_bare + 1e-9)))


# Udział REIP w blendzie NDRE+REIP, zależny od fazy fenologicznej.
#
# REIP jest z natury bardziej niestabilny przy rzadkiej okrywie: formuła
# (700 + 40*((RED+RE3)/2 - RE1)/(RE2-RE1)) ma w mianowniku różnicę dwóch
# blisko siebie leżących pasm (RE2-RE1), która przy słabo wykształconym
# "ramieniu" red-edge (mało biomasy, gleba dominuje sygnał) jest mała i
# podatna na szum — dokładnie faza "early", w której reszta architektury
# (patrz CONFIDENCE_WEIGHTS_BY_PHASE) już świadomie obniża zaufanie do
# wskaźników wymagających gęstego listowia (EVI) na rzecz SAVI/NDWI. Blend
# 50/50 niezależny od fazy ignorowałby to samo ryzyko akurat tam, gdzie
# reszta kodu je uwzględnia. W fazie "peak"/"winter" canopy jest bardziej
# wykształcone (albo ocena już bardziej polega na NDRE z innych powodów),
# więc zostaje standardowe 50/50.
REIP_BLEND_WEIGHT_BY_PHASE = {"early": 0.25, "peak": 0.5, "winter": 0.5}


def blend_red_edge_norm(ndre: Optional[float], reip: Optional[float],
                        season_phase: str = "peak") -> tuple:
    """Scala NDRE i REIP w jeden znormalizowany komponent 'red edge' [0,1] —
    dokładnie ta sama logika, której używa compute_confidence_score do
    wyliczenia komponentu 'ndre'. Wydzielone tutaj i re-użyte w
    calibration.compute_suggested_weights, żeby sugestie wag oceniały
    separację TEGO SAMEGO sygnału, który faktycznie jest ważony w
    produkcji — inaczej kalibracja ocenia surowy NDRE, podczas gdy scoring
    już dawno liczy coś innego (NDRE+REIP), i sugestie wag z czasem coraz
    bardziej rozjeżdżają się z rzeczywistością (patrz audyt).

    Udział REIP w średniej jest fazowo-zależny (patrz REIP_BLEND_WEIGHT_BY_PHASE)
    — w fazie "early" REIP jest z natury mniej wiarygodny (rzadka okrywa),
    więc waży mniej niż w pozostałych fazach.

    Zwraca (wartość_znormalizowana_lub_None, liczba_źródeł 0/1/2)."""
    ndre_norm = _normalize_indicator(ndre, THRESHOLDS["ndre"]["bare"], THRESHOLDS["ndre"]["high"])
    reip_norm = _normalize_indicator(reip, *REIP_NORM_BOUNDS)
    if ndre_norm is not None and reip_norm is not None:
        w_reip = REIP_BLEND_WEIGHT_BY_PHASE.get(season_phase, 0.5)
        return ndre_norm * (1.0 - w_reip) + reip_norm * w_reip, 2
    if ndre_norm is not None:
        return ndre_norm, 1
    if reip_norm is not None:
        return reip_norm, 1
    return None, 0

# Wagi Confidence Score — zależne od fazy fenologicznej, suma = 1.0 gdy
# wszystkie wskaźniki dostępne (renormalizowane, gdy część brakuje).
#
# Dawniej: jeden stały zestaw wag przez cały sezon, mimo że select_primary_index()
# już wie, że SAVI jest lepszy we wczesnej fazie (rzadka okrywa, silny wpływ
# gleby), EVI w fazie szczytowej (gęsta biomasa, NDVI wtedy saturuje), a NDRE
# zimą (sygnał fizjologiczny/chlorofilowy odporniejszy na niskie kąty słońca
# i przymrozki niż wskaźniki strukturalne). Teraz waga każdego wskaźnika
# odzwierciedla tę samą wiedzę agronomiczną co select_primary_index — jeśli
# zmienia się jedno, warto sprawdzić czy drugie nie wymaga korekty.
#
# NDWI (wilgotność, NIR-SWIR) ma bazowo niewielką wagę — pomaga głównie
# odróżnić suchą ściernię/resztki pożniwne od żywej, uwodnionej okrywy w
# sytuacjach, gdy NDVI/EVI/SAVI są niejednoznaczne. Waga NDWI jest dodatkowo
# podnoszona kontekstowo blisko terminu likwidacji poplonu — patrz
# NDWI_TERMINATION_BOOST i _weights_for() niżej.
DEFAULT_CONFIDENCE_WEIGHTS_BY_PHASE = {
    # early: okrywa rzadka, gleba mocno widoczna — SAVI (koryguje wpływ tła
    # glebowego) i NDWI zyskują na znaczeniu, EVI (wymaga gęstego listowia,
    # by nie szumieć) traci.
    "early": {
        "ndvi": 0.24, "evi": 0.14, "savi": 0.30,
        "ndre": 0.14, "rvi": 0.10, "ndwi": 0.08,
    },
    # peak: gęsta biomasa — EVI (nie saturuje jak NDVI) dominuje, SAVI (po co
    # korygować glebę, gdy jej prawie nie widać) traci na znaczeniu.
    "peak": {
        "ndvi": 0.28, "evi": 0.26, "savi": 0.12,
        "ndre": 0.16, "rvi": 0.10, "ndwi": 0.08,
    },
    # winter (ozimy, grudzień-luty): niski kąt słońca, przymrozki, śnieg —
    # NDRE (sygnał chlorofilowy) i RVI (SAR, niewrażliwy na oświetlenie/śnieg)
    # rosną kosztem EVI/SAVI, które przy dormatnej roślinie są mniej miarodajne.
    "winter": {
        "ndvi": 0.22, "evi": 0.10, "savi": 0.10,
        "ndre": 0.32, "rvi": 0.18, "ndwi": 0.08,
    },
}

# Kopia "na żywo" — może zostać nadpisana przez reload_weights() na podstawie
# sugestii z calibration.py (operator zatwierdza ręcznie, tak jak progi NDVI).
CONFIDENCE_WEIGHTS_BY_PHASE = {k: dict(v) for k, v in DEFAULT_CONFIDENCE_WEIGHTS_BY_PHASE.items()}

# Alias wstecznie kompatybilny (dawne API modułu) — wagi fazy szczytowej,
# najbliższe dawnemu jedynemu zestawowi.
CONFIDENCE_WEIGHTS = CONFIDENCE_WEIGHTS_BY_PHASE["peak"]

# Mnożnik wagi NDWI, gdy obserwacja przypada blisko spodziewanego/deklarowanego
# terminu likwidacji poplonu — wtedy rozróżnienie "sucha ściernia vs żywa
# okrywa" ma największe znaczenie praktyczne. Ograniczone do NDWI_MAX_WEIGHT,
# żeby pojedynczy wskaźnik pomocniczy nie zdominował wyniku.
NDWI_TERMINATION_BOOST = 2.0
NDWI_MAX_WEIGHT = 0.20


def reload_weights():
    """Wczytuje ewentualne nadpisania wag Confidence Score z tabeli `settings`
    (klucz 'confidence_weights_override', JSON: {phase: {indicator: waga}}) i
    nakłada je na domyślne wagi per-fazowe. Analogicznie do core.reload_thresholds()
    — wywoływane przy starcie appki i po zatwierdzeniu sugestii kalibracyjnej.

    Buduje kompletną nową tabelę wag w zmiennej lokalnej i podmienia moduły-
    globalne CONFIDENCE_WEIGHTS_BY_PHASE/CONFIDENCE_WEIGHTS JEDNYM przypisaniem
    na końcu (atomowe pod GIL-em) — naprawa: dawniej funkcja mutowała
    istniejący słownik krok po kroku (najpierw reset wszystkich faz do
    domyślnych, dopiero potem nałożenie nadpisania), więc wątek batcha
    czytający wagi w trakcie reloadu mógł chwilowo zobaczyć zresetowane,
    niezatwierdzone-jeszcze wagi zamiast zatwierdzonej kalibracji."""
    import json
    import db

    new_table = {phase: dict(base) for phase, base in DEFAULT_CONFIDENCE_WEIGHTS_BY_PHASE.items()}

    raw = db.get_setting("confidence_weights_override")
    if raw:
        try:
            overrides = json.loads(raw)
            for phase, indicator_weights in overrides.items():
                if phase in new_table:
                    new_table[phase].update(
                        {k: float(v) for k, v in indicator_weights.items()
                         if k in new_table[phase]})
        except (ValueError, TypeError, AttributeError):
            pass

    global CONFIDENCE_WEIGHTS_BY_PHASE, CONFIDENCE_WEIGHTS
    CONFIDENCE_WEIGHTS_BY_PHASE = new_table
    CONFIDENCE_WEIGHTS = CONFIDENCE_WEIGHTS_BY_PHASE["peak"]

# Progi biomasy (t suchej masy/ha) — regresja wielowskaźnikowa
# Oparty na: Kross et al. (2015) i Marin et al. (2020) dla upraw rolnych
#
# UWAGA (znane ograniczenie modelu): współczynniki pochodzą z literatury dla
# PEŁNOSEZONOWYCH upraw polowych (kukurydza w Kross et al. 2015) — gęstej,
# jednogatunkowej biomasy zbieranej pod koniec pełnego sezonu wegetacyjnego.
# Poplon (ścierniskowy/ozimy) to co innego: krótkie okno (6-16 tyg.), często
# mieszanka gatunków (gorczyca, facelia, owies, wyka, rzodkiew...), rzadsza i
# strukturalnie inna okrywa. Nie mamy danych referencyjnych (wagi/zdjęcia z
# pola) pozwalających przekalibrować same współczynniki dla poplonów — to
# wymagałoby serii pomiarów biomasy ściętej z próbek kwadratowych,
# skorelowanych z EVI/NDRE/RVI z tego samego okresu. Dopóki takich danych nie
# ma, transfer modelu między typami upraw jest źródłem systematycznego,
# niewiadomego błędu — dlatego funkcja niżej NIE udaje fałszywej precyzji:
# zwraca przedział niepewności zamiast pojedynczej liczby, i ogranicza pułap
# do realistycznego zakresu dla poplonu (patrz BIOMASS_MAX_T_HA), zamiast
# dziedziczyć pułap odpowiedni dla kukurydzy.
BIOMASS_COEFFICIENTS = {
    "intercept": -1.2,
    "evi":       12.5,
    "ndre":       8.3,
    "rvi":        6.1,
}

# Realistyczny pułap biomasy dla poplonu (nie dla pełnosezonowej uprawy
# towarowej). Literatura agrotechniczna dla międzyplonów ścierniskowych i
# ozimych w warunkach polskich wskazuje typowo 1-4 t s.m./ha, wyjątkowo do
# ok. 6-7 t/ha przy bujnych mieszankach i długim oknie wegetacji. Dawniej
# model przycinał wynik do 15 t/ha — to pułap sensowny dla kukurydzy na
# kiszonkę, nie dla kilkutygodniowego poplonu; przy takim pułapie model
# mógłby zwrócić fałszywie wysoką wartość zamiast zasygnalizować, że wyszedł
# poza swój zakres stosowalności.
BIOMASS_MAX_T_HA = 8.0

# Heurystyczna szerokość przedziału niepewności (względna, ±% wartości
# środkowej) — NIE jest to błąd standardowy z dopasowania modelu (nie mamy
# do tego danych), tylko jawnie zadeklarowany margines odzwierciedlający
# niepewność transferu modelu między typami upraw. Szerszy, gdy RVI
# niedostępne (człon pomijany zamiast dopasowany, patrz estimate_biomass).
BIOMASS_UNCERTAINTY_WITH_RVI    = 0.30
BIOMASS_UNCERTAINTY_WITHOUT_RVI = 0.45

# Detekcja zdarzeń agrotechnicznych
TILLAGE_NDVI_DROP   = 0.15   # min. spadek NDVI świadczący o likwidacji
TILLAGE_VV_JUMP_DB  = 2.5    # min. skok VV [dB] potwierdzający orkę
TILLAGE_WINDOW_DAYS = 12     # max. przerwa między obserwacjami
# min. spadek NDTI (B11/B12) towarzyszący spadkowi NDVI, świadczący o
# odsłonięciu gleby (a nie tylko o strome roślinności) — dodatkowy dowód
# OPTYCZNY, niezależny od SAR, gdy scena S1 jest niedostępna/niedopasowana.
#
# UWAGA (znana luka, ta sama kategoria co REIP_NORM_BOUNDS i
# BIOMASS_COEFFICIENTS powyżej): w przeciwieństwie do TILLAGE_NDVI_DROP i
# TILLAGE_VV_JUMP_DB (skalibrowanych na podstawie realnych obserwacji orki
# w tym projekcie), próg 0.06 jest roboczym szacunkiem — NDTI liczone z
# B11/B12 (natywnie 20m, resamplowane do 10m) nie ma tu żadnej referencyjnej
# walidacji na danych poplonu. Zbyt niski próg = ryzyko fałszywych
# potwierdzeń przy zwykłym szumie SWIR; zbyt wysoki = utracone realne
# potwierdzenia. Traktuj jako wartość roboczą do czasu zebrania próbki
# (analogicznie jak przy modelu biomasy) — kandydat do rewizji, gdy
# pojawią się dane pozwalające ocenić rzeczywisty rozkład NDTI dla
# potwierdzonych zdarzeń orki na poplonach.
TILLAGE_NDTI_DROP   = 0.06


# ============================================================
# TOKEN
# ============================================================

_token_cache: dict = {"token": None, "exp": 0.0}


def _get_token() -> str:
    """
    Pobiera token OAuth z CDSE — własny cache zsynchronizowany z core.py.
    Próbuje najpierw przez core.cdse_token(), jako fallback pobiera samodzielnie.
    """
    import time

    # Próba 1: użyj core (wspólny cache gdy moduł jest ten sam)
    try:
        import core as _core
        tok = _core.cdse_token()
        if tok:
            return tok
    except Exception:
        pass

    # Próba 2: własny cache
    if _token_cache["token"] and time.time() < _token_cache["exp"] - 30:
        return _token_cache["token"]

    # Próba 3: pobierz świeży token bezpośrednio
    cid = os.environ.get("CDSE_CLIENT_ID")
    sec = os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError(
            "Brak CDSE_CLIENT_ID / CDSE_CLIENT_SECRET w zmiennych środowiskowych.")
    r = requests.post(
        CDSE_TOKEN_URL,
        data={"grant_type": "client_credentials",
              "client_id": cid, "client_secret": sec},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Autoryzacja CDSE nie powiodła się: {r.status_code} {r.text[:200]}")
    tok = r.json()
    _token_cache["token"] = tok["access_token"]
    _token_cache["exp"]   = time.time() + tok.get("expires_in", 600)
    return _token_cache["token"]


# ============================================================
# POBIERANIE DANYCH WIELOWSKAŹNIKOWYCH
# ============================================================

def fetch_multi_series(
    geom_4326, t_start: date, t_end: date, token: Optional[str] = None
) -> list[dict]:
    """
    Pobiera szereg czasowy 7 wskaźników optycznych (NDVI, EVI, SAVI, NDRE, NDWI,
    REIP, NDTI) jednym zapytaniem do Sentinel Hub Statistical API.

    Zwraca listę słowników (jeden per dzień z danymi):
    {
        "date":       date,
        "ndvi":       float | None,
        "evi":        float | None,
        "savi":       float | None,
        "ndre":       float | None,
        "ndwi":       float | None,
        "reip":       float | None,   # Red Edge Inflection Point [nm], ~700-728
                                       # — dziś tylko informacyjnie/w bazie, NIE
                                       # wchodzi jeszcze do compute_confidence_score
                                       # (zostanie połączony z NDRE w kroku 3 planu)
        "ndti":       float | None,   # Normalized Difference Tillage Index (B11/B12)
                                       # — dziś tylko informacyjnie/w bazie, NIE
                                       # wchodzi do Confidence Score ani do
                                       # detect_agro_events_fused (krok 3 planu:
                                       # dodatkowy dowód optyczny na orkę/glebę
                                       # gołą, analogicznie do dzisiejszej roli RVI)
        "valid_frac": float,      # frakcja wiarygodnych pikseli
    }
    """
    from shapely.geometry import mapping as shp_mapping
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    to_3857 = Transformer.from_crs(4326, 3857, always_xy=True).transform
    geom_3857 = shp_transform(to_3857, geom_4326)

    auth_token = token or _get_token()
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    def _fetch_with(evalscript: str, max_cloud: int) -> list[dict]:
        payload = {
            "input": {
                "bounds": {
                    "geometry": shp_mapping(geom_3857),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"maxCloudCoverage": max_cloud},
                }],
            },
            "aggregation": {
                "timeRange": {
                    "from": f"{t_start.isoformat()}T00:00:00Z",
                    "to":   f"{t_end.isoformat()}T23:59:59Z",
                },
                "aggregationInterval": {"of": "P1D"},
                "resx": 10, "resy": 10,
                "evalscript": evalscript,
            },
        }

        resp = _post_with_retry(CDSE_STATS_URL, payload, headers, timeout=240,
                                 refresh_token_fn=_get_token)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Sentinel Hub Statistical API błąd {resp.status_code}: {resp.text[:400]}"
            )

        out = []
        for item in resp.json().get("data", []):
            outputs = item.get("outputs", {})
            dm = outputs.get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {})
            if not dm:
                dm = outputs.get("dataMask", {}).get("bands", {}).get("B0", {}).get("stats", {})
            n_all = dm.get("sampleCount", 0)
            n_bad = dm.get("noDataCount", 0)
            if not n_all:
                continue
            valid_frac = round(1.0 - n_bad / n_all, 3)

            def _mean(key):
                try:
                    v = outputs[key]["bands"]["B0"]["stats"]["mean"]
                    return round(float(v), 4) if v is not None and math.isfinite(float(v)) else None
                except (KeyError, TypeError, ValueError):
                    return None

            ndvi = _mean("ndvi")
            if ndvi is None:
                continue

            d = datetime.fromisoformat(item["interval"]["from"].replace("Z", "+00:00")).date()
            out.append({
                "date": d,
                "ndvi": ndvi,
                "evi": _mean("evi"),
                "savi": _mean("savi"),
                "ndre": _mean("ndre"),
                "ndwi": _mean("ndwi"),
                "reip": _mean("reip"),
                "ndti": _mean("ndti"),
                "valid_frac": valid_frac,
            })
        return sorted(out, key=lambda x: x["date"])

    records = _fetch_with(EVALSCRIPT_MULTI, 90)
    if records:
        return records
    return _fetch_with(EVALSCRIPT_MULTI_RELAXED, 95)


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def _weights_for(season_phase: str = "peak", near_termination: bool = False) -> dict:
    """Zwraca zestaw wag Confidence Score dla danej fazy fenologicznej,
    z opcjonalnym podbiciem wagi NDWI blisko terminu likwidacji poplonu."""
    base = CONFIDENCE_WEIGHTS_BY_PHASE.get(season_phase, CONFIDENCE_WEIGHTS_BY_PHASE["peak"])
    weights = dict(base)
    if near_termination:
        weights["ndwi"] = min(NDWI_MAX_WEIGHT, weights["ndwi"] * NDWI_TERMINATION_BOOST)
    return weights


def compute_confidence_score(
    ndvi:  Optional[float] = None,
    evi:   Optional[float] = None,
    savi:  Optional[float] = None,
    ndre:  Optional[float] = None,
    rvi:   Optional[float] = None,   # z Sentinel-1
    ndwi:  Optional[float] = None,   # wskaźnik wilgotności — pomocniczy
    reip:  Optional[float] = None,   # Red Edge Inflection Point [nm] — łączony z NDRE
    valid_frac: float = 1.0,         # jakość danych optycznych
    season_phase: str = "peak",      # "early" | "peak" | "winter" — dobiera wagi
    near_termination: bool = False,  # blisko terminu likwidacji -> waga NDWI w górę
) -> dict:
    """
    Syntetyczny wskaźnik 'Pewności Poplonu' (0-100%).

    Architektura: ważona średnia z normalizowanych wskaźników, z wagami
    zależnymi od fazy fenologicznej (patrz CONFIDENCE_WEIGHTS_BY_PHASE) —
    ta sama logika co select_primary_index (SAVI wcześnie, EVI w szczycie,
    NDRE/RVI zimą) — z korekcją za jakość danych i sygnałem radarowym jako
    backup. Waga NDWI rośnie kontekstowo blisko terminu likwidacji.

    REIP (Red Edge Inflection Point) NIE jest osobnym, 7. ważonym składnikiem
    — jest zbyt skorelowany z NDRE (oba mierzą to samo zjawisko: stan
    chlorofilu na krawędzi czerwieni), więc dodanie go obok NDRE tylko
    zawyżałoby sztucznie n_available_indicators (a przez to omijało
    sparse_penalty) bez realnego wzrostu niezależnych dowodów. Zamiast tego
    reip i ndre są łączone (średnia znormalizowanych wartości, gdy oba
    dostępne) w jeden komponent "red edge" pod dotychczasowym kluczem wag
    "ndre" — patrz REIP_NORM_BOUNDS.

    Zwraca:
    {
        "score":        float,    # 0-100
        "score_low":    float,    # score - margines niepewności (patrz
                                  #   SCORE_UNCERTAINTY_BASE_PP) — heurystyka
                                  #   jakości/kompletności danych, NIE
                                  #   statystyczny przedział ufności
        "score_high":   float,    # score + ten sam margines
        "grade":        str,      # "A" | "B" | "C" | "D" | "F"
        "label":        str,      # opis słowny
        "components":   dict,     # znormalizowane składowe [0-1] (backward-compat)
                                  #   "ndre" to tu SCALONY komponent red-edge
                                  #   (NDRE + REIP, gdy oba dostępne)
        "raw_components": dict,   # surowe wskaźniki (NDVI/EVI/SAVI/NDRE/RVI/NDWI/REIP)
        "normalized_components": dict,  # jawna nazwa składowych [0-1]
        "data_quality": float,    # 0-1
        "sar_used":     bool,     # czy RVI był potrzebny jako backup
        "weights_used": dict,     # PEŁNA tabela wag fazy (po podbiciu NDWI),
                                  #   niezależnie od tego, które wskaźniki są
                                  #   akurat dostępne w tej obserwacji
        "effective_weights": dict,  # realny udział KAŻDEGO DOSTĘPNEGO wskaźnika
                                    #   w wyniku, PO renormalizacji (sumuje się
                                    #   do 1.0) — to jest to, co faktycznie
                                    #   wpłynęło na score, w odróżnieniu od
                                    #   weights_used
        "season_phase": str,      # faza użyta do doboru wag
        "red_edge_sources": int,  # ile z {ndre, reip} faktycznie wniosło dane
                                  #   do komponentu "ndre" (0/1/2) — przejrzystość
    }
    """
    weights = _weights_for(season_phase, near_termination)
    # Normalizacja pojedynczego wskaźnika i scalanie NDRE+REIP — funkcje
    # modułowe (nie lokalne domknięcia), żeby calibration.py mogło użyć
    # DOKŁADNIE tej samej logiki (patrz blend_red_edge_norm).
    _normalize = _normalize_indicator

    raw_components = {
        "ndvi": ndvi,
        "evi": evi,
        "savi": savi,
        "ndre": ndre,
        "rvi": rvi,
        "ndwi": ndwi,
        "reip": reip,
    }

    # Red edge: NDRE i REIP scalone w jeden komponent (patrz docstring).
    red_edge, red_edge_sources = blend_red_edge_norm(ndre, reip, season_phase)

    # Normalizacja każdego wskaźnika do [0,1]
    components = {
        "ndvi": _normalize(ndvi, *ndvi_norm_bounds()),
        "evi":  _normalize(evi,  THRESHOLDS["evi"]["bare"],  THRESHOLDS["evi"]["high"]),
        "savi": _normalize(savi, THRESHOLDS["savi"]["bare"], THRESHOLDS["savi"]["high"]),
        "ndre": red_edge,
        "rvi":  _normalize(rvi,  *RVI_NORM_BOUNDS),
        "ndwi": _normalize(ndwi, THRESHOLDS["ndwi"]["bare"], THRESHOLDS["ndwi"]["high"]),
    }

    # Sprawdź czy mamy dane optyczne
    optical = {k: v for k, v in components.items()
               if k != "rvi" and v is not None}
    sar_used = len(optical) == 0 and components["rvi"] is not None

    # Buduj wagowaną sumę tylko z dostępnych wskaźników
    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return {"score": None, "grade": "?", "label": "Brak danych",
                "components": components,
                "raw_components": raw_components,
                "normalized_components": components,
                "data_quality": 0.0, "sar_used": False,
                "weights_used": weights, "effective_weights": {},
                "season_phase": season_phase,
                "n_available_indicators": 0, "n_available_families": 0,
                "sparse_data": False, "red_edge_sources": red_edge_sources,
                "score_low": None, "score_high": None}

    total_w = sum(weights.get(k, 0.0) for k in available)
    if total_w < 1e-9:
        return {"score": None, "grade": "?", "label": "Brak danych",
                "components": components,
                "raw_components": raw_components,
                "normalized_components": components,
                "data_quality": 0.0, "sar_used": False,
                "weights_used": weights, "effective_weights": {},
                "season_phase": season_phase,
                "n_available_indicators": 0, "n_available_families": 0,
                "sparse_data": False, "red_edge_sources": red_edge_sources,
                "score_low": None, "score_high": None}

    raw_score = sum(v * weights.get(k, 0.0) for k, v in available.items()) / total_w

    # Efektywne wagi PO renormalizacji do dostępnych wskaźników (sumują się
    # do 1.0) — to jest realny udział każdego wskaźnika w wyniku, w
    # odróżnieniu od "weights_used", które pokazuje pełną tabelę wag fazy
    # (także dla wskaźników, których nie było w tej obserwacji).
    effective_weights = {k: round(weights.get(k, 0.0) / total_w, 4) for k in available}

    # Kara za NIEKOMPLETNY zestaw dowodów (mało NIEZALEŻNYCH wskaźników).
    #
    # Luka wykryta przy audycie: renormalizacja (dzielenie przez total_w
    # dostępnych wag) sprawia, że gdy zmierzony jest TYLKO JEDEN wskaźnik
    # (pozostałe to None — brak pomiaru, np. częściowy błąd ekstrakcji pasm w
    # scenie, NIE zmierzona-i-niska wartość), ten jeden wskaźnik dostaje 100%
    # wagi i może samodzielnie dać score=100. Zweryfikowane liczbowo: samo
    # NDVI=0.75 (reszta None) -> score 100.0, podczas gdy te same NDVI=0.75 z
    # resztą ZMIERZONĄ i niską -> score 28.0. To dokładnie ten sam mechanizm
    # "jeden wskaźnik dominuje", przeciwko któremu były skierowane naprawy w
    # app.py (usunięcie nadpisania werdyktu samym NDVI, wybór dnia po score
    # zamiast po NDVI, kontrola odrostu po incydencie) — bez tej łaty ten sam
    # efekt mógł wrócić tylnymi drzwiami przez sam brak danych, zamiast przez
    # jawne nadpisanie w kodzie.
    #
    # Druga luka (naprawiona w tej samej rundzie audytu): liczenie SUROWYCH
    # dostępnych komponentów zamiast NIEZALEŻNYCH rodzin dowodów. NDVI, EVI,
    # SAVI to praktycznie trzy przekształcenia tej samej osi NIR-RED (patrz
    # INDICATOR_FAMILY) — scena z samym NDVI+EVI+SAVI (bez NDRE/NDWI/RVI)
    # dostawała pełne zaufanie (n=3 >= próg) mimo braku realnie niezależnych
    # dowodów, czyli dokładnie ten sam problem "jeden sygnał udaje wiele",
    # tylko rozcieńczony do 3 skorelowanych komponentów zamiast 1. Liczymy
    # więc liczbę NIEZALEŻNYCH RODZIN (nir_red/red_edge/ndwi/radar), nie
    # surowych kluczy.
    #
    # "sar_used" (czysty fallback radarowy, gdy CAŁA optyka jest niedostępna)
    # to świadomie wspierana ścieżka jednowskaźnikowa i NIE jest tu karana —
    # to zaprojektowana sytuacja "chmury zasłoniły wszystko, radar i tak
    # widzi", a nie przypadkowa luka w ekstrakcji pasm.
    n_available = len(available)
    n_available_families = len({INDICATOR_FAMILY[k] for k in available})
    MIN_FAMILIES_FOR_FULL_TRUST = 3
    if sar_used or n_available_families >= MIN_FAMILIES_FOR_FULL_TRUST:
        sparse_penalty = 1.0
    else:
        # 1 rodzina -> 0.5x, 2 rodziny -> 0.75x (liniowo do 1.0x przy progu)
        sparse_penalty = 0.5 + 0.5 * (n_available_families - 1) / (MIN_FAMILIES_FOR_FULL_TRUST - 1)

    # Korekcja za jakość danych optycznych (valid_frac)
    # Przy valid_frac=0.35 (core.MIN_VALID_FRAC, dolna granica) →
    # (0.65-0.35)*0.25 = 0.075, czyli obniżamy pewność o MAX 7.5% (nie 15% —
    # poprawka błędnego komentarza z audytu; formuła i reszta systemu, patrz
    # test_check_multi_slaba_jakosc_o_wyzszym_score_nie_wygrywa_z_wiarygodna_scena
    # i komentarz w app.py przy usunięciu podwójnego karania za jakość,
    # zawsze zakładały 7.5% — to sam komentarz tutaj był nieaktualny).
    quality_penalty = 1.0 - max(0.0, (0.65 - valid_frac) * 0.25)
    score = round(raw_score * quality_penalty * sparse_penalty * 100, 1)

    # Mapowanie na grade (progi skalibrowane agronomicznie).
    #
    # UWAGA (znana luka): te progi (70/50/30/15) zostały dobrane pod jeden,
    # stały zestaw wag sprzed wprowadzenia CONFIDENCE_WEIGHTS_BY_PHASE. Same
    # skrajności (wszystkie wskaźniki na poziomie "bare" albo wszystkie na
    # poziomie "high") dają score niezależny od wag (0 lub 100 w każdej
    # fazie), więc tam progi są bezpieczne. Ryzyko dotyczy przypadków
    # granicznych, gdzie wskaźniki się nie zgadzają (np. NDRE wysoki, EVI
    # niski) — tam wynik może się różnić między fazami czysto z powodu wag, a
    # nie realnej różnicy w stanie okrywy. Progi NIE zostały ponownie
    # zwalidowane empirycznie pod nowe wagi (brak wystarczających danych
    # zdjęciowych w momencie wdrożenia) — do czasu zebrania próbki przez
    # calibration.py traktuj oceny B/C blisko granic progów z ostrożnością,
    # szczególnie dla fazy "winter" (najmocniej odbiegająca od dawnych wag).
    if score >= SCORE_THRESHOLDS["A"]:
        grade, label = "A", "Poplon potwierdzony"
    elif score >= SCORE_THRESHOLDS["B"]:
        grade, label = "B", "Poplon prawdopodobny"
    elif score >= SCORE_THRESHOLDS["C"]:
        grade, label = "C", "Słaba okrywa — weryfikacja"
    elif score >= SCORE_THRESHOLDS["D"]:
        grade, label = "D", "Niewystarczająca okrywa"
    else:
        grade, label = "F", "Brak poplonu"

    if sar_used:
        label += " (SAR backup)"

    # Orientacyjny zakres wokół score — patrz UWAGA przy SCORE_UNCERTAINTY_BASE_PP.
    # Rośnie, gdy quality_penalty/sparse_penalty odchylają się od 1.0 (gorsza
    # jakość sceny / mniej niezależnych rodzin dowodów).
    uncertainty_margin = round(
        SCORE_UNCERTAINTY_BASE_PP / max(quality_penalty * sparse_penalty, 0.1), 1)
    score_low = max(0.0, round(score - uncertainty_margin, 1))
    score_high = min(100.0, round(score + uncertainty_margin, 1))

    return {
        "score":        score,
        "score_low":    score_low,
        "score_high":   score_high,
        "grade":        grade,
        "label":        label,
        "components":   {k: round(v, 4) if v is not None else None
                         for k, v in components.items()},
        "raw_components": {
            k: round(float(v), 4) if v is not None and math.isfinite(float(v)) else None
            for k, v in raw_components.items()
        },
        "normalized_components": {k: round(v, 4) if v is not None else None
                                  for k, v in components.items()},
        "data_quality": round(valid_frac, 3),
        "sar_used":     sar_used,
        "weights_used": {k: round(v, 4) for k, v in weights.items()},
        "effective_weights": effective_weights,
        "season_phase": season_phase,
        "n_available_indicators": n_available,
        "n_available_families": n_available_families,
        "sparse_data":  sparse_penalty < 1.0,
        "red_edge_sources": red_edge_sources,
    }


# ============================================================
# ESTYMACJA BIOMASY
# ============================================================

def estimate_biomass(
    evi:  Optional[float],
    ndre: Optional[float],
    rvi:  Optional[float] = None,
) -> Optional[dict]:
    """
    Estymacja biomasy nadziemnej (sucha masa, t/ha) — SZACUNEK modelowy, nie
    pomiar. Model regresji wielowskaźnikowej (Kross et al. 2015, kukurydza):
        Biomass = -1.2 + 12.5·EVI + 8.3·NDRE + 6.1·RVI

    Współczynniki pochodzą z literatury dla innego typu uprawy (patrz
    komentarz przy BIOMASS_COEFFICIENTS) — bez danych referencyjnych z pola
    (ważenie próbek biomasy poplonu skorelowane z EVI/NDRE/RVI z tego samego
    terminu) nie da się ich uczciwie przekalibrować pod poplon. Dlatego
    funkcja celowo NIE zwraca fałszywie precyzyjnej pojedynczej liczby —
    zwraca punkt środkowy PLUS przedział niepewności (patrz
    BIOMASS_UNCERTAINTY_WITH_RVI/WITHOUT_RVI), a pułap jest ograniczony do
    realistycznego zakresu dla poplonu (BIOMASS_MAX_T_HA), nie dla
    pełnosezonowej uprawy towarowej.

    Gdy RVI niedostępne: człon RVI jest po prostu pomijany, a nie
    dopasowywany na nowo (nie mamy do tego danych) — model w tej sytuacji
    systematycznie zaniża wynik o brakujący wkład tego członu. Rekompensujemy
    to szerszym przedziałem niepewności i etykietą "confidence": "low", a nie
    próbą numerycznej korekty bez podstaw.

    Zwraca None jeśli brak minimalnych danych (EVI i NDRE).
    """
    if evi is None or ndre is None:
        return None

    coef = BIOMASS_COEFFICIENTS
    bm = coef["intercept"] + coef["evi"] * evi + coef["ndre"] * ndre
    has_rvi = rvi is not None
    if has_rvi:
        bm += coef["rvi"] * rvi

    bm = max(0.0, min(BIOMASS_MAX_T_HA, bm))
    at_cap = bm >= BIOMASS_MAX_T_HA - 1e-9

    uncertainty = BIOMASS_UNCERTAINTY_WITH_RVI if has_rvi else BIOMASS_UNCERTAINTY_WITHOUT_RVI
    bm_low  = max(0.0, round(bm * (1 - uncertainty), 2))
    bm_high = min(BIOMASS_MAX_T_HA, round(bm * (1 + uncertainty), 2))

    def _co2(t_ha):
        # Ekwiwalent CO2 — uproszczona sekwestracja węgla.
        # Reguła: ~45% suchej masy = węgiel, mnożnik CO2eq = 44/12 = 3.67.
        # To standardowy przelicznik biologiczny (nie zależy od typu uprawy),
        # więc w przeciwieństwie do współczynników regresji nie wymaga
        # kalibracji polowej.
        return round(t_ha * 0.45 * 3.67, 2)

    model_desc = ("EVI+NDRE+RVI (regresja liniowa)" if has_rvi
                 else "EVI+NDRE (regresja liniowa, bez RVI — pomiar niedostępny)")

    return {
        "biomass_t_ha":       round(bm, 2),
        "biomass_low_t_ha":   bm_low,
        "biomass_high_t_ha":  bm_high,
        "carbon_co2eq_t_ha":      _co2(bm),
        "carbon_co2eq_low_t_ha":  _co2(bm_low),
        "carbon_co2eq_high_t_ha": _co2(bm_high),
        "model":           model_desc,
        "confidence":      "medium" if has_rvi else "low",
        "at_model_cap":    at_cap,
    }


# ============================================================
# DETEKCJA ZDARZEŃ AGROTECHNICZNYCH (fuzja S2+S1)
# ============================================================

def detect_agro_events_fused(
    optical_records: list[dict],   # z fetch_multi_series()
    s1_records: list[dict],        # z sentinel1.fetch_s1_series()
    sowing_date:      Optional[date] = None,
    termination_date: Optional[date] = None,
    weather_frost_dates: Optional[set] = None,   # z weather.detect_frost_dates()
    weather_dry_dates:   Optional[set] = None,   # z weather.detect_dry_spell_dates()
) -> list[dict]:
    """
    Detekcja orki/likwidacji poplonu łącząca sygnały optyczne i radarowe.

    Algorytm 3-stopniowy:

    Krok 1 — Kandydaci optyczni:
      Znajdź dni z gwałtownym spadkiem NDVI (≥ TILLAGE_NDVI_DROP) między
      dwiema KOLEJNYMI obserwacjami odległymi o ≤ 2×TILLAGE_WINDOW_DAYS
      (poprawka z audytu: doprecyzowanie — dawniej ten opis mylnie sugerował
      pojedyncze TILLAGE_WINDOW_DAYS, podczas gdy kod celowo dopuszcza
      dwukrotność tego okna, żeby nie tracić par obserwacji rozdzielonych
      dłuższą przerwą w zachmurzonych okresach; TILLAGE_WINDOW_DAYS samo w
      sobie jest używane wprost w Kroku 4 przy sprawdzaniu, czy optyka
      "miała szansę" zobaczyć zdarzenie widoczne w radarze).

    Krok 2 — Potwierdzenie radarowe:
      Dla każdego kandydata sprawdź czy w oknie ±6 dni istnieje
      obserwacja S1 z skokiem VV ≥ TILLAGE_VV_JUMP_DB.
      Radar nie jest zablokowany przez chmury — potwierdza lub odrzuca.

    Krok 2b — Dodatkowe potwierdzenie optyczne (NDTI, B11/B12):
      Gdy radar NIE potwierdza, sprawdź spadek NDTI między tymi samymi
      dniami co spadek NDVI. NDTI mierzy sygnaturę gleby (odsłonięta gleba/
      orka vs resztki pożniwne), więc jego równoczesny spadek to NIEZALEŻNY
      od NDVI dowód na odsłonięcie gleby (a nie np. sam stres wodny czy
      szkodnik, które obniżają NDVI, ale niekoniecznie zmieniają sygnaturę
      SWIR gleby).

      WAŻNE rozróżnienie (naprawione po audycie): "radar NIE potwierdza" to
      DWIE różne sytuacje, traktowane inaczej:
        - S1 w ogóle niedostępny w oknie (brak sceny) — radar nie miał
          okazji sprawdzić. NDTI może tu bezpiecznie podnieść pewność z
          "medium" do "high", bo uzupełnia realną lukę w danych.
        - S1 dostępny, ale skok VV poniżej progu — radar SPRAWDZIŁ i nie
          zobaczył wystarczającej zmiany. To aktywna sprzeczność, nie luka.
          NDTI w tym przypadku NIE podnosi pewności ponad "medium" (patrz
          s1_contradicts w zwracanym zdarzeniu) — nie chcemy przebijać
          realnego pomiaru radarowego samym drugim wskaźnikiem optycznym
          znad tej samej, potencjalnie niereprezentatywnej sceny S2.

      Nie zastępuje SAR (wciąż ta sama scena S2, więc nie jest odporne na
      chmury), ale podnosi pewność kandydata z "medium" do "high" bez
      konieczności czekania na dopasowaną scenę S1 — tylko gdy scena S1
      faktycznie nie istniała.

    Krok 2c — Kontekst pogodowy (weather_frost_dates/weather_dry_dates,
      patrz weather.py): gdy kandydat NIE MA żadnego niezależnego
      potwierdzenia fizycznego (ani SAR, ani NDTI) — czyli klasyfikuje się
      jako "possible_tillage"/medium z gałęzi s1_contradicts albo zwykłego
      braku danych S1 — a spadek NDVI zbiega się w czasie z przymrozkiem
      lub przedłużającą się posuchą, to istnieje wiarygodne ALTERNATYWNE
      wytłumaczenie (naturalna reakcja rośliny na stres, nie fizyczne
      usunięcie okrywy). Pewność takiego zdarzenia jest wtedy obniżana z
      "medium" do "low" — NIE usuwamy zdarzenia z listy (nadal widoczne dla
      operatora z jawnym wyjaśnieniem), ale app.py filtruje
      `unexpected_events` po `confidence in ("high", "medium")`, więc
      zdarzenie "low" przestaje automatycznie blokować werdykt. Dotyczy
      TYLKO gałęzi bez potwierdzenia fizycznego — potwierdzonej orki (SAR
      lub NDTI) przymrozek nie unieważnia, bo to niezależny, bezpośredni
      pomiar zmiany struktury/gleby, którego przypadkowa pogoda nie tłumaczy.

    Krok 3 — Klasyfikacja zdarzenia:
      - NDVI + SAR → "confirmed_tillage" (pewność: high)
      - NDVI + NDTI, S1 niedostępny → "possible_tillage" (pewność: high)
      - NDVI, S1 dostępny ale nie potwierdza (sprzeczne dowody) →
        "possible_tillage" (pewność: medium, niezależnie od NDTI)
      - Tylko NDVI (brak jakichkolwiek innych danych) → "possible_tillage"
        (pewność: medium)
      - Tylko radarowy → "radar_only_event" (pewność: medium) — patrz Krok 4
      Zdarzenia w oknie ±14 dni od termination_date → "declared_termination"

    Krok 4 — Zdarzenia widoczne WYŁĄCZNIE w radarze:
      Krok 1 wymaga kandydata optycznego jako punktu wejścia, więc gdy S2
      jest w danym okresie całkowicie zasłonięte chmurami (brak użytecznych
      scen), orka nigdy by nie została wykryta — mimo że S1, niewrażliwy na
      chmury, wyraźnie ją widzi. Ten krok szuka skoku VV bezpośrednio w
      szeregu S1 i zgłasza "radar_only_event" TYLKO gdy w oknie
      ±TILLAGE_WINDOW_DAYS wokół daty zdarzenia nie ma ŻADNEJ użytecznej
      sceny optycznej — czyli optyka faktycznie nie miała szansy tego
      zobaczyć. Jeśli optyka miała dane (nawet jeśli nie pokazały spadku),
      zdarzenie NIE jest duplikowane tutaj — to już zostało ocenione (albo
      świadomie pominięte) w Kroku 1.
    """
    events = []

    # Buduj słownik obserwacji S1 po dacie (dla szybkiego lookup)
    s1_by_date = {}
    for r in s1_records:
        d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(str(r["date"]))
        s1_by_date[d] = r

    # Krok 1: kandydaci z NDVI
    for i in range(1, len(optical_records)):
        prev = optical_records[i - 1]
        curr = optical_records[i]

        try:
            d_prev = prev["date"] if isinstance(prev["date"], date) else date.fromisoformat(str(prev["date"]))
            d_curr = curr["date"] if isinstance(curr["date"], date) else date.fromisoformat(str(curr["date"]))
        except (ValueError, KeyError):
            continue

        days_gap = (d_curr - d_prev).days
        if days_gap > TILLAGE_WINDOW_DAYS * 2:
            continue

        ndvi_prev = prev.get("ndvi") or 0
        ndvi_curr = curr.get("ndvi") or 0
        ndvi_drop = ndvi_prev - ndvi_curr

        # Kandydat optyczny — znaczący spadek z wyraźnej wegetacji
        if ndvi_drop < TILLAGE_NDVI_DROP or ndvi_prev < 0.25:
            continue

        # Krok 2: szukaj potwierdzenia w S1.
        # Szukamy najbliższej scysji S1 do d_prev i do d_curr NIEZALEŻNIE
        # (nie wymagamy tego samego przesunięcia "delta" dla obu dat).
        # Wcześniej wymagany był dokładnie ten sam offset dla obu dat
        # jednocześnie, co przy nieregularnym rewizycie S1 (~6-12 dni) i
        # zachmurzonych scenach S2 rzadko się trafiało nawet gdy dobre dane
        # S1 istniały — po prostu na innych, niedopasowanych datach.
        sar_confirmation = None
        vv_jump_found    = 0.0

        def _nearest_s1(target_date, max_delta=6):
            best = None
            best_dist = None
            for delta in range(-max_delta, max_delta + 1):
                cand = s1_by_date.get(target_date + timedelta(days=delta))
                if cand is not None and (best_dist is None or abs(delta) < best_dist):
                    best, best_dist = cand, abs(delta)
            return best

        s1_prev = _nearest_s1(d_prev)
        s1_curr = _nearest_s1(d_curr)
        # Rozróżnienie: "S1 w ogóle niedostępny w oknie" (brak dowodu, optyka
        # nie ma z czym się zderzyć) vs "S1 dostępny, ale skok VV za mały"
        # (radar AKTYWNIE nie potwierdza — to inny stan niż zwykły brak
        # danych, patrz audyt). ndti_confirmation niżej celowo dopuszczamy
        # TYLKO w pierwszym przypadku — NDTI nie powinno przebijać radaru,
        # który faktycznie sprawdził i nie zobaczył wystarczającej zmiany.
        s1_data_available = s1_prev is not None and s1_curr is not None
        if s1_data_available:
            vv_jump = s1_curr.get("vv_db", 0) - s1_prev.get("vv_db", 0)
            if vv_jump >= TILLAGE_VV_JUMP_DB:
                sar_confirmation = s1_curr
                vv_jump_found    = round(vv_jump, 2)

        # Krok 2b: dodatkowe potwierdzenie optyczne przez NDTI (patrz docstring)
        # — używane tylko gdy SAR nie potwierdził I nie miał okazji sprawdzić
        # (brak sceny S1 w oknie), nie gdy sprawdził i aktywnie nie potwierdził.
        ndti_prev = prev.get("ndti")
        ndti_curr = curr.get("ndti")
        ndti_drop = (None if ndti_prev is None or ndti_curr is None
                     else round(ndti_prev - ndti_curr, 3))
        ndti_confirmation = (sar_confirmation is None and not s1_data_available
                             and ndti_drop is not None
                             and ndti_drop >= TILLAGE_NDTI_DROP)
        # S1 był dostępny i sprawdził, ale nie potwierdził — aktywna
        # sprzeczność z ewentualnym spadkiem NDTI, nie zwykły brak dowodu.
        s1_contradicts = (s1_data_available and sar_confirmation is None)

        # Krok 3: klasyfikacja
        within_declared = False
        if termination_date:
            delta_term = abs((d_curr - termination_date).days)
            within_declared = delta_term <= 14

        if sar_confirmation is not None:
            etype = "declared_termination" if within_declared else "confirmed_tillage"
            conf  = "high"
            msg   = (f"Optyczny: NDVI {ndvi_prev:.2f}→{ndvi_curr:.2f} "
                     f"(−{ndvi_drop:.2f}) + SAR: VV +{vv_jump_found} dB. "
                     + ("Zgodne z deklarowaną likwidacją." if within_declared
                        else "⚠ Nieoczekiwana likwidacja — wymaga wyjaśnienia."))
        elif ndti_confirmation:
            etype = "declared_termination" if within_declared else "possible_tillage"
            conf  = "high"
            msg   = (f"Optyczny: NDVI {ndvi_prev:.2f}→{ndvi_curr:.2f} "
                     f"(−{ndvi_drop:.2f}) + NDTI {ndti_prev:.2f}→{ndti_curr:.2f} "
                     f"(−{ndti_drop:.2f}, odsłonięta gleba). Brak sceny S1 w oknie "
                     "(radar nie miał okazji sprawdzić), ale niezależny sygnał "
                     "optyczny (SWIR) "
                     + ("zgodny z deklarowaną likwidacją." if within_declared
                        else "⚠ potwierdza odsłonięcie gleby — wymaga wyjaśnienia."))
        elif s1_contradicts:
            # Radar był dostępny i NIE zobaczył wystarczającego skoku VV —
            # to aktywna sprzeczność, nie luka w danych. NDTI (jeśli w ogóle
            # spadło) jest tu tylko dopiskiem informacyjnym, nie podnosi
            # pewności ponad radar, który faktycznie sprawdził.
            ndti_note = (f" NDTI {ndti_prev:.2f}→{ndti_curr:.2f} (−{ndti_drop:.2f}) "
                        "sugeruje odsłonięcie gleby, ale radar temu przeczy."
                        if ndti_drop is not None and ndti_drop >= TILLAGE_NDTI_DROP
                        else "")
            etype = "declared_termination" if within_declared else "possible_tillage"
            conf  = "medium"
            msg   = (f"Optyczny: NDVI {ndvi_prev:.2f}→{ndvi_curr:.2f} "
                     f"(−{ndvi_drop:.2f}). Radar (S1) sprawdził, ale NIE zobaczył "
                     "wystarczającego skoku VV — sprzeczne dowody."
                     + ndti_note
                     + (" (deklarowana likwidacja)." if within_declared else ""))
        else:
            etype = "declared_termination" if within_declared else "possible_tillage"
            conf  = "medium"
            msg   = (f"Optyczny: NDVI {ndvi_prev:.2f}→{ndvi_curr:.2f} "
                     f"(−{ndvi_drop:.2f}). Brak potwierdzenia radarowego "
                     + ("(deklarowana likwidacja)." if within_declared
                        else "— możliwy stres lub zagubiona scena S1."))

        # Krok 2c: kontekst pogodowy (patrz docstring) — tylko dla gałęzi BEZ
        # niezależnego potwierdzenia fizycznego (SAR/NDTI) i BEZ zgodności z
        # deklarowaną likwidacją (tam obniżanie pewności niczego by nie
        # zmieniło w app.py, tylko myliłoby operatora, patrz audyt dot.
        # analogicznego dead-branchu przy rozbieżności NDVI).
        weather_note = None
        if (conf == "medium" and not within_declared
                and (weather_frost_dates is not None or weather_dry_dates is not None)):
            frost_hit = (weather_frost_dates is not None
                        and (d_curr in weather_frost_dates or d_prev in weather_frost_dates))
            dry_hit = (weather_dry_dates is not None and d_curr in weather_dry_dates)
            if frost_hit or dry_hit:
                reason = "przymrozek" if frost_hit else "przedłużająca się posucha"
                conf = "low"
                weather_note = (
                    f" Pogoda: {reason} w tym oknie — wiarygodne naturalne "
                    "wytłumaczenie spadku NDVI (stres, nie orka), pewność obniżona."
                )
                msg += weather_note

        # Nie alarmuj jeśli siew jeszcze nie nastąpił
        if sowing_date and d_curr < sowing_date:
            continue

        events.append({
            "date":               d_curr.isoformat(),
            "type":               etype,
            "confidence":         conf,
            "ndvi_before":        round(ndvi_prev, 3),
            "ndvi_after":         round(ndvi_curr, 3),
            "ndvi_drop":          round(ndvi_drop, 3),
            "ndti_drop":          ndti_drop,
            "ndti_confirmed":     ndti_confirmation,
            "s1_contradicts":     s1_contradicts,
            "weather_explained":  weather_note is not None,
            "vv_jump_db":         vv_jump_found,
            "sar_confirmed":      sar_confirmation is not None,
            "within_declared":    within_declared,
            "days_gap":           days_gap,
            "message":            msg,
        })

    # Krok 4: zdarzenia widoczne wyłącznie w radarze (patrz docstring wyżej).
    optical_dates_with_data = set()
    for r in optical_records:
        if r.get("ndvi") is not None:
            d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(str(r["date"]))
            optical_dates_with_data.add(d)

    s1_sorted = sorted(s1_by_date.keys())
    for i in range(1, len(s1_sorted)):
        d_prev = s1_sorted[i - 1]
        d_curr = s1_sorted[i]

        days_gap = (d_curr - d_prev).days
        if days_gap > TILLAGE_WINDOW_DAYS * 2:
            continue

        s1_prev_rec = s1_by_date[d_prev]
        s1_curr_rec = s1_by_date[d_curr]
        vv_jump = s1_curr_rec.get("vv_db", 0) - s1_prev_rec.get("vv_db", 0)
        if vv_jump < TILLAGE_VV_JUMP_DB:
            continue

        # Optyka miała szansę to zobaczyć? Jeśli w oknie wokół d_curr istnieje
        # jakakolwiek użyteczna scena S2, nie duplikujemy — to już zostało
        # (albo świadomie nie zostało) zgłoszone w Kroku 1.
        has_optical_nearby = any(
            abs((od - d_curr).days) <= TILLAGE_WINDOW_DAYS for od in optical_dates_with_data
        )
        if has_optical_nearby:
            continue

        if sowing_date and d_curr < sowing_date:
            continue

        within_declared = False
        if termination_date:
            within_declared = abs((d_curr - termination_date).days) <= 14

        etype = "declared_termination" if within_declared else "radar_only_event"
        msg = (f"Tylko SAR: VV +{vv_jump:.1f} dB ({d_prev.isoformat()}→{d_curr.isoformat()}), "
               "brak użytecznych scen optycznych w tym oknie (chmury). "
               + ("Zgodne z deklarowaną likwidacją." if within_declared
                  else "⚠ Niepotwierdzone optycznie — wymaga wyjaśnienia."))

        events.append({
            "date":               d_curr.isoformat(),
            "type":               etype,
            "confidence":         "medium",
            "ndvi_before":        None,
            "ndvi_after":         None,
            "ndvi_drop":          None,
            "vv_jump_db":         round(vv_jump, 2),
            "sar_confirmed":      True,
            "within_declared":    within_declared,
            "days_gap":           days_gap,
            "message":            msg,
        })

    return events


# ============================================================
# ZINTEGROWANA AKTYWNOŚĆ — wybór optymalnego wskaźnika
# ============================================================

def select_primary_index(
    record: dict,
    season_phase: str,   # "early" | "peak" | "winter"
    cloud_frac: float,
    rvi: Optional[float] = None,
) -> dict:
    """
    Wybiera optymalny wskaźnik do wykresu 'Zintegrowanej Aktywności'.

    Logika:
    - Gdy chmury > 50%: użyj RVI (SAR, nie zablokowany)
    - Faza wczesna (< 6 tyg. od siewu): SAVI (mniej wrażliwy na gołą glebę)
    - Faza szczytowa (6-12 tyg.): EVI (lepszy dla gęstej biomasy)
    - Faza zimowa (ozimy): NDRE (monitoring chlorofilu przez przymrozki)
    - Fallback: NDVI

    Zwraca słownik z wybraną wartością i nazwą wskaźnika.
    """
    if cloud_frac > 0.50 and rvi is not None:
        return {"index": "rvi", "value": rvi, "source": "SAR (backup)"}

    phase_map = {
        "early":  "savi",
        "peak":   "evi",
        "winter": "ndre",
    }
    preferred = phase_map.get(season_phase, "ndvi")
    value = record.get(preferred)

    if value is None:
        # Fallback przez kolejność
        for idx in ("evi", "savi", "ndre", "ndvi"):
            if record.get(idx) is not None:
                return {"index": idx, "value": record[idx], "source": "S2"}
        if rvi is not None:
            return {"index": "rvi", "value": rvi, "source": "SAR (fallback)"}
        return {"index": None, "value": None, "source": "brak danych"}

    return {"index": preferred, "value": round(value, 4), "source": "S2"}


def get_season_phase(obs_date: date, sowing_date: Optional[date], typ: str,
                      first_frost_date: Optional[date] = None,
                      cumulative_gdd_since_sowing: Optional[float] = None) -> str:
    """Określa fazę sezonu dla danej obserwacji.

    Sprawdzenie kalendarzowe "winter" (ozimy, grudzień-luty) ma PRIORYTET
    nad klasyfikacją tygodni-od-siewu. Naprawa realnego błędu: dawniej
    kolejność była odwrotna, więc ozimy wysiany później (np. połowa
    października — to legalny termin) miał w styczniu wciąż <16 tygodni od
    siewu i dostawał fazę "peak" zamiast "winter", mimo że kalendarzowo i
    fizjologicznie roślina jest już w spoczynku zimowym (niski kąt słońca,
    przymrozki, ew. śnieg) — dokładnie sytuacja, dla której faza "winter"
    (wagi NDRE/RVI w górę, EVI/SAVI w dół) została zaprojektowana.

    `first_frost_date` (opcjonalne, patrz weather.first_frost_date) pozwala
    uznać fazę "winter" za rozpoczętą WCZEŚNIEJ niż grudzień, gdy realny
    pierwszy przymrozek sezonu w tym roku/regionie wystąpił wcześniej (np.
    wczesny listopad) — sam kalendarz (grudzień-luty) to sztywne przybliżenie
    "typowej" zimy, a fizjologiczny spoczynek rośliny zaczyna się od
    faktycznego mrozu, nie od daty w kalendarzu.

    `cumulative_gdd_since_sowing` (opcjonalne, patrz weather.cumulative_gdd)
    zastępuje klasyfikację "tygodnie od siewu" klasyfikacją opartą na
    STOPNIODNIACH WZROSTU (Growing Degree Days) — chłodny i ciepły sezon
    dają identyczną liczbę "tygodni od siewu", ale roślina jest w zupełnie
    innym stadium fizjologicznym. Progi GDD (weather.GDD_EARLY_TO_PEAK/
    GDD_PEAK_TO_LATE) są wyprowadzone tak, by w PRZECIĘTNYM sezonie dawać
    ten sam wynik co progi tygodniowe — różnica ujawnia się tylko przy
    sezonie wyraźnie cieplejszym/chłodniejszym niż typowy. Gdy podane,
    całkowicie zastępuje klasyfikację tygodniową (nie tylko dla ozimego
    poza kalendarzową zimą, także dla scierniskowego).

    Bez podania first_frost_date/cumulative_gdd_since_sowing zachowanie
    jest identyczne jak dawniej (czysto kalendarzowe/tygodniowe) —
    backward compatible dla wszystkich pozostałych wywołań (patrz
    save_multi_observation w scheduler.py, gdzie dane pogodowe nie są
    dziś jeszcze pobierane — znana, udokumentowana luka do rozszerzenia w
    kolejnym kroku)."""
    if typ == "ozimy":
        if obs_date.month in (12, 1, 2):
            return "winter"
        if first_frost_date is not None and obs_date >= first_frost_date:
            return "winter"
    if cumulative_gdd_since_sowing is not None:
        import weather as _weather
        # Tylko "early"/"peak" (nie ma osobnej fazy "late") — GDD_PEAK_TO_LATE
        # dziś nie zmienia wyniku (peak jest też ostatnią fazą nie-zimową),
        # ale zostaje udokumentowany jako punkt odniesienia na przyszłość
        # (np. gdyby pojawiła się faza "late"/"senescence").
        if cumulative_gdd_since_sowing < _weather.GDD_EARLY_TO_PEAK:
            return "early"
        return "peak"
    if sowing_date:
        weeks = (obs_date - sowing_date).days / 7
        if weeks < 6:
            return "early"
        if weeks < 16:
            return "peak"
    return "peak"


# ============================================================
# SCHEMAT BAZY DANYCH
# ============================================================

ANALYTICS_DB_SCHEMA = """
-- Rozszerzenie tabeli obserwacji o wskaźniki wielowskaźnikowe
-- Dodaje kolumny EVI, SAVI, NDRE, NDWI do istniejących obserwacji
-- Działa przez migrację ALTER TABLE (bezpieczne, nie niszczy danych)

-- Wskaźniki z Sentinel-2 (wielowskaźnikowe — nowa tabela)
CREATE TABLE IF NOT EXISTS multi_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    parcel_db_id     INTEGER NOT NULL,
    obs_date         TEXT NOT NULL,
    -- Wskaźniki optyczne
    ndvi             REAL,
    evi              REAL,
    savi             REAL,
    ndre             REAL,
    ndwi             REAL,
    reip             REAL,      -- Red Edge Inflection Point [nm]; informacyjny,
                                 -- nie wchodzi jeszcze do confidence_score (krok 3 planu)
    ndti             REAL,      -- Normalized Difference Tillage Index (B11/B12);
                                 -- informacyjny, nie wchodzi jeszcze do detect_agro_events_fused
    valid_frac       REAL NOT NULL DEFAULT 0,
    -- Wskaźniki radarowe (S1) — null jeśli niedostępne tego dnia
    vv_db            REAL,
    vh_db            REAL,
    rvi              REAL,
    -- Syntetyczne
    confidence_score REAL,
    primary_index    TEXT,      -- który wskaźnik był użyty jako główny
    primary_value    REAL,
    -- Biomasa
    biomass_t_ha     REAL,
    carbon_co2eq     REAL,
    -- Meta
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE(parcel_db_id, obs_date),
    FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
);

-- Zdarzenia agrotechniczne (połączone S1+S2)
CREATE TABLE IF NOT EXISTS agro_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    parcel_db_id     INTEGER NOT NULL,
    event_date       TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    confidence       TEXT,
    ndvi_before      REAL,
    ndvi_after       REAL,
    ndvi_drop        REAL,
    vv_jump_db       REAL,
    sar_confirmed    INTEGER DEFAULT 0,
    within_declared  INTEGER DEFAULT 0,
    message          TEXT,
    reviewed         INTEGER DEFAULT 0,
    operator_note    TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
);

-- Szybki dostęp do danych wykresu — widok (VIEW)
-- Łączy multi_observations + s1_observations po dacie
-- Używany przez endpoint /api/chart/{parcel_db_id}
-- VIEW chart_data jest tworzone dynamicznie przez init_analytics_tables()
-- po upewnieniu się że tabela s1_observations istnieje.
"""


def init_analytics_tables(db_path: str = "monitoring.db"):
    """Tworzy tabele analityczne. Bezpieczne przy wielokrotnym wywołaniu."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.executescript(ANALYTICS_DB_SCHEMA)
        # Migracja dla baz założonych przed dodaniem REIP/NDTI —
        # CREATE TABLE IF NOT EXISTS nie dotknie już istniejącej tabeli.
        for col in ("reip", "ndti"):
            try:
                conn.execute(f"ALTER TABLE multi_observations ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass  # kolumna już istnieje
        # s1_observations może być tworzona przez sentinel1.py — upewnij się że istnieje
        conn.execute("""
            CREATE TABLE IF NOT EXISTS s1_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parcel_db_id INTEGER NOT NULL,
                obs_date TEXT NOT NULL,
                vv_db REAL, vh_db REAL, vv_lin REAL, vh_lin REAL,
                cross_ratio_db REAL, rvi REAL,
                orbit_direction TEXT DEFAULT 'DESCENDING',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(parcel_db_id, obs_date, orbit_direction),
                FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
            )
        """)
        # Teraz VIEW może być utworzony bezpiecznie
        conn.execute("DROP VIEW IF EXISTS chart_data")
        conn.execute("""
            CREATE VIEW chart_data AS
            SELECT
                m.parcel_db_id, m.obs_date,
                m.ndvi, m.evi, m.savi, m.ndre, m.ndwi, m.reip, m.ndti, m.valid_frac,
                m.confidence_score, m.primary_index, m.primary_value,
                m.biomass_t_ha, m.carbon_co2eq,
                s.vv_db, s.vh_db, s.rvi AS s1_rvi, s.cross_ratio_db
            FROM multi_observations m
            LEFT JOIN s1_observations s
                ON m.parcel_db_id = s.parcel_db_id
                AND m.obs_date    = s.obs_date
                AND s.orbit_direction = 'DESCENDING'
        """)


def save_multi_observation(parcel_db_id: int, rec: dict,
                            rvi: Optional[float] = None,
                            db_path: str = "monitoring.db",
                            sowing_date: Optional[date] = None,
                            typ: Optional[str] = None,
                            termination_date: Optional[date] = None):
    """Zapisuje pełną obserwację wielowskaźnikową do bazy.

    sowing_date/typ/termination_date są opcjonalne (backward-compat) — jeśli
    podane, obserwacja dostaje właściwą fazę fenologiczną (wagi Confidence
    Score + wybór wskaźnika głównego), zamiast domyślnie zakładać "peak"."""
    import sqlite3

    obs_date_val = rec["date"] if isinstance(rec["date"], date) else date.fromisoformat(str(rec["date"]))
    obs_date = obs_date_val.isoformat()

    season_phase = get_season_phase(obs_date_val, sowing_date, typ) if (sowing_date or typ) else "peak"
    near_termination = bool(termination_date) and abs((obs_date_val - termination_date).days) <= 21

    # Oblicz confidence score
    cs = compute_confidence_score(
        ndvi=rec.get("ndvi"), evi=rec.get("evi"),
        savi=rec.get("savi"), ndre=rec.get("ndre"),
        rvi=rvi, ndwi=rec.get("ndwi"), reip=rec.get("reip"),
        valid_frac=rec.get("valid_frac", 1.0),
        season_phase=season_phase, near_termination=near_termination,
    )

    # Estymacja biomasy
    bm = estimate_biomass(rec.get("evi"), rec.get("ndre"), rvi)

    # Wybór wskaźnika głównego — zgodny z fazą użytą do Confidence Score
    primary = select_primary_index(rec, season_phase, 1.0 - rec.get("valid_frac", 1.0), rvi)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO multi_observations
            (parcel_db_id, obs_date, ndvi, evi, savi, ndre, ndwi, reip, ndti, valid_frac,
             rvi, confidence_score, primary_index, primary_value,
             biomass_t_ha, carbon_co2eq)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            parcel_db_id, obs_date,
            rec.get("ndvi"), rec.get("evi"), rec.get("savi"),
            rec.get("ndre"), rec.get("ndwi"), rec.get("reip"), rec.get("ndti"),
            rec.get("valid_frac", 0),
            rvi,
            cs["score"] if cs else None,
            primary["index"], primary["value"],
            bm["biomass_t_ha"] if bm else None,
            bm["carbon_co2eq_t_ha"] if bm else None,
        ))


def get_multi_indicators_near(parcel_db_id: int, obs_date, max_delta_days: int = 5,
                               db_path: str = "monitoring.db") -> dict:
    """Zwraca EVI/SAVI/NDRE/NDWI/RVI/REIP z najbliższej (w oknie max_delta_days)
    obserwacji multi_observations dla danej działki i daty — używane przy
    tworzeniu alarmu (scheduler.py), żeby dołączyć pełny snapshot wskaźników,
    nie tylko NDVI. Bez tego calibration.py nie miałby danych do sugerowania
    korekt wag Confidence Score, a jedynie do progów NDVI.

    REIP jest tu potrzebny z tego samego powodu — bez niego calibration.py
    mogłoby ocenić separację TYLKO surowego NDRE, podczas gdy produkcja od
    kroku 3 planu integracji pasm liczy zblendowany komponent NDRE+REIP
    (patrz blend_red_edge_norm) — bez tego pola sugestie wag dla klucza
    "ndre" systematycznie rozjeżdżałyby się z tym, co faktycznie jest ważone.

    Zwraca {} jeśli tabela nie istnieje jeszcze albo brak dopasowania w oknie."""
    import sqlite3

    target = obs_date if isinstance(obs_date, date) else date.fromisoformat(str(obs_date))
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT obs_date, evi, savi, ndre, ndwi, rvi, reip FROM multi_observations "
                "WHERE parcel_db_id=?", (parcel_db_id,)
            ).fetchall()
    except sqlite3.OperationalError:
        return {}

    best, best_delta = None, None
    for r in rows:
        try:
            d = date.fromisoformat(r["obs_date"])
        except ValueError:
            continue
        delta = abs((d - target).days)
        if delta <= max_delta_days and (best_delta is None or delta < best_delta):
            best, best_delta = r, delta

    if best is None:
        return {}
    return {"evi": best["evi"], "savi": best["savi"], "ndre": best["ndre"],
            "ndwi": best["ndwi"], "rvi": best["rvi"], "reip": best["reip"]}


def get_chart_data(parcel_db_id: int, db_path: str = "monitoring.db") -> list[dict]:
    """
    Pobiera dane do wykresu wielowskaźnikowego.
    Zwraca posortowaną listę słowników — gotową do serialization JSON.
    """
    import sqlite3
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chart_data WHERE parcel_db_id=? ORDER BY obs_date",
                (parcel_db_id,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        # VIEW może nie istnieć przy starszej bazie — fallback
        return []
