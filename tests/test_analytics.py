"""Testy analytics.py: Confidence Score (w tym wagę NDWI), fuzję zdarzeń
SAR (nearest-neighbor matching) oraz wiązanie agro_events -> fused_status
w /api/check_multi."""
from datetime import date
from unittest.mock import patch

import pytest

import analytics


# --- evalscript: skalowanie pasm (regresja "EVI/SAVI zawsze zero") ---

@pytest.mark.parametrize("script_name", ["EVALSCRIPT_MULTI", "EVALSCRIPT_MULTI_RELAXED"])
def test_evalscript_nie_skaluje_pasm_przez_10000(script_name):
    """Regresja realnego błędu: evalscripty dzieliły każde pasmo przez 10000
    ("z DN na reflektancję"), podczas gdy Sentinel Hub zwraca pasma optyczne
    już jako reflektancję [0-1] (tak zakłada działający EVALSCRIPT w core.py).

    Skutek był podstępny, bo dotyczył TYLKO dwóch wskaźników: NDVI/NDRE/NDWI
    to czyste ilorazy, więc skala się w nich skraca i wyglądały poprawnie —
    ale EVI i SAVI mają w mianowniku stałe addytywne (+1.0 oraz L=0.428),
    które nie skalują się razem z pasmami, więc przy pasmach rzędu 3e-5 stała
    dominowała mianownik i oba wskaźniki zawsze wychodziły ~0.0001 → 0.000."""
    script = getattr(analytics, script_name)
    for band in ("B02", "B04", "B05", "B08", "B11"):
        assert f"s.{band} / 10000" not in script.replace("  ", " ")
        assert f"s.{band}/10000" not in script


def test_formuly_evi_savi_daja_sensowne_wartosci_dla_reflektancji():
    """Kontrola liczbowa tych samych wzorów co w evalscripcie: dla typowej
    gęstej okrywy EVI/SAVI muszą wyjść w okolicach progu 'high' z THRESHOLDS,
    a nie ~0. Gdyby ktoś przywrócił dzielenie pasm przez 10000, ten test
    natychmiast pokaże wartości bliskie zeru."""
    NIR, RED, BLUE = 0.35, 0.06, 0.03  # reflektancja: zdrowa, gęsta okrywa

    evi = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0 + 1e-9)
    L = 0.428
    savi = ((NIR - RED) / (NIR + RED + L + 1e-9)) * (1.0 + L)

    # nie zero i nie szum — realne wartości dla rozwiniętego poplonu
    assert 0.25 < evi < 1.0
    assert 0.25 < savi < 1.0
    # i sensownie blisko progów, na których opiera się scoring
    assert evi >= analytics.THRESHOLDS["evi"]["low"]
    assert savi >= analytics.THRESHOLDS["savi"]["high"]


# --- kara za niekompletny zestaw dowodów (regresja z audytu) ---

def test_pojedynczy_wskaznik_nie_daje_juz_score_100():
    """Regresja z audytu: renormalizacja sprawiała, że gdy zmierzony był
    TYLKO JEDEN wskaźnik (reszta None — brak pomiaru, nie: zmierzona i niska
    wartość), ten jeden wskaźnik dostawał 100% wagi i mógł samodzielnie dać
    score=100. To dokładnie ten sam mechanizm 'jeden wskaźnik dominuje',
    przeciwko któremu były skierowane wcześniejsze naprawy w app.py (usunięcie
    nadpisania werdyktu samym NDVI, wybór dnia po score, kontrola odrostu po
    incydencie) — bez tej łaty mógł wrócić przez sam brak danych."""
    r = analytics.compute_confidence_score(
        ndvi=0.75, evi=None, savi=None, ndre=None, ndwi=None, rvi=None,
        valid_frac=1.0, season_phase="peak")
    assert r["n_available_indicators"] == 1
    assert r["sparse_data"] is True
    assert r["score"] < 70  # nie ma prawa osiągnąć progu OBECNY na 1 wskaźniku


def test_dwa_wskazniki_z_roznych_rodzin_karane_lagodniej_niz_jeden():
    """'Drugi dowód' musi pochodzić z NIEZALEŻNEJ rodziny (patrz
    INDICATOR_FAMILY), żeby coś realnie zmienić — tu NDVI (rodzina nir_red)
    + NDWI (rodzina ndwi) to dwie niezależne rodziny, więc kara powinna
    złagodnieć względem samego NDVI."""
    r1 = analytics.compute_confidence_score(
        ndvi=0.75, evi=None, savi=None, ndre=None, ndwi=None, rvi=None,
        valid_frac=1.0, season_phase="peak")
    r2 = analytics.compute_confidence_score(
        ndvi=0.75, evi=None, savi=None, ndre=None, ndwi=0.35, rvi=None,
        valid_frac=1.0, season_phase="peak")
    assert r1["n_available_families"] == 1
    assert r2["n_available_families"] == 2
    assert r2["score"] > r1["score"]


def test_dwa_wskazniki_tej_samej_rodziny_nie_daja_dodatkowej_ulgi():
    """Regresja z audytu: NDVI i EVI to praktycznie dwa przekształcenia tej
    samej osi NIR-RED (te same natywnie 10m pasma) — 'drugi' wskaźnik z TEJ
    SAMEJ rodziny nie może obniżyć kary tak, jakby to był realnie niezależny
    dowód. n_available_indicators rośnie z 1 do 2, ale n_available_families
    zostaje na 1 — sparse_penalty (a więc i jego wpływ na wynik) ma być taki
    sam jak dla samego NDVI."""
    r1 = analytics.compute_confidence_score(
        ndvi=0.75, evi=None, savi=None, ndre=None, ndwi=None, rvi=None,
        valid_frac=1.0, season_phase="peak")
    r2 = analytics.compute_confidence_score(
        ndvi=0.75, evi=0.75, savi=None, ndre=None, ndwi=None, rvi=None,
        valid_frac=1.0, season_phase="peak")
    assert r1["n_available_indicators"] == 1
    assert r2["n_available_indicators"] == 2
    assert r1["n_available_families"] == 1
    assert r2["n_available_families"] == 1
    assert r1["sparse_data"] is True
    assert r2["sparse_data"] is True


def test_ndvi_evi_savi_same_nie_wystarczaja_do_pelnego_zaufania():
    """Kluczowa regresja z audytu: NDVI+EVI+SAVI (3 SUROWE komponenty, ale
    jedna rodzina — ta sama para pasm B04/B08, +B02 dla EVI) dotąd
    trywialnie spełniały próg 'pełnego zaufania' (n_available>=3), mimo że
    to jedna oś informacji, nie trzy niezależne dowody. Realny scenariusz:
    natywnie 20m pasma (B05/B06/B07/B11/B12, potrzebne do NDRE/NDWI/REIP/
    NDTI) nie zwracają danych przy granicy działki, podczas gdy natywnie 10m
    pasma (B02/B04/B08) tak — wtedy NDVI/EVI/SAVI są dostępne, reszta nie."""
    r = analytics.compute_confidence_score(
        ndvi=0.75, evi=0.75, savi=0.75, ndre=None, ndwi=None, rvi=None,
        valid_frac=1.0, season_phase="peak")
    assert r["n_available_indicators"] == 3
    assert r["n_available_families"] == 1
    assert r["sparse_data"] is True


def test_trzy_i_wiecej_wskaznikow_bez_kary():
    """Próg pełnego zaufania (3 NIEZALEŻNE RODZINY) — realne, dobrze
    umeblowane dane (jak w praktycznych przypadkach z tej sesji) nie tracą
    punktów. Tu 6 surowych komponentów obejmuje 4 rodziny (nir_red,
    red_edge, ndwi, radar) — powyżej progu."""
    full = dict(ndvi=0.705, evi=0.515, savi=0.486, ndre=0.410, ndwi=0.441, rvi=0.691)
    r = analytics.compute_confidence_score(**full, valid_frac=1.0, season_phase="peak")
    assert r["n_available_indicators"] == 6
    assert r["n_available_families"] == 4
    assert r["sparse_data"] is False


def test_sam_rvi_jako_celowy_fallback_radarowy_nie_jest_karany():
    """sar_used (cała optyka niedostępna, tylko radar) to świadomie wspierana
    ścieżka jednowskaźnikowa — nie ta sama sytuacja co przypadkowy brak
    ekstrakcji jednego pasma. Nie powinna być tłumiona przez sparse_penalty."""
    r = analytics.compute_confidence_score(
        ndvi=None, evi=None, savi=None, ndre=None, ndwi=None, rvi=0.69,
        valid_frac=1.0, season_phase="peak")
    assert r["sar_used"] is True
    assert r["sparse_data"] is False
    assert r["score"] == 100.0


def test_ndwi_wplywa_na_confidence_score():
    base = dict(ndvi=0.45, evi=0.35, savi=0.30, ndre=0.25, rvi=0.20, valid_frac=1.0)
    cs_bez = analytics.compute_confidence_score(**base)
    cs_wysokie = analytics.compute_confidence_score(**base, ndwi=0.40)
    cs_niskie = analytics.compute_confidence_score(**base, ndwi=0.00)

    assert cs_wysokie["score"] != cs_bez["score"]
    assert cs_wysokie["score"] > cs_niskie["score"]
    assert cs_bez["raw_components"]["ndwi"] is None
    assert cs_wysokie["raw_components"]["ndwi"] == 0.40


def test_confidence_score_dziala_bez_ndwi_backward_compat():
    cs = analytics.compute_confidence_score(ndvi=0.5, evi=0.4)
    assert cs["score"] is not None


def test_wagi_zaleza_od_fazy_fenologicznej():
    """Naprawa: dawniej jeden stały zestaw wag przez cały sezon, mimo że
    select_primary_index() już wie, że SAVI jest lepszy wcześnie, EVI w
    szczycie, NDRE zimą. Ten sam rekord powinien dać inny score zależnie od
    podanej season_phase, bo wagi się różnią."""
    kwargs = dict(ndvi=0.35, evi=0.20, savi=0.45, ndre=0.30, rvi=0.15, ndwi=0.20)
    cs_early = analytics.compute_confidence_score(**kwargs, season_phase="early")
    cs_peak = analytics.compute_confidence_score(**kwargs, season_phase="peak")
    cs_winter = analytics.compute_confidence_score(**kwargs, season_phase="winter")

    assert cs_early["score"] != cs_peak["score"]
    assert cs_peak["score"] != cs_winter["score"]
    assert cs_early["weights_used"]["savi"] > cs_peak["weights_used"]["savi"]
    assert cs_peak["weights_used"]["evi"] > cs_early["weights_used"]["evi"]
    assert cs_winter["weights_used"]["ndre"] > cs_peak["weights_used"]["ndre"]
    assert cs_early["season_phase"] == "early"


def test_wagi_domyslnie_peak_gdy_faza_nieznana():
    cs = analytics.compute_confidence_score(ndvi=0.5, evi=0.4, season_phase="nonsense")
    assert cs["weights_used"] == pytest.approx(analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"])


def test_ndwi_waga_rosnie_blisko_terminu_likwidacji():
    """Kontekstowa waga NDWI: blisko terminu likwidacji różnica sucha
    ściernia/żywa okrywa ma największe znaczenie -> waga NDWI powinna
    wzrosnąć (i wynik score też się zmienić przy tych samych wskaźnikach)."""
    kwargs = dict(ndvi=0.35, evi=0.30, savi=0.25, ndre=0.20, rvi=0.15, ndwi=0.35,
                  season_phase="peak")
    cs_normal = analytics.compute_confidence_score(**kwargs, near_termination=False)
    cs_near = analytics.compute_confidence_score(**kwargs, near_termination=True)

    assert cs_near["weights_used"]["ndwi"] > cs_normal["weights_used"]["ndwi"]
    assert cs_near["weights_used"]["ndwi"] <= analytics.NDWI_MAX_WEIGHT + 1e-9
    assert cs_near["score"] != cs_normal["score"]


def test_get_season_phase_zima_ma_priorytet_nad_tygodniami_od_siewu():
    """Naprawa: dawniej ozimy wysiany późno (np. 15.10 — legalny termin) miał
    w styczniu wciąż <16 tygodni od siewu, więc dostawał fazę 'peak' zamiast
    'winter', mimo że kalendarzowo to głęboka zima. Kalendarz ma teraz
    priorytet nad tygodniami-od-siewu dla ozimego."""
    sowing = date(2026, 10, 15)
    obs_january = date(2027, 1, 10)  # ~12.6 tygodnia od siewu -> < 16
    assert analytics.get_season_phase(obs_january, sowing, "ozimy") == "winter"

    # ozimy wysiany wcześnie (sierpień) też powinien dostać "winter" w styczniu
    early_sowing = date(2026, 8, 1)
    assert analytics.get_season_phase(obs_january, early_sowing, "ozimy") == "winter"

    # scierniskowy nie ma fazy "winter" -> kalendarz nie powinien nic zmieniać
    assert analytics.get_season_phase(obs_january, sowing, "scierniskowy") == "peak"

    # poza grudniem-lutym nadal działa klasyfikacja tygodniowa
    obs_early = date(2026, 11, 1)  # ~2.4 tyg. od siewu 15.10 -> early
    assert analytics.get_season_phase(obs_early, sowing, "ozimy") == "early"


def test_get_season_phase_first_frost_date_przyspiesza_faze_winter():
    """Nowa funkcjonalność: first_frost_date (z weather.first_frost_date)
    pozwala uznać fazę 'winter' za rozpoczętą wcześniej niż grudzień, gdy
    realny pierwszy przymrozek sezonu wystąpił wcześniej (np. wczesny
    listopad). Bez podania first_frost_date zachowanie jest identyczne jak
    dawniej (czysto kalendarzowe)."""
    sowing = date(2026, 8, 15)
    obs_november = date(2026, 11, 10)  # ~12.4 tyg. od siewu -> "peak" kalendarzowo/tygodniowo

    # bez first_frost_date -> stare, czysto kalendarzowe zachowanie (listopad -> "peak")
    assert analytics.get_season_phase(obs_november, sowing, "ozimy") == "peak"

    # z first_frost_date PRZED datą obserwacji -> "winter" mimo że to jeszcze nie grudzień
    assert analytics.get_season_phase(
        obs_november, sowing, "ozimy", first_frost_date=date(2026, 11, 1)) == "winter"

    # z first_frost_date PO dacie obserwacji -> przymrozek jeszcze nie nastąpił, zostaje "peak"
    assert analytics.get_season_phase(
        obs_november, sowing, "ozimy", first_frost_date=date(2026, 11, 20)) == "peak"

    # scierniskowy nie ma fazy "winter" -> first_frost_date nie powinien nic zmieniać
    assert analytics.get_season_phase(
        obs_november, sowing, "scierniskowy", first_frost_date=date(2026, 11, 1)) == "peak"


def test_get_season_phase_gdd_zastepuje_klasyfikacje_tygodniowa():
    """Nowa funkcjonalność: cumulative_gdd_since_sowing (z weather.cumulative_gdd)
    zastępuje klasyfikację 'tygodnie od siewu' — chłodny i ciepły sezon dają
    identyczną liczbę tygodni od siewu, ale różną liczbę GDD. Bez podania
    tego argumentu zachowanie jest identyczne jak dawniej (czysto
    tygodniowe)."""
    import weather
    sowing = date(2026, 8, 1)
    obs = date(2026, 9, 12)  # dokładnie 6 tygodni od siewu

    # bez GDD -> stare, czysto tygodniowe zachowanie: dokładnie 6 tyg. -> "peak"
    assert analytics.get_season_phase(obs, sowing, "scierniskowy") == "peak"

    # z GDD poniżej progu (sezon chłodniejszy niż przeciętny) -> nadal "early"
    # mimo że kalendarzowo minęło już 6 tygodni
    assert analytics.get_season_phase(
        obs, sowing, "scierniskowy",
        cumulative_gdd_since_sowing=weather.GDD_EARLY_TO_PEAK - 50) == "early"

    # z GDD powyżej progu -> "peak"
    assert analytics.get_season_phase(
        obs, sowing, "scierniskowy",
        cumulative_gdd_since_sowing=weather.GDD_EARLY_TO_PEAK + 50) == "peak"


def test_effective_weights_sumuja_sie_do_jednego_i_sa_renormalizowane():
    """effective_weights (w odróżnieniu od weights_used) powinno pokazywać
    realny udział KAŻDEGO DOSTĘPNEGO wskaźnika po renormalizacji — sumuje się
    do 1.0, nawet gdy część wskaźników fazy jest nieobecna w obserwacji."""
    cs = analytics.compute_confidence_score(ndvi=0.4, evi=0.3, season_phase="peak")
    assert set(cs["effective_weights"].keys()) == {"ndvi", "evi"}
    assert sum(cs["effective_weights"].values()) == pytest.approx(1.0, abs=0.001)
    # proporcja między ndvi i evi w effective_weights odpowiada proporcji wag fazy
    phase_w = analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]
    assert cs["effective_weights"]["ndvi"] / cs["effective_weights"]["evi"] == \
        pytest.approx(phase_w["ndvi"] / phase_w["evi"], rel=0.01)


def test_ndvi_norm_bounds_sledzi_na_zywo_core_ndvi_cover_high():
    """Regresja z audytu: THRESHOLDS["ndvi"]["high"] było zamrożoną kopią
    core.DEFAULT_NDVI_COVER_HIGH (0.42) bez żadnego mechanizmu przeładowania
    — po zatwierdzeniu kalibracji NDVI przez operatora (core.NDVI_COVER_HIGH
    zmienia się na żywo przez core.reload_thresholds()) normalizacja NDVI w
    compute_confidence_score cicho zostawałaby przy starym progu, mimo że
    core.assess() i kontrola rozbieżności NDVI w app.py już liczą względem
    nowego. ndvi_norm_bounds() musi czytać core.NDVI_COVER_HIGH NA ŻYWO."""
    import core
    old_high = core.NDVI_COVER_HIGH
    try:
        # Ten sam surowy NDVI z domyślnym progiem (0.42) daje score < 100
        # (bo (0.35-0.15)/(0.42-0.15) < 1.0)...
        cs_default = analytics.compute_confidence_score(ndvi=0.35, season_phase="peak")
        assert cs_default["normalized_components"]["ndvi"] < 1.0

        # ...ale po obniżeniu progu przez kalibrację (np. do 0.35) TEN SAM
        # surowy NDVI powinien normalizować się do 1.0 (pełna okrywa wg
        # nowego, zatwierdzonego progu) — jeśli funkcja czytałaby wciąż
        # zamrożoną wartość 0.42, wynik by się nie zmienił.
        core.NDVI_COVER_HIGH = 0.35
        cs_recalibrated = analytics.compute_confidence_score(ndvi=0.35, season_phase="peak")
        assert cs_recalibrated["normalized_components"]["ndvi"] == pytest.approx(1.0, abs=0.01)
        assert cs_recalibrated["normalized_components"]["ndvi"] > cs_default["normalized_components"]["ndvi"]

        # calibration.py musi oceniać separację NDVI względem TEGO SAMEGO,
        # aktualnego progu — nie osobnej, potencjalnie rozjeżdżającej się kopii.
        assert analytics.ndvi_norm_bounds() == (analytics.THRESHOLDS["ndvi"]["bare"], 0.35)
    finally:
        core.NDVI_COVER_HIGH = old_high


def test_reload_weights_wczytuje_nadpisanie_z_bazy(isolated_db):
    """reload_weights() powinien nałożyć nadpisanie z tabeli settings na
    domyślne wagi per-fazowe, tak jak core.reload_thresholds() dla progów."""
    import json
    import db as dbmod
    dbmod.init_db()
    dbmod.set_setting("confidence_weights_override",
                      json.dumps({"peak": {"evi": 0.40}}))
    analytics.reload_weights()
    try:
        assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]["evi"] == pytest.approx(0.40)
        # inne wskaźniki tej fazy i pozostałe fazy nie powinny się zmienić
        assert analytics.CONFIDENCE_WEIGHTS_BY_PHASE["early"] == \
            analytics.DEFAULT_CONFIDENCE_WEIGHTS_BY_PHASE["early"]
    finally:
        analytics.reload_weights()  # bez nadpisania -> wraca do wartości domyślnych


def test_agro_events_fused_nearest_neighbor_potwierdza_mimo_roznych_offsetow():
    """Naprawa: dawniej wymagany był identyczny offset S1 dla obu dat
    (prev/curr), co przy nieregularnym rewizycie S1 rzadko się trafiało.
    Teraz szukamy najbliższej sceny S1 niezależnie dla każdej z dat."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55},
        {"date": date(2026, 9, 8), "ndvi": 0.10},
    ]
    s1 = [
        {"date": date(2026, 9, 2), "vv_db": -12.0},
        {"date": date(2026, 9, 10), "vv_db": -4.0},
    ]
    events = analytics.detect_agro_events_fused(optical, s1, sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["sar_confirmed"] is True
    assert events[0]["type"] == "confirmed_tillage"


def test_agro_events_fused_wykrywa_zdarzenie_tylko_radarowe():
    """Naprawa realnej luki: gdy S2 jest całkowicie zasłonięte chmurami
    (brak jakichkolwiek użytecznych scen optycznych w oknie), ale S1
    wyraźnie widzi skok VV, funkcja MUSI zgłosić 'radar_only_event' —
    wcześniej takie zdarzenie nigdy nie było wykrywane (Krok 1 wymagał
    kandydata optycznego jako punktu wejścia)."""
    optical = []  # brak jakichkolwiek scen S2 w tym okresie
    s1 = [
        {"date": date(2026, 9, 1), "vv_db": -16.0},
        {"date": date(2026, 9, 8), "vv_db": -12.0},  # skok +4 dB
    ]
    events = analytics.detect_agro_events_fused(optical, s1, sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["type"] == "radar_only_event"
    assert events[0]["confidence"] == "medium"
    assert events[0]["sar_confirmed"] is True
    assert events[0]["vv_jump_db"] == 4.0


def test_agro_events_fused_nie_duplikuje_gdy_optyka_miala_dane():
    """Jeśli w oknie wokół daty S1 istnieje jakakolwiek użyteczna scena
    optyczna, nie generujemy 'radar_only_event' — to już zostało (albo
    świadomie nie zostało) ocenione w Kroku 1, nie chcemy duplikatu."""
    optical = [
        {"date": date(2026, 9, 3), "ndvi": 0.40},  # scena istnieje blisko okna S1
    ]
    s1 = [
        {"date": date(2026, 9, 1), "vv_db": -16.0},
        {"date": date(2026, 9, 8), "vv_db": -12.0},
    ]
    events = analytics.detect_agro_events_fused(optical, s1, sowing_date=None, termination_date=None)
    assert not any(e["type"] == "radar_only_event" for e in events)


def test_agro_events_fused_radar_only_respektuje_declared_termination():
    optical = []
    s1 = [
        {"date": date(2026, 9, 1), "vv_db": -16.0},
        {"date": date(2026, 9, 8), "vv_db": -12.0},
    ]
    events = analytics.detect_agro_events_fused(
        optical, s1, sowing_date=None, termination_date=date(2026, 9, 8))
    assert len(events) == 1
    assert events[0]["type"] == "declared_termination"
    assert events[0]["within_declared"] is True


def test_agro_events_fused_bez_s1_nie_wywala_sie():
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55},
        {"date": date(2026, 9, 8), "ndvi": 0.10},
    ]
    events = analytics.detect_agro_events_fused(optical, [], sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["sar_confirmed"] is False
    assert events[0]["type"] == "possible_tillage"


# --- wiązanie agro_events -> fused_status w /api/check_multi ---

def _peak_record(rok, typ="scierniskowy"):
    import core
    cover_w = core.CFG[typ]["cover"](rok)
    return {"date": cover_w[0], "ndvi": 0.55, "evi": 0.45, "savi": 0.40,
            "ndre": 0.30, "ndwi": 0.30, "valid_frac": 1.0}


def _run_check_multi(app_client, agro_events, rok=2026, typ="scierniskowy"):
    record = _peak_record(rok, typ)
    with patch("analytics.fetch_multi_series", return_value=[record]), \
         patch("analytics.detect_agro_events_fused", return_value=agro_events), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", return_value=[]), \
         patch("terrain.fetch_terrain_context", return_value=None):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": typ, "rok": rok,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    return r.json()["fused_verdict"]


def _run_check_multi_full(app_client, records, rok=2026, typ="scierniskowy"):
    """Jak _run_check_multi, ale zwraca CAŁĄ odpowiedź (peak_record, confidence...)
    i pozwala podać własny zestaw dni wejściowych."""
    with patch("analytics.fetch_multi_series", return_value=records), \
         patch("analytics.detect_agro_events_fused", return_value=[]), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", return_value=[]), \
         patch("terrain.fetch_terrain_context", return_value=None):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": typ, "rok": rok,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    return r.json()


def test_check_multi_wybiera_dzien_po_wyniku_wielowskaznikowym_nie_po_ndvi(app_client):
    """Regresja: dzień reprezentatywny dla okna oceny był wybierany WYŁĄCZNIE
    po najwyższym NDVI (max(cover_recs, key=ndvi)), a dopiero potem dla tego
    dnia doliczano EVI/SAVI/NDRE/NDWI/RVI. Pozostałe wskaźniki nie miały więc
    żadnego wpływu na to, KTÓRY dzień jest w ogóle oceniany — co jest tym
    gorsze, że NDVI saturuje na gęstej biomasie (po to istnieje EVI)."""
    import core
    from datetime import timedelta
    cover_w = core.CFG["scierniskowy"]["cover"](2026)

    # Dzień A: najwyższe NDVI, ale wszystkie pozostałe wskaźniki na poziomie
    # gołej gleby -> słabe łączne świadectwo okrywy (score ~31).
    rec_hi_ndvi = {"date": cover_w[0], "ndvi": 0.70, "evi": 0.10, "savi": 0.08,
                   "ndre": 0.08, "ndwi": 0.00, "valid_frac": 1.0}
    # Dzień B: NIŻSZE NDVI, ale wszystkie pozostałe wskaźniki świetne
    # -> mocne łączne świadectwo okrywy (score ~100).
    rec_best_multi = {"date": cover_w[0] + timedelta(days=5), "ndvi": 0.50, "evi": 0.50,
                      "savi": 0.40, "ndre": 0.45, "ndwi": 0.40, "valid_frac": 1.0}

    data = _run_check_multi_full(app_client, [rec_hi_ndvi, rec_best_multi])

    # Stara logika wybrałaby dzień A (NDVI 0.70 > 0.50). Nowa musi wybrać B.
    assert data["peak_record"]["date"] == rec_best_multi["date"].isoformat()
    assert data["peak_record"]["ndvi"] == 0.50
    assert data["confidence"]["score"] > 90


def test_check_multi_slaba_jakosc_o_wyzszym_score_nie_wygrywa_z_wiarygodna_scena(app_client):
    """Regresja z audytu: kara za jakość w samym score jest łagodna (max 7.5%
    przy dolnej granicy valid_frac=0.35), więc scena słabej jakości z akurat
    korzystnymi odczytami mogła wygrać wybór dnia z solidniejszą, ale gorzej
    zachmurzoną sceną — dokładnie ten sam mechanizm 'jeden wynik dominuje',
    przeciwko któremu skierowane były wcześniejsze naprawy. Zweryfikowane
    liczbowo: dzień słabej jakości (valid_frac=0.40) potrafi dać WYŻSZY
    surowy score niż solidny dzień (valid_frac=1.0) — mimo to wybór ma paść
    na dzień wiarygodny."""
    import core
    from datetime import timedelta
    cover_w = core.CFG["scierniskowy"]["cover"](2026)

    reliable_day = {"date": cover_w[0], "ndvi": 0.35, "evi": 0.30, "savi": 0.25,
                     "ndre": 0.25, "ndwi": 0.20, "valid_frac": 1.0}
    low_quality_day = {"date": cover_w[0] + timedelta(days=5), "ndvi": 0.55, "evi": 0.45,
                        "savi": 0.40, "ndre": 0.35, "ndwi": 0.30, "valid_frac": 0.40}

    data = _run_check_multi_full(app_client, [reliable_day, low_quality_day])

    # Kontrola założenia testu: dzień słabej jakości MUSI mieć wyższy surowy
    # score niż wiarygodny, inaczej test niczego nie sprawdza.
    import analytics as A
    score_reliable = A.compute_confidence_score(
        ndvi=0.35, evi=0.30, savi=0.25, ndre=0.25, ndwi=0.20, rvi=None,
        valid_frac=1.0, season_phase="peak")["score"]
    score_low_q = A.compute_confidence_score(
        ndvi=0.55, evi=0.45, savi=0.40, ndre=0.35, ndwi=0.30, rvi=None,
        valid_frac=0.40, season_phase="peak")["score"]
    assert score_low_q > score_reliable

    # Mimo to wybrany ma zostać dzień WIARYGODNY, nie ten o wyższym score.
    assert data["peak_record"]["date"] == reliable_day["date"].isoformat()
    assert data["peak_record"]["valid_frac"] == 1.0
    # Przejrzystość: odrzucony, wyższy wynik ma być widoczny jako alternatywa.
    assert data["confidence"]["best_unfiltered_score"] == pytest.approx(score_low_q)
    assert data["confidence"]["best_unfiltered_date"] == low_quality_day["date"].isoformat()


def test_check_multi_wysokie_ndvi_nie_wymusza_juz_statusu_obecny(app_client):
    """Regresja: NDVI miało DRUGI, uprzywilejowany głos — najpierw jako
    składnik o największej wadze w Confidence Score, a potem jeszcze raz jako
    nadpisanie werdyktu ('ndvi >= NDVI_COVER_HIGH and score >= 50 -> wymuś
    OBECNY'). Działka, którą wszystkie sześć wskaźników razem oceniało na
    'prawdopodobny' (score 50-69), była awansowana do OBECNY wyłącznie dlatego,
    że NDVI przekroczyło próg."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    # score wychodzi ~60 (przedział, w którym stare nadpisanie wymuszało OBECNY)
    rec = {"date": cover_w[0], "ndvi": 0.55, "evi": 0.268, "savi": 0.214,
           "ndre": 0.235, "ndwi": 0.168, "valid_frac": 1.0}

    data = _run_check_multi_full(app_client, [rec])
    score = data["confidence"]["score"]

    # Warunki starego nadpisania są spełnione...
    assert rec["ndvi"] >= core.NDVI_COVER_HIGH
    assert 50 <= score < 70
    # ...ale werdykt ma teraz wynikać z oceny wielowskaźnikowej, nie z NDVI.
    assert data["fused_verdict"]["status"] == "PRAWDOPODOBNY"


def test_grade_i_fused_status_uzywaja_tej_samej_stalej_progow(app_client):
    """Regresja z audytu: grade (compute_confidence_score) i fused_status
    (app.py) dawniej kodowały 70/50/30/15 NIEZALEŻNIE — mogły się rozjechać,
    gdyby ktoś zmienił próg tylko w jednym miejscu. Teraz oba czytają
    analytics.SCORE_THRESHOLDS na żywo, więc przesunięcie progu 'A' musi
    spójnie przesunąć ZARÓWNO grade, JAK I fused_status/kategorię."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    # score dla tego rekordu wychodzi w okolicach 54% — poniżej domyślnego
    # progu A=70 (więc domyślnie grade B / status PRAWDOPODOBNY)
    rec = {"date": cover_w[0], "ndvi": 0.33, "evi": 0.31, "savi": 0.25,
           "ndre": 0.22, "ndwi": 0.22, "valid_frac": 1.0}

    data_default = _run_check_multi_full(app_client, [rec])
    score = data_default["confidence"]["score"]
    assert 50 <= score < 70  # kontrola założenia testu
    assert data_default["confidence"]["grade"] == "B"
    assert data_default["fused_verdict"]["status"] == "PRAWDOPODOBNY"

    lowered = dict(analytics.SCORE_THRESHOLDS)
    lowered["A"] = int(score) - 1  # próg A poniżej naszego score
    with patch.dict(analytics.SCORE_THRESHOLDS, lowered, clear=True):
        data_lowered = _run_check_multi_full(app_client, [rec])

    # grade i fused_status muszą przesunąć się RAZEM, spójnie
    assert data_lowered["confidence"]["grade"] == "A"
    assert data_lowered["fused_verdict"]["status"] == "OBECNY"


def test_check_multi_rozbieznosc_ndvi_vs_reszta_kieruje_do_operatora(app_client):
    """Zamiast pozwalać NDVI wygrać z pozostałymi pięcioma wskaźnikami, silna
    rozbieżność ma być traktowana jak to, czym jest — sygnałem niepewności
    kierującym sprawę do operatora."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    rec = {"date": cover_w[0], "ndvi": 0.70, "evi": 0.10, "savi": 0.08,
           "ndre": 0.08, "ndwi": 0.00, "valid_frac": 1.0}

    data = _run_check_multi_full(app_client, [rec])
    fv = data["fused_verdict"]

    assert rec["ndvi"] >= core.NDVI_COVER_HIGH   # NDVI mówi "okrywa"
    assert data["confidence"]["score"] < 50       # reszta wskaźników mówi inaczej
    assert fv["requires_operator_approval"] is True
    assert any("Rozbieżność" in d for d in fv["details"])


def test_check_multi_rozbieznosc_ndvi_nie_pojawia_sie_gdy_status_juz_brak(app_client):
    """Regresja z audytu #3: gdy score jest tak niski, że fused_status i tak
    jest już wymuszony na 'BRAK' (categorize_verdict() zawsze da czerwony dla
    status=='BRAK', niezależnie od requires_operator_approval), komunikat
    'Rozbieżność NDVI vs wynik łączny' nie może się pojawiać — sugerowałby
    operatorowi, że to ta rozbieżność zdecydowała o kategorii, podczas gdy
    faktycznie nie miała żadnego wpływu. W fazie 'early' waga NDVI jest niższa
    niż w 'peak', więc nawet przy NDVI >= NDVI_COVER_HIGH score może spaść
    poniżej progu C (30) -> status BRAK."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    rec = {"date": cover_w[0], "ndvi": 0.45, "evi": 0.0, "savi": 0.0,
           "ndre": 0.0, "ndwi": 0.0, "valid_frac": 1.0}

    with patch("analytics.fetch_multi_series", return_value=[rec]), \
         patch("analytics.detect_agro_events_fused", return_value=[]), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="early"), \
         patch("weather.fetch_weather_series", return_value=[]), \
         patch("terrain.fetch_terrain_context", return_value=None):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    data = r.json()
    fv = data["fused_verdict"]

    assert rec["ndvi"] >= core.NDVI_COVER_HIGH   # NDVI nadal mówi "okrywa"
    assert data["confidence"]["score"] < 30       # ale score < próg C -> BRAK
    assert fv["status"] == "BRAK"
    assert not any("Rozbieżność" in d for d in fv["details"])
    assert not any("Rozbieżność" in r for r in data.get("category_reasons", []))


def _quality_scenes(rok, vfs, typ="scierniskowy"):
    """Zestaw dni w oknie oceny o zadanych valid_frac, wszystkie ze świetnymi
    wskaźnikami (żeby jedyną zmienną była jakość danych)."""
    import core
    from datetime import timedelta
    cover_w = core.CFG[typ]["cover"](rok)
    return [{"date": cover_w[0] + timedelta(days=5 * i), "ndvi": 0.705, "evi": 0.515,
             "savi": 0.486, "ndre": 0.410, "ndwi": 0.441, "valid_frac": vf}
            for i, vf in enumerate(vfs)]


def test_check_multi_pojedyncza_slabsza_scena_nie_blokuje_zatwierdzenia(app_client):
    """Regresja realnego objawu: Confidence Score 94 (ocena A, 'poplon
    potwierdzony') trafiał do kategorii żółtej zamiast zielonej, bo JEDNA
    częściowo zachmurzona scena bezwarunkowo ustawiała
    requires_operator_approval.

    Były w tym dwa błędy: (1) podwójne karanie za to samo —
    compute_confidence_score() już obniża wynik proporcjonalnie do valid_frac,
    więc score 94 to wynik JUŻ uwzględniający zachmurzenie, a mimo to ta sama
    informacja działała drugi raz jako weto; (2) niespójność — ścieżka NDVI
    (core.assess) rozstrzyga to mądrzej: zatwierdzenia wymaga tylko wynik
    niejednoznaczny albo przewaga słabych scen."""
    recs = _quality_scenes(2026, [0.95, 0.90, 0.45, 0.88, 0.92])
    data = _run_check_multi_full(app_client, recs)
    fv = data["fused_verdict"]

    assert data["confidence"]["grade"] == "A"
    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False
    assert data["fused_category"] == "green"


def test_check_multi_wiekszosc_scen_slabej_jakosci_dalej_wymaga_operatora(app_client):
    """Kontrola do powyższego: gdy WIĘKSZOŚĆ scen w oknie jest słabej jakości
    (i nie ma potwierdzenia radarowego), zatwierdzenie operatora nadal jest
    wymagane — tak samo jak w core.assess. Naprawa dotyczy tylko pojedynczych
    słabszych scen przy jednoznacznym wyniku, nie rozluźnia kontroli jakości
    jako takiej."""
    recs = _quality_scenes(2026, [0.45, 0.42, 0.40, 0.88, 0.92])
    data = _run_check_multi_full(app_client, recs)
    fv = data["fused_verdict"]

    assert fv["requires_operator_approval"] is True
    assert data["fused_category"] == "yellow"
    assert any("3/5 scen obniżonej jakości" in d for d in fv["details"])


def _run_check_multi_with_s1(app_client, records, s1_records, rok=2026, typ="scierniskowy"):
    """Jak _run_check_multi_full, ale z include_s1=True i zamockowanym
    fetch_s1_series — do testów potwierdzenia radarowego."""
    with patch("analytics.fetch_multi_series", return_value=records), \
         patch("sentinel1.fetch_s1_series", return_value=s1_records), \
         patch("analytics.detect_agro_events_fused", return_value=[]), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", return_value=[]), \
         patch("terrain.fetch_terrain_context", return_value=None):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": typ, "rok": rok,
            "include_s1": True,
        })
    assert r.status_code == 200, r.text
    return r.json()


def test_check_multi_radar_znosi_blokade_slabej_optyki(app_client):
    """Sens fuzji S1: radar nie jest blokowany przez chmury. Gdy większość
    scen optycznych jest słabej jakości (zachmurzenie), ale wynik jest
    jednoznaczny (OBECNY) i niezależny od chmur RVI sam potwierdza wyraźną
    okrywę, wysyłanie sprawy do ręcznego zatwierdzenia 'z powodu chmur'
    ignorowałoby dokładnie ten pomiar, który chmur nie widzi."""
    recs = _quality_scenes(2026, [0.45, 0.42, 0.40, 0.88, 0.92])
    # RVI 0.69 -> znormalizowane 1.0 (granice 0.05-0.40) — radar widzi gęstą okrywę
    s1 = [{"date": r["date"], "rvi": 0.69, "vv_db": -14.0, "vh_db": -20.0}
          for r in recs]
    data = _run_check_multi_with_s1(app_client, recs, s1)
    fv = data["fused_verdict"]

    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False
    assert data["fused_category"] == "green"
    assert any("radar Sentinel-1" in d for d in fv["details"])


def test_check_multi_radar_slaby_nie_znosi_blokady_slabej_optyki(app_client):
    """Kontrola: radar dostępny, ale wskazujący słabą/żadną strukturę roślinną
    (niskie RVI) NIE znosi blokady — zatwierdzenie operatora zostaje."""
    recs = _quality_scenes(2026, [0.45, 0.42, 0.40, 0.88, 0.92])
    # RVI 0.10 -> znormalizowane ~0.14 — radar NIE potwierdza wyraźnej okrywy
    s1 = [{"date": r["date"], "rvi": 0.10, "vv_db": -8.0, "vh_db": -16.0}
          for r in recs]
    data = _run_check_multi_with_s1(app_client, recs, s1)
    fv = data["fused_verdict"]

    assert fv["requires_operator_approval"] is True
    assert data["fused_category"] == "yellow"


def _run_check_multi_with_events(app_client, records, agro_events, rok=2026, typ="scierniskowy"):
    with patch("analytics.fetch_multi_series", return_value=records), \
         patch("analytics.detect_agro_events_fused", return_value=agro_events), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", return_value=[]), \
         patch("terrain.fetch_terrain_context", return_value=None):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": typ, "rok": rok,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    return r.json()


def test_check_multi_incydent_z_pozniejszym_dowodem_okrywy_nie_blokuje(app_client):
    """To, o co poprosił użytkownik: detekcja zdarzenia patrzy tylko na parę
    kolejnych obserwacji (spadek NDVI), nie sprawdza pozostałych wskaźników
    ani tego, co dzieje się PÓŹNIEJ. Jeśli istnieje wiarygodna, PÓŹNIEJSZA
    obserwacja, której pełny wielowskaźnikowy Confidence Score niezależnie
    potwierdza wyraźną okrywę, incydent był najwyraźniej przejściowy —
    blokowanie pozytywnej decyzji mimo takiego dowodu ignorowałoby właśnie
    te wskaźniki."""
    import core
    from datetime import timedelta
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    event_date = cover_w[0] + timedelta(days=5)

    dip = {"date": event_date, "ndvi": 0.15, "evi": 0.12, "savi": 0.10,
           "ndre": 0.10, "ndwi": 0.05, "valid_frac": 1.0}
    recovery = {"date": event_date + timedelta(days=10), "ndvi": 0.70, "evi": 0.515,
                "savi": 0.486, "ndre": 0.410, "ndwi": 0.441, "valid_frac": 1.0}

    event = {"date": event_date.isoformat(), "type": "confirmed_tillage", "confidence": "high",
             "within_declared": False, "sar_confirmed": True,
             "message": "Optyczny: NDVI 0.60->0.15 + SAR: VV +6 dB."}

    data = _run_check_multi_with_events(app_client, [dip, recovery], [event])
    fv = data["fused_verdict"]

    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False
    assert data["fused_category"] == "green"
    assert any("uznany za przejściowy" in d for d in fv["details"])


def test_check_multi_odrost_po_zdarzeniu_uwzglednia_near_termination(app_client):
    """Regresja z audytu #4: _post_event_recovery() liczyła Confidence Score
    dla kandydata na dowód odrostu BEZ near_termination, w odróżnieniu od
    wyboru peak_record (_evaluate()), który tę wagę przekazuje. Blisko końca
    okna oceny NDWI dostaje podbitą wagę (NDWI_TERMINATION_BOOST) — bez tego
    podbicia scena z realnie silnym NDWI mogła nie przekroczyć progu
    POST_EVENT_RECOVERY_MIN_SCORE=70, mimo że w pełnej, spójnej z resztą
    systemu ocenie by go przekroczyła. Wartości dobrane tak, by surowy score
    (bez podbicia) wypadał tuż PONIŻEJ 70, a z podbiciem tuż POWYŻEJ."""
    import core
    from datetime import timedelta
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    event_date = cover_w[0] + timedelta(days=3)
    # W obrębie 21 dni od końca okna oceny -> near_termination=True.
    recovery_date = cover_w[1] - timedelta(days=5)

    dip = {"date": event_date, "ndvi": 0.15, "evi": 0.12, "savi": 0.10,
           "ndre": 0.10, "ndwi": 0.05, "valid_frac": 1.0}
    # Bez near_termination score ~68.8 (< 70), z near_termination ~71.3 (>= 70).
    recovery = {"date": recovery_date, "ndvi": 0.37, "evi": 0.33,
                "savi": 0.28, "ndre": 0.28, "ndwi": 0.60, "valid_frac": 1.0}

    event = {"date": event_date.isoformat(), "type": "confirmed_tillage", "confidence": "high",
             "within_declared": False, "sar_confirmed": True,
             "message": "Optyczny: NDVI 0.60->0.15 + SAR: VV +6 dB."}

    data = _run_check_multi_with_events(app_client, [dip, recovery], [event])
    fv = data["fused_verdict"]

    # Bez naprawy odrost nie zostałby uznany (score < 70) i incydent nadal
    # blokowałby zatwierdzenie -> ta asercja wymaga poprawionego kodu.
    assert any("uznany za przejściowy" in d for d in fv["details"])
    assert fv["requires_operator_approval"] is False


def test_check_multi_incydent_bez_pozniejszego_dowodu_nadal_blokuje(app_client):
    """Kontrola: bez wiarygodnej, PÓŹNIEJSZEJ obserwacji potwierdzającej
    okrywę, incydent nadal obniża werdykt i wymaga zatwierdzenia — naprawa
    dotyczy tylko przypadku z realnym dowodem odrostu/przejściowości.
    (Tu wcześniejsza dobra obserwacja ustala peak_record/OBECNY, ale zdarzenie
    następuje PO niej i nic już go później nie potwierdza.)"""
    import core
    from datetime import timedelta
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    good_early = {"date": cover_w[0], "ndvi": 0.70, "evi": 0.515, "savi": 0.486,
                  "ndre": 0.410, "ndwi": 0.441, "valid_frac": 1.0}
    event_date = cover_w[0] + timedelta(days=10)
    dip = {"date": event_date, "ndvi": 0.15, "evi": 0.12, "savi": 0.10,
           "ndre": 0.10, "ndwi": 0.05, "valid_frac": 1.0}

    event = {"date": event_date.isoformat(), "type": "confirmed_tillage", "confidence": "high",
             "within_declared": False, "sar_confirmed": True,
             "message": "Optyczny: NDVI 0.60->0.15 + SAR: VV +6 dB."}

    data = _run_check_multi_with_events(app_client, [good_early, dip], [event])
    fv = data["fused_verdict"]

    assert fv["status"] == "PRAWDOPODOBNY"
    assert fv["requires_operator_approval"] is True
    assert data["fused_category"] == "yellow"


def test_check_multi_pozniejszy_niepotwierdzony_incydent_nie_jest_ignorowany(app_client):
    """Regresja z audytu: przy DWÓCH incydentach — wczesnym z odrostem po nim
    i PÓŹNIEJSZYM bez odrostu (czyli faktyczną, trwałą likwidacją) — kod
    wybierał 'worst' po samej pewności (pierwszy o high confidence, czyli w
    praktyce ten wcześniejszy) i sprawdzał odrost tylko względem NIEGO.
    Odrost leżący MIĘDZY dwoma incydentami mylnie 'kasował' oba, całkowicie
    ignorując drugi, nierozwiązany incydent. Zweryfikowane reprodukcją przed
    naprawą: dawało to status OBECNY/green/bez zatwierdzenia mimo trwałej
    likwidacji. Naprawa: liczy się NAJPÓŹNIEJSZY incydent w oknie oceny."""
    import core
    from datetime import timedelta
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    d0 = cover_w[0]

    good1     = {"date": d0, "ndvi": 0.70, "evi": 0.515, "savi": 0.486,
                 "ndre": 0.410, "ndwi": 0.441, "valid_frac": 1.0}
    dip1_date = d0 + timedelta(days=5)
    dip1      = {"date": dip1_date, "ndvi": 0.15, "evi": 0.12, "savi": 0.10,
                 "ndre": 0.10, "ndwi": 0.05, "valid_frac": 1.0}
    recov     = {"date": d0 + timedelta(days=10), "ndvi": 0.70, "evi": 0.515,
                 "savi": 0.486, "ndre": 0.410, "ndwi": 0.441, "valid_frac": 1.0}
    dip2_date = d0 + timedelta(days=20)
    dip2      = {"date": dip2_date, "ndvi": 0.12, "evi": 0.08, "savi": 0.08,
                 "ndre": 0.08, "ndwi": 0.02, "valid_frac": 1.0}

    event1 = {"date": dip1_date.isoformat(), "type": "confirmed_tillage", "confidence": "high",
              "within_declared": False, "sar_confirmed": True,
              "message": "Incydent 1 (wczesny, z odrostem)."}
    event2 = {"date": dip2_date.isoformat(), "type": "confirmed_tillage", "confidence": "high",
              "within_declared": False, "sar_confirmed": True,
              "message": "Incydent 2 (pozny, TRWALY - brak odrostu)."}

    data = _run_check_multi_with_events(
        app_client, [good1, dip1, recov, dip2], [event1, event2])
    fv = data["fused_verdict"]

    assert fv["status"] != "OBECNY"
    assert fv["requires_operator_approval"] is True
    assert data["fused_category"] != "green"
    assert any("Incydent 2" in d for d in fv["details"])


def test_check_multi_bez_zdarzen_daje_obecny(app_client):
    fv = _run_check_multi(app_client, [])
    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False


def test_check_multi_niezadeklarowane_zdarzenie_high_obniza_werdykt(app_client):
    event = {"date": "2026-09-20", "type": "confirmed_tillage", "confidence": "high",
              "within_declared": False, "sar_confirmed": True,
              "message": "Optyczny: NDVI 0.55->0.10 + SAR: VV +8 dB."}
    fv = _run_check_multi(app_client, [event])
    assert fv["status"] == "PRAWDOPODOBNY"
    assert fv["requires_operator_approval"] is True
    assert any("niezadeklarowane zdarzenie" in d for d in fv["details"])


def test_check_multi_zdarzenie_zgodne_z_deklaracja_nie_obniza_werdyktu(app_client):
    event = {"date": "2026-09-20", "type": "declared_termination", "confidence": "high",
              "within_declared": True, "sar_confirmed": True, "message": "Zgodne z deklaracją."}
    fv = _run_check_multi(app_client, [event])
    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False


def test_check_multi_zdarzenie_radar_only_tez_obniza_werdykt(app_client):
    """Teraz, gdy detect_agro_events_fused faktycznie potrafi wygenerować
    radar_only_event (Krok 4), sprawdzamy że filtr w check_multi (który już
    wcześniej dopuszczał ten typ) rzeczywiście go obsługuje end-to-end."""
    event = {"date": "2026-09-20", "type": "radar_only_event", "confidence": "medium",
              "within_declared": False, "sar_confirmed": True,
              "message": "Tylko SAR: VV +4.0 dB, brak scen optycznych."}
    fv = _run_check_multi(app_client, [event])
    assert fv["status"] == "PRAWDOPODOBNY"
    assert fv["requires_operator_approval"] is True


def test_check_multi_zdarzenie_niskiej_pewnosci_nie_wplywa(app_client):
    event = {"date": "2026-09-20", "type": "possible_tillage", "confidence": "low",
              "within_declared": False, "sar_confirmed": False, "message": "Niepewne."}
    fv = _run_check_multi(app_client, [event])
    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False


def test_check_multi_ozimy_orka_po_15_lutego_nie_obniza_werdyktu(app_client):
    """Regresja krytycznego bugu: dla ozimego okno DANYCH/WYŚWIETLANIA sięga
    do 10.03 (core.CFG['ozimy']['t_end']), ale wymóg programu LEN to
    utrzymanie okrywy tylko do 15.02 (core.OZIMY_MAINTENANCE_END). Orka po
    15.02 jest w pełni zgodna z przepisami — a mimo to check_multi
    traktował KAŻDE zdarzenie w oknie 15.02-10.03 jako "niezadeklarowaną
    likwidację" i zawsze zrzucał werdykt do PRAWDOPODOBNY + wymuszał
    przegląd operatora. Efekt: praktycznie żadna działka ozima nie mogła
    wpaść w kategorię 'OK', nawet gdy w pełni spełniła wymóg."""
    event = {"date": "2027-03-01", "type": "confirmed_tillage", "confidence": "high",
              "within_declared": False, "sar_confirmed": True,
              "message": "Optyczny: NDVI 0.60->0.15 + SAR: VV +6 dB."}
    fv = _run_check_multi(app_client, [event], rok=2026, typ="ozimy")
    assert fv["status"] == "OBECNY"
    assert fv["requires_operator_approval"] is False


def test_check_multi_ozimy_orka_przed_15_lutego_dalej_obniza_werdykt(app_client):
    """Kontrola do powyższego: zdarzenie PRZED terminem wymogu (15.02) nadal
    ma obniżać werdykt — naprawa dotyczy tylko okna po terminie."""
    event = {"date": "2027-02-01", "type": "confirmed_tillage", "confidence": "high",
              "within_declared": False, "sar_confirmed": True,
              "message": "Optyczny: NDVI 0.60->0.15 + SAR: VV +6 dB."}
    fv = _run_check_multi(app_client, [event], rok=2026, typ="ozimy")
    assert fv["status"] == "PRAWDOPODOBNY"
    assert fv["requires_operator_approval"] is True


# --- estimate_biomass: zakres niepewności zamiast fałszywej precyzji ---

def test_estimate_biomass_zwraca_none_bez_evi_lub_ndre():
    assert analytics.estimate_biomass(None, 0.2, rvi=0.3) is None
    assert analytics.estimate_biomass(0.4, None, rvi=0.3) is None


def test_estimate_biomass_zwraca_przedzial_a_nie_pojedyncza_liczbe():
    res = analytics.estimate_biomass(0.4, 0.2, rvi=0.3)
    assert res is not None
    assert res["biomass_low_t_ha"] <= res["biomass_t_ha"] <= res["biomass_high_t_ha"]
    assert res["carbon_co2eq_low_t_ha"] <= res["carbon_co2eq_t_ha"] <= res["carbon_co2eq_high_t_ha"]


def test_estimate_biomass_pulap_ograniczony_do_realistycznego_zakresu_poplonu():
    """Regresja: dawniej model przycinał do 15 t/ha (pułap dla kukurydzy).
    Dla poplonu realistyczny pułap to BIOMASS_MAX_T_HA=8.0."""
    res = analytics.estimate_biomass(0.9, 0.9, rvi=0.9)
    assert res["biomass_t_ha"] <= analytics.BIOMASS_MAX_T_HA
    assert res["biomass_high_t_ha"] <= analytics.BIOMASS_MAX_T_HA
    assert res["at_model_cap"] is True


def test_estimate_biomass_nie_jest_na_pulapie_dla_typowych_wartosci():
    res = analytics.estimate_biomass(0.15, 0.1, rvi=0.15)
    assert res["at_model_cap"] is False


def test_estimate_biomass_szerszy_przedzial_niepewnosci_bez_rvi():
    """Brak pomiaru RVI (np. brak scen S1) musi poszerzać przedział
    niepewności, nie tylko obniżać etykietę confidence — inaczej brakujący
    składnik modelu byłby cicho ignorowany bez żadnej kompensacji."""
    with_rvi = analytics.estimate_biomass(0.3, 0.15, rvi=0.25)
    without_rvi = analytics.estimate_biomass(0.3, 0.15, rvi=None)
    width_with = with_rvi["biomass_high_t_ha"] - with_rvi["biomass_low_t_ha"]
    width_without = without_rvi["biomass_high_t_ha"] - without_rvi["biomass_low_t_ha"]
    assert width_without > width_with
    assert with_rvi["confidence"] == "medium"
    assert without_rvi["confidence"] == "low"


def test_estimate_biomass_model_string_odzwierciedla_uzyte_skladniki():
    with_rvi = analytics.estimate_biomass(0.3, 0.15, rvi=0.25)
    without_rvi = analytics.estimate_biomass(0.3, 0.15, rvi=None)
    assert "RVI" in with_rvi["model"]
    assert "bez RVI" in without_rvi["model"] or "RVI" not in without_rvi["model"].split("(")[0]


def test_estimate_biomass_nie_schodzi_ponizej_zera():
    res = analytics.estimate_biomass(0.0, 0.0, rvi=0.0)
    assert res["biomass_t_ha"] >= 0.0
    assert res["biomass_low_t_ha"] >= 0.0


def test_estimate_biomass_co2_to_stala_konwersja_z_biomasy():
    """Współczynnik 0.45*44/12=3.67 to stała biologiczna, nie wymaga
    kalibracji polowej — powinien być stosowany identycznie do każdego
    punktu przedziału (low/mid/high)."""
    res = analytics.estimate_biomass(0.4, 0.2, rvi=0.3)
    factor = 0.45 * 44 / 12
    assert res["carbon_co2eq_t_ha"] == pytest.approx(res["biomass_t_ha"] * factor, abs=0.02)
    assert res["carbon_co2eq_low_t_ha"] == pytest.approx(res["biomass_low_t_ha"] * factor, abs=0.02)


# --- REIP (Red Edge Inflection Point, B06+B07) — krok 1 planu integracji pasm ---

def _reip_formula(red, re1, re2, re3):
    """Ta sama formuła (Guyot & Baret 1988) co w EVALSCRIPT_MULTI/RELAXED —
    powielona w Pythonie, bo evalscript to JS wysyłany do Sentinel Hub i nie
    da się go tu bezpośrednio wykonać."""
    return 700.0 + 40.0 * (((red + re3) / 2.0) - re1) / (re2 - re1 + 1e-9)


@pytest.mark.parametrize("script_name", ["EVALSCRIPT_MULTI", "EVALSCRIPT_MULTI_RELAXED"])
def test_evalscript_pobiera_b06_b07_i_liczy_reip(script_name):
    """Regresja: oba evalscripty muszą pobierać B06/B07 i zwracać REIP —
    inaczej fetch_multi_series dostanie same NaN dla tego wskaźnika."""
    script = getattr(analytics, script_name)
    assert '"B06"' in script
    assert '"B07"' in script
    assert '"reip"' in script
    assert "700.0 + 40.0" in script


def test_reip_dla_zdrowej_okrywy_miesci_sie_w_typowym_zakresie_lit():
    """Dla zdrowej, gęstej roślinności REIP powinien wypaść w typowym dla
    literatury zakresie ~715-725nm (Guyot & Baret). Kontrola liczbowa tej
    samej formuły co w evalscripcie, żeby wychwycić pomyłkę w kolejności
    pasm (RE1/RE2/RE3) zanim trafi do produkcji."""
    # Typowe reflektancje zdrowej okrywy: czerwień niska, red-edge rosnący
    red, re1, re2, re3 = 0.05, 0.12, 0.32, 0.36
    reip = _reip_formula(red, re1, re2, re3)
    assert 710 < reip < 730


def test_reip_dla_gleby_golej_jest_nizszy_niz_dla_okrywy():
    """Gołą glebę (płaskie, niskie odbicie na całym red-edge) REIP powinien
    odróżniać od gęstej okrywy — niższą wartością, zgodnie z fizycznym
    sensem wskaźnika (brak wyraźnego 'skoku' na red-edge)."""
    red_soil, re1_soil, re2_soil, re3_soil = 0.15, 0.17, 0.19, 0.20
    red_veg, re1_veg, re2_veg, re3_veg = 0.05, 0.12, 0.32, 0.36
    reip_soil = _reip_formula(red_soil, re1_soil, re2_soil, re3_soil)
    reip_veg = _reip_formula(red_veg, re1_veg, re2_veg, re3_veg)
    assert reip_soil < reip_veg


def test_blend_red_edge_norm_ufa_reip_mniej_w_fazie_early():
    """Audyt: REIP jest z natury bardziej niestabilny przy rzadkiej okrywie
    (mianownik formuły to różnica dwóch blisko leżących pasm, mała i
    zaszumiona gdy brak wykształconego 'ramienia' red-edge). Blend w fazie
    'early' musi więc ufać REIP mniej niż w fazie 'peak' — przy tych samych
    wartościach wejściowych wynik w 'early' powinien leżeć bliżej samego
    NDRE, a w 'peak' bliżej równej średniej."""
    ndre, reip = 0.30, 720.0  # różne "kierunki" po normalizacji

    val_early, _ = analytics.blend_red_edge_norm(ndre, reip, season_phase="early")
    val_peak, _ = analytics.blend_red_edge_norm(ndre, reip, season_phase="peak")

    ndre_norm = analytics._normalize_indicator(
        ndre, analytics.THRESHOLDS["ndre"]["bare"], analytics.THRESHOLDS["ndre"]["high"])
    reip_norm = analytics._normalize_indicator(reip, *analytics.REIP_NORM_BOUNDS)

    # peak: dokładnie średnia 50/50
    assert val_peak == pytest.approx((ndre_norm + reip_norm) / 2.0)
    # early: bliżej NDRE niż w peak (mniejszy udział REIP)
    assert abs(val_early - ndre_norm) < abs(val_peak - ndre_norm)


def test_fetch_multi_series_zwraca_reip_z_odpowiedzi_statistical_api():
    """Regresja parsowania: fetch_multi_series musi wyciągać 'reip' z bloku
    outputs tak samo jak pozostałe wskaźniki (_mean('reip')), inaczej pole
    zniknie mimo że evalscript je zwraca."""
    from shapely.geometry import Polygon

    geom = Polygon([(19.0, 50.0), (19.001, 50.0), (19.001, 50.001), (19.0, 50.001)])

    # Wartości w kodowaniu INT16 z EVALSCRIPT_MULTI (wskaźniki x10000,
    # REIP jako (reip-680) x 400) — patrz SKALOWANIE w analytics.py.
    fake_response_json = {
        "data": [{
            "interval": {"from": "2026-06-01T00:00:00Z", "to": "2026-06-02T00:00:00Z"},
            "outputs": {
                "ndvi": {"bands": {"B0": {"stats": {"mean": 5500, "sampleCount": 100, "noDataCount": 5}}}},
                "evi":  {"bands": {"B0": {"stats": {"mean": 4000}}}},
                "savi": {"bands": {"B0": {"stats": {"mean": 4500}}}},
                "ndre": {"bands": {"B0": {"stats": {"mean": 3000}}}},
                "ndwi": {"bands": {"B0": {"stats": {"mean": 1000}}}},
                "reip": {"bands": {"B0": {"stats": {"mean": 15360}}}},  # (718.4-680)*400
            },
        }],
    }

    class FakeResp:
        status_code = 200
        def json(self): return fake_response_json

    with patch("analytics._post_with_retry", return_value=FakeResp()), \
         patch("analytics._get_token", return_value="fake-token"):
        records = analytics.fetch_multi_series(geom, date(2026, 6, 1), date(2026, 6, 2))

    assert len(records) == 1
    assert records[0]["reip"] == pytest.approx(718.4)
    assert records[0]["ndvi"] == pytest.approx(0.55)


def test_reip_i_ndre_laczone_w_jeden_komponent_red_edge():
    """Krok 3: REIP nie jest 7. osobno ważonym składnikiem — jest uśredniany
    z NDRE (gdy oba dostępne) w jeden komponent pod dotychczasowym kluczem
    wag 'ndre'. Regresja: gdyby ktoś dodał reip jako osobny ważony składnik,
    n_available_indicators wzrosłoby do 7 mimo braku nowego niezależnego
    dowodu — dokładnie ten problem, przeciwko któremu jest sparse_penalty."""
    # Sam NDRE (bez REIP) — komponent = znormalizowany NDRE
    r_ndre_only = analytics.compute_confidence_score(
        ndvi=0.5, evi=0.4, savi=0.35, ndre=0.30, valid_frac=1.0)
    assert r_ndre_only["red_edge_sources"] == 1
    assert r_ndre_only["n_available_indicators"] == 4  # ndvi, evi, savi, ndre/red_edge

    # Sam REIP (bez zmierzonego NDRE) — komponent = znormalizowany REIP,
    # ale wciąż liczy się jako JEDEN dostępny wskaźnik, nie dwa.
    r_reip_only = analytics.compute_confidence_score(
        ndvi=0.5, evi=0.4, savi=0.35, reip=718.0, valid_frac=1.0)
    assert r_reip_only["red_edge_sources"] == 1
    assert r_reip_only["n_available_indicators"] == 4

    # Oba dostępne — scalone w jeden komponent (średnia), nadal JEDEN
    # wskaźnik dla n_available_indicators, nie dwa.
    r_both = analytics.compute_confidence_score(
        ndvi=0.5, evi=0.4, savi=0.35, ndre=0.30, reip=718.0, valid_frac=1.0)
    assert r_both["red_edge_sources"] == 2
    assert r_both["n_available_indicators"] == 4
    assert r_both["raw_components"]["ndre"] == pytest.approx(0.30)
    assert r_both["raw_components"]["reip"] == pytest.approx(718.0)

    # Komponent "ndre" w wyniku z obydwoma dostępnymi musi leżeć między
    # komponentem liczonym z samego NDRE a komponentem z samego REIP
    # (bo to ich średnia) — a nie być identyczny z żadnym z nich osobno,
    # inaczej scalanie nic by nie robiło.
    ndre_norm_only = r_ndre_only["components"]["ndre"]
    reip_norm_only = r_reip_only["components"]["ndre"]
    both_norm = r_both["components"]["ndre"]
    lo, hi = sorted([ndre_norm_only, reip_norm_only])
    assert lo <= both_norm <= hi


def test_reip_bez_ndre_nie_jest_karany_jako_brak_danych():
    """Fallback: gdy zmierzony jest TYLKO REIP (np. sceptyczny scenariusz —
    inny etap ekstrakcji pasm zawiódł dla samego NDRE), red edge nadal ma
    wartość zamiast None, żeby nie tracić realnego dowodu bez powodu."""
    r = analytics.compute_confidence_score(ndvi=0.5, evi=0.4, savi=0.35, reip=720.0)
    assert r["components"]["ndre"] is not None
    assert r["red_edge_sources"] == 1


# --- NDTI (Normalized Difference Tillage Index, B12) — krok 2 planu integracji pasm ---

def _ndti_formula(swir1, swir2):
    """Ta sama formuła (Van Deventer et al. 1997) co w EVALSCRIPT_MULTI/RELAXED."""
    return (swir1 - swir2) / (swir1 + swir2 + 1e-9)


@pytest.mark.parametrize("script_name", ["EVALSCRIPT_MULTI", "EVALSCRIPT_MULTI_RELAXED"])
def test_evalscript_pobiera_b12_i_liczy_ndti(script_name):
    """Regresja: oba evalscripty muszą pobierać B12 i zwracać NDTI —
    inaczej fetch_multi_series dostanie same NaN dla tego wskaźnika."""
    script = getattr(analytics, script_name)
    assert '"B12"' in script
    assert '"ndti"' in script
    assert "SWIR1 - SWIR2" in script


def test_ndti_dla_goleij_gleby_nizszy_niz_dla_resztek_lub_biomasy():
    """NDTI powinien odróżniać glebę goła/świeżo zaoraną (niski) od resztek
    pożniwnych/gęstej biomasy (wyższy) — to fizyczny sens wskaźnika, na
    którym oprze się przyszła korekta detekcji orki (krok 3)."""
    ndti_bare_soil = _ndti_formula(swir1=0.25, swir2=0.24)   # gleba: SWIR1≈SWIR2
    ndti_residue    = _ndti_formula(swir1=0.30, swir2=0.18)   # resztki/biomasa: SWIR1>>SWIR2
    assert ndti_bare_soil < ndti_residue


def test_fetch_multi_series_zwraca_ndti_z_odpowiedzi_statistical_api():
    """Regresja parsowania: fetch_multi_series musi wyciągać 'ndti' z bloku
    outputs tak samo jak pozostałe wskaźniki (_mean('ndti'))."""
    from shapely.geometry import Polygon

    geom = Polygon([(19.0, 50.0), (19.001, 50.0), (19.001, 50.001), (19.0, 50.001)])

    # Wartości w kodowaniu INT16 z EVALSCRIPT_MULTI (wskaźniki x10000,
    # REIP jako (reip-680) x 400) — patrz SKALOWANIE w analytics.py.
    fake_response_json = {
        "data": [{
            "interval": {"from": "2026-06-01T00:00:00Z", "to": "2026-06-02T00:00:00Z"},
            "outputs": {
                "ndvi": {"bands": {"B0": {"stats": {"mean": 5500, "sampleCount": 100, "noDataCount": 5}}}},
                "evi":  {"bands": {"B0": {"stats": {"mean": 4000}}}},
                "savi": {"bands": {"B0": {"stats": {"mean": 4500}}}},
                "ndre": {"bands": {"B0": {"stats": {"mean": 3000}}}},
                "ndwi": {"bands": {"B0": {"stats": {"mean": 1000}}}},
                "reip": {"bands": {"B0": {"stats": {"mean": 15360}}}},  # (718.4-680)*400
                "ndti": {"bands": {"B0": {"stats": {"mean": 2200}}}},
            },
        }],
    }

    class FakeResp:
        status_code = 200
        def json(self): return fake_response_json

    with patch("analytics._post_with_retry", return_value=FakeResp()), \
         patch("analytics._get_token", return_value="fake-token"):
        records = analytics.fetch_multi_series(geom, date(2026, 6, 1), date(2026, 6, 2))

    assert len(records) == 1
    assert records[0]["ndti"] == pytest.approx(0.22)


def test_ndti_nie_jest_parametrem_confidence_score():
    """Zgodnie z ustaloną architekturą (krok 3) NDTI nie wchodzi do
    Confidence Score wcale — trafia wyłącznie do detect_agro_events_fused
    jako dowód na orkę/odsłonięcie gleby, analogicznie do roli RVI."""
    import inspect
    sig = inspect.signature(analytics.compute_confidence_score)
    assert "ndti" not in sig.parameters


def test_agro_events_ndti_podnosi_pewnosc_bez_potwierdzenia_radarowego():
    """Krok 3: gdy SAR nie potwierdza (brak scysji S1 w oknie), równoczesny
    spadek NDTI (odsłonięcie gleby, niezależny sygnał optyczny od NDVI) musi
    podnieść pewność z 'medium' do 'high' — to dokładnie ta sytuacja, w
    której wcześniej zdarzenie zostawało z etykietą 'możliwy stres lub
    zagubiona scena S1', mimo dostępnego dodatkowego dowodu w danych, które
    już i tak pobieramy (fetch_multi_series)."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55, "ndti": 0.28},
        {"date": date(2026, 9, 8), "ndvi": 0.10, "ndti": 0.05},  # spadek NDTI >= 0.06
    ]
    events = analytics.detect_agro_events_fused(optical, [], sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["type"] == "possible_tillage"
    assert events[0]["confidence"] == "high"
    assert events[0]["ndti_confirmed"] is True
    assert events[0]["ndti_drop"] == pytest.approx(0.23)


def test_agro_events_bez_spadku_ndti_zostaje_medium():
    """Kontrola: gdy NDTI NIE spada razem z NDVI (np. sam stres wodny/
    szkodnik, gleba pozostaje zakryta), pewność zostaje 'medium' — NDTI nie
    podnosi pewności na siłę, tylko gdy faktycznie koroboruje odsłonięcie
    gleby."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55, "ndti": 0.20},
        {"date": date(2026, 9, 8), "ndvi": 0.10, "ndti": 0.19},  # praktycznie bez zmian
    ]
    events = analytics.detect_agro_events_fused(optical, [], sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["confidence"] == "medium"
    assert events[0]["ndti_confirmed"] is False


def test_agro_events_ndti_nie_podnosi_pewnosci_gdy_s1_dostepny_ale_nie_potwierdza():
    """Regresja z audytu: gdy S1 BYŁ dostępny w oknie, ale skok VV był za
    mały (radar aktywnie sprawdził i nie potwierdził), NDTI NIE powinno
    podnosić pewności do 'high' — to inna sytuacja niż 'S1 w ogóle
    niedostępny'. Wcześniej ndti_confirmation sprawdzało tylko
    'sar_confirmation is None', co myliło te dwa różne stany."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55, "ndti": 0.28},
        {"date": date(2026, 9, 8), "ndvi": 0.10, "ndti": 0.05},  # spadek NDTI >= 0.06
    ]
    s1 = [
        # S1 JEST dostępny w oknie obu dat, ale skok VV poniżej progu (1.0 dB < 2.5 dB)
        {"date": date(2026, 9, 2), "vv_db": -12.0},
        {"date": date(2026, 9, 9), "vv_db": -11.0},
    ]
    events = analytics.detect_agro_events_fused(optical, s1, sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["confidence"] == "medium"
    assert events[0]["type"] == "possible_tillage"
    assert events[0]["sar_confirmed"] is False
    assert events[0]["s1_contradicts"] is True
    # NDTI faktycznie spadło wystarczająco, ale to nie mogło przebić radaru
    assert events[0]["ndti_drop"] == pytest.approx(0.23)


def test_agro_events_ndti_nie_nadpisuje_potwierdzenia_radarowego():
    """SAR wciąż ma pierwszeństwo — gdy radar potwierdza, zdarzenie jest
    'confirmed_tillage'/high niezależnie od tego, co pokazuje NDTI (nawet
    jeśli brak danych NDTI albo NDTI nie spada)."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55, "ndti": 0.20},
        {"date": date(2026, 9, 8), "ndvi": 0.10, "ndti": None},  # brak NDTI dla tego dnia
    ]
    s1 = [
        {"date": date(2026, 9, 2), "vv_db": -12.0},
        {"date": date(2026, 9, 10), "vv_db": -4.0},
    ]
    events = analytics.detect_agro_events_fused(optical, s1, sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["type"] == "confirmed_tillage"
    assert events[0]["confidence"] == "high"
    assert events[0]["sar_confirmed"] is True


def test_agro_events_brak_ndti_w_rekordach_nie_wywala_sie():
    """Kompatybilność wstecz: starsze/mockowane rekordy optyczne bez klucza
    'ndti' w ogóle (nie tylko None) nie mogą wywracać detekcji zdarzeń."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55},
        {"date": date(2026, 9, 8), "ndvi": 0.10},
    ]
    events = analytics.detect_agro_events_fused(optical, [], sowing_date=None, termination_date=None)
    assert len(events) == 1
    assert events[0]["ndti_confirmed"] is False
    assert events[0]["ndti_drop"] is None


# --- Pogoda (weather.py): kontekst przymrozku/posuchy w detekcji zdarzeń ---

def test_agro_events_przymrozek_obniza_pewnosc_niepotwierdzonego_zdarzenia():
    """Nowa funkcjonalność: spadek NDVI bez żadnego niezależnego potwierdzenia
    (brak S1, brak NDTI) zbiegający się z przymrozkiem ma wiarygodne
    alternatywne wytłumaczenie -> pewność obniżona z 'medium' do 'low', żeby
    NIE blokować automatycznie werdyktu w app.py (filtr tam bierze tylko
    high/medium), ale zdarzenie nadal jest widoczne z wyjaśnieniem."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55},
        {"date": date(2026, 9, 8), "ndvi": 0.10},
    ]
    frost_dates = {date(2026, 9, 8)}

    events_bez_pogody = analytics.detect_agro_events_fused(
        optical, [], sowing_date=None, termination_date=None)
    assert events_bez_pogody[0]["confidence"] == "medium"
    assert events_bez_pogody[0]["weather_explained"] is False

    events_z_mrozem = analytics.detect_agro_events_fused(
        optical, [], sowing_date=None, termination_date=None,
        weather_frost_dates=frost_dates)
    assert events_z_mrozem[0]["confidence"] == "low"
    assert events_z_mrozem[0]["weather_explained"] is True
    assert "przymrozek" in events_z_mrozem[0]["message"]


def test_agro_events_posucha_obniza_pewnosc_niepotwierdzonego_zdarzenia():
    optical = [
        {"date": date(2026, 7, 1), "ndvi": 0.55},
        {"date": date(2026, 7, 8), "ndvi": 0.10},
    ]
    dry_dates = {date(2026, 7, 8)}

    events = analytics.detect_agro_events_fused(
        optical, [], sowing_date=None, termination_date=None,
        weather_dry_dates=dry_dates)
    assert events[0]["confidence"] == "low"
    assert events[0]["weather_explained"] is True
    assert "posucha" in events[0]["message"]


def test_agro_events_pogoda_nie_obniza_potwierdzonej_orki_sar():
    """Krytyczne rozróżnienie: przymrozek NIE unieważnia potwierdzonej orki —
    SAR to niezależny, bezpośredni pomiar zmiany struktury powierzchni,
    którego przypadkowa pogoda nie tłumaczy. confidence zostaje 'high'."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55},
        {"date": date(2026, 9, 8), "ndvi": 0.10},
    ]
    s1 = [
        {"date": date(2026, 9, 2), "vv_db": -12.0},
        {"date": date(2026, 9, 10), "vv_db": -4.0},
    ]
    frost_dates = {date(2026, 9, 8)}

    events = analytics.detect_agro_events_fused(
        optical, s1, sowing_date=None, termination_date=None,
        weather_frost_dates=frost_dates)
    assert events[0]["type"] == "confirmed_tillage"
    assert events[0]["confidence"] == "high"
    assert events[0]["weather_explained"] is False


def test_agro_events_pogoda_nie_obniza_potwierdzonego_ndti():
    """Analogicznie do SAR: NDTI to niezależny dowód odsłonięcia gleby —
    przymrozek go nie unieważnia."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55, "ndti": 0.20},
        {"date": date(2026, 9, 8), "ndvi": 0.10, "ndti": 0.10},
    ]
    frost_dates = {date(2026, 9, 8)}

    events = analytics.detect_agro_events_fused(
        optical, [], sowing_date=None, termination_date=None,
        weather_frost_dates=frost_dates)
    assert events[0]["ndti_confirmed"] is True
    assert events[0]["confidence"] == "high"
    assert events[0]["weather_explained"] is False


def test_agro_events_bez_zbiegu_z_pogoda_pewnosc_bez_zmian():
    """Kontrola: gdy podane są zbiory dat przymrozku/posuchy, ale żadna z
    nich nie pokrywa się z datami zdarzenia, pewność zostaje niezmieniona."""
    optical = [
        {"date": date(2026, 9, 1), "ndvi": 0.55},
        {"date": date(2026, 9, 8), "ndvi": 0.10},
    ]
    events = analytics.detect_agro_events_fused(
        optical, [], sowing_date=None, termination_date=None,
        weather_frost_dates={date(2026, 1, 1)}, weather_dry_dates={date(2026, 2, 1)})
    assert events[0]["confidence"] == "medium"
    assert events[0]["weather_explained"] is False


# --- Audyt #6: test end-to-end /api/check_multi z realnym REIP/NDTI ---
# (dotąd integracja REIP/NDTI była przetestowana jednostkowo na poziomie
# compute_confidence_score/detect_agro_events_fused, ale nigdy przez cały
# check_multi — z realnymi rekordami zawierającymi te pola.)

def test_check_multi_reip_realnie_wplywa_na_wynik_przez_caly_pipeline(app_client):
    """REIP wysoki (bliski/powyżej górnej granicy REIP_NORM_BOUNDS) powinien
    podnieść scalony komponent red-edge, a przez to i całościowy score,
    względem identycznego rekordu bez REIP — sprawdzone na żywym
    /api/check_multi, nie tylko wywołaniem compute_confidence_score wprost."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    base = {"date": cover_w[0], "ndvi": 0.55, "evi": 0.45, "savi": 0.40,
            "ndre": 0.30, "ndwi": 0.30, "valid_frac": 1.0}
    rec_bez_reip = dict(base)
    rec_z_reip = {**base, "reip": 725.0}  # blisko/powyżej górnej granicy REIP_NORM_BOUNDS

    data_bez = _run_check_multi_full(app_client, [rec_bez_reip])
    data_z = _run_check_multi_full(app_client, [rec_z_reip])

    assert data_z["confidence"]["red_edge_sources"] == 2
    assert data_bez["confidence"]["red_edge_sources"] == 1
    # REIP=725 normalizuje się powyżej samego NDRE (0.30) -> blend musi
    # podnieść score względem wersji bez REIP.
    assert data_z["confidence"]["score"] > data_bez["confidence"]["score"]
    # I surowy REIP ma być widoczny w odpowiedzi do audytu/UI.
    assert data_z["confidence"]["raw_components"]["reip"] == pytest.approx(725.0)


def test_check_multi_ndti_widoczny_w_peak_record_ale_nie_zmienia_score(app_client):
    """NDTI ma trafiać do peak_record (do detekcji zdarzeń / UI), ale
    celowo NIE wchodzi do Confidence Score — score musi być identyczny
    z/bez NDTI przy tych samych pozostałych wskaźnikach."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    base = {"date": cover_w[0], "ndvi": 0.55, "evi": 0.45, "savi": 0.40,
            "ndre": 0.30, "ndwi": 0.30, "valid_frac": 1.0}
    rec_bez_ndti = dict(base)
    rec_z_ndti = {**base, "ndti": 0.05}

    data_bez = _run_check_multi_full(app_client, [rec_bez_ndti])
    data_z = _run_check_multi_full(app_client, [rec_z_ndti])

    assert data_z["peak_record"]["ndti"] == pytest.approx(0.05)
    assert data_z["confidence"]["score"] == pytest.approx(data_bez["confidence"]["score"])


# --- Pogoda/teren: kontekst z weather.py/terrain.py w /api/check_multi ---

def test_check_multi_zawiera_weather_context_i_terrain_context(app_client):
    """Nowa funkcjonalność: check_multi pobiera (nieblokująco) kontekst
    pogodowy i rzeźby terenu i wystawia je w odpowiedzi. Weryfikuje samo
    okablowanie (parsowanie odpowiedzi API jest testowane osobno w
    test_weather.py/test_terrain.py)."""
    import core
    from datetime import date as date_cls
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    rec = {"date": cover_w[0], "ndvi": 0.55, "evi": 0.45, "savi": 0.40,
           "ndre": 0.30, "ndwi": 0.30, "valid_frac": 1.0}

    fake_weather_series = [
        {"date": date_cls(2026, 9, 1), "tmin_c": -3.0, "precip_mm": 0.0},
        {"date": date_cls(2026, 9, 8), "tmin_c": 5.0, "precip_mm": 0.0},
    ]
    fake_terrain = {"elevation_m": 150.0, "slope_deg": 8.0,
                    "aspect_deg": 180.0, "erosion_risk": "wysokie"}

    with patch("analytics.fetch_multi_series", return_value=[rec]), \
         patch("analytics.detect_agro_events_fused", return_value=[]), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", return_value=fake_weather_series), \
         patch("terrain.fetch_terrain_context", return_value=fake_terrain):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["weather_context"]["frost_days"] == 1
    assert data["weather_error"] is None
    assert data["terrain_context"] == fake_terrain
    # growing_conditions: informacyjne podsumowanie GDD/opadów dla okna
    # oceny (cover_window) — fake_weather_series ma tylko tmin_c/precip_mm
    # (bez tmean/tmax), więc GDD nie da się policzyć (None). Oczekiwaną
    # wartość opadów liczymy tym samym mechanizmem co produkcja, żeby test
    # nie zakładał na sztywno, które dni fake_weather_series wpadają w
    # cover_window.
    import weather as weather_mod
    expected_gc = weather_mod.growing_conditions_summary(
        fake_weather_series, cover_w[0], cover_w[1])
    gc = data["growing_conditions"]
    assert gc["cumulative_gdd"] is None
    assert gc["total_precip_mm"] == expected_gc["total_precip_mm"]
    # Wysokie ryzyko erozyjne -> informacyjna notatka w fused_details, ale
    # BEZ wpływu na status/requires_operator_approval (inny typ dowodu).
    assert any("erozyjne" in d for d in data["fused_verdict"]["details"])
    assert data["fused_verdict"]["status"] == "OBECNY"
    assert data["fused_verdict"]["requires_operator_approval"] is False


def test_check_multi_dziala_gdy_pogoda_i_teren_niedostepne(app_client):
    """Nieblokujące zachowanie: gdy weather.fetch_weather_series rzuca
    wyjątek, check_multi MUSI mimo to zwrócić poprawny wynik (z
    weather_error ustawionym, ale bez przerywania całej oceny) —
    analogicznie do s1_error dla Sentinel-1."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    rec = {"date": cover_w[0], "ndvi": 0.55, "evi": 0.45, "savi": 0.40,
           "ndre": 0.30, "ndwi": 0.30, "valid_frac": 1.0}

    with patch("analytics.fetch_multi_series", return_value=[rec]), \
         patch("analytics.detect_agro_events_fused", return_value=[]), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", side_effect=RuntimeError("Open-Meteo niedostępne")), \
         patch("terrain.fetch_terrain_context", side_effect=RuntimeError("boom")):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["weather_error"] is not None
    assert data["terrain_context"] is None
    assert data["fused_verdict"]["status"] == "OBECNY"


# --- score_low/score_high: zakres niepewności wynikający z jakości/kompletności danych ---

def test_compute_confidence_score_zakres_wezszy_przy_pelnych_danych_dobrej_jakosci():
    """Przy komplecie z >=3 niezależnych rodzin dowodów i valid_frac=1.0
    (sparse_penalty=1.0, quality_penalty=1.0) margines to dokładnie
    SCORE_UNCERTAINTY_BASE_PP w każdą stronę."""
    import analytics as A
    cs = A.compute_confidence_score(
        ndvi=0.55, evi=0.45, savi=0.40, ndre=0.30, ndwi=0.30, rvi=None,
        valid_frac=1.0, season_phase="peak")
    assert cs["score"] is not None
    margin = round(A.SCORE_UNCERTAINTY_BASE_PP, 1)
    assert cs["score_low"] == pytest.approx(max(0.0, cs["score"] - margin), abs=0.15)
    assert cs["score_high"] == pytest.approx(min(100.0, cs["score"] + margin), abs=0.15)


def test_compute_confidence_score_zakres_szerszy_przy_niepelnych_danych():
    """Mniej niezależnych rodzin dowodów (tu: tylko NDVI, jedna rodzina
    nir_red) -> sparse_penalty < 1.0 -> SZERSZY zakres niż przy komplecie."""
    import analytics as A
    cs_pelne = A.compute_confidence_score(
        ndvi=0.55, evi=0.45, savi=0.40, ndre=0.30, ndwi=0.30, rvi=None,
        valid_frac=1.0, season_phase="peak")
    cs_sparse = A.compute_confidence_score(
        ndvi=0.55, evi=None, savi=None, ndre=None, ndwi=None, rvi=None,
        valid_frac=1.0, season_phase="peak")
    margin_pelne = cs_pelne["score_high"] - cs_pelne["score_low"]
    margin_sparse = cs_sparse["score_high"] - cs_sparse["score_low"]
    assert margin_sparse > margin_pelne


def test_compute_confidence_score_zakres_none_gdy_brak_danych():
    import analytics as A
    cs = A.compute_confidence_score(season_phase="peak")
    assert cs["score"] is None
    assert cs["score_low"] is None
    assert cs["score_high"] is None


def test_compute_confidence_score_zakres_ograniczony_do_0_100():
    import analytics as A
    # Skrajnie niska jakość + tylko jeden wskaźnik -> szeroki margines,
    # ale wynikowy przedział MUSI zostać przycięty do [0, 100].
    cs = A.compute_confidence_score(
        ndvi=0.05, evi=None, savi=None, ndre=None, ndwi=None, rvi=None,
        valid_frac=0.35, season_phase="peak")
    assert cs["score_low"] >= 0.0
    assert cs["score_high"] <= 100.0


def test_check_multi_udostepnia_score_low_high_w_odpowiedzi(app_client):
    """Okablowanie: score_low/score_high z compute_confidence_score muszą
    przejść przez cały check_multi aż do JSON-a widocznego przez operatora."""
    import core
    cover_w = core.CFG["scierniskowy"]["cover"](2026)
    rec = {"date": cover_w[0], "ndvi": 0.55, "evi": 0.45, "savi": 0.40,
           "ndre": 0.30, "ndwi": 0.30, "valid_frac": 1.0}

    with patch("analytics.fetch_multi_series", return_value=[rec]), \
         patch("analytics.detect_agro_events_fused", return_value=[]), \
         patch("analytics.save_multi_observation", return_value=None), \
         patch("analytics.select_primary_index", return_value={}), \
         patch("analytics.get_season_phase", return_value="peak"), \
         patch("weather.fetch_weather_series", return_value=[]), \
         patch("terrain.fetch_terrain_context", return_value=None):
        r = app_client.post("/api/check_multi", json={
            "parcel_id": "141201_1.0001.6509", "typ": "scierniskowy", "rok": 2026,
            "include_s1": False,
        })
    assert r.status_code == 200, r.text
    cs = r.json()["confidence"]
    assert cs["score_low"] is not None
    assert cs["score_high"] is not None
    assert cs["score_low"] <= cs["score"] <= cs["score_high"]


# --- EVI2 i NDMI z B8A (listopad 2026) ------------------------------------

@pytest.mark.parametrize("script_name", ["EVALSCRIPT_MULTI", "EVALSCRIPT_MULTI_RELAXED"])
def test_evalscript_evi2_bez_niebieskiego_i_ndmi_z_b8a(script_name):
    script = getattr(analytics, script_name)
    assert "s.B02" not in script and '"B02"' not in script         # brak niebieskiego w kodzie JS
    assert '"B8A"' in script
    assert "2.4*RED" in script                                       # EVI2
    assert "(NIR_N - SWIR1) / (NIR_N + SWIR1" in script              # NDMI z B8A


def test_evalscript_multi_liczba_pasm_wejsciowych_bez_zmian():
    """Koszt PU zależy od liczby pasm wejściowych: B02 wypadł, B8A doszedł."""
    import re
    m = re.search(r'bands: \[([^\]]+)\]', analytics.EVALSCRIPT_MULTI)
    bands = [b.strip().strip('"') for b in m.group(1).split(",")]
    assert len([b for b in bands if b != "dataMask"]) == 9


def test_evi2_bliskie_evi_dla_gestej_okrywy():
    NIR, RED, BLUE = 0.35, 0.06, 0.03
    evi = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)
    evi2 = 2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1.0)
    assert abs(evi2 - evi) / evi < 0.08
    assert evi2 >= analytics.THRESHOLDS["evi"]["low"]



def test_ndwi_distribution_z_percentyli_i_histogramu():
    outputs = {"ndwi": {"bands": {"B0": {
        "stats": {"mean": 2800, "percentiles": {"10.0": 1200, "50.0": 3000}},
        "histogram": {"bins": [
            {"lowEdge": 1000, "highEdge": 1250, "count": 20},   # < 0.30-0.10 -> suchy
            {"lowEdge": 1750, "highEdge": 2000, "count": 5},    # = granica 0.20 -> suchy
            {"lowEdge": 2750, "highEdge": 3000, "count": 50},
            {"lowEdge": 3000, "highEdge": 3250, "count": 25},
        ]}}}}}
    d = analytics._ndwi_distribution(outputs)
    assert d == {"ndwi_p10": 0.12, "ndwi_p50": 0.30, "ndwi_low_frac": 0.25}


def test_ndwi_distribution_none_gdy_brak_danych():
    assert analytics._ndwi_distribution({}) == {"ndwi_p10": None, "ndwi_p50": None, "ndwi_low_frac": None}
