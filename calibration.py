"""Adaptacja progów NDVI na podstawie zweryfikowanych zdjęć od rolników."""

import logging
import threading

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import db
from auth import require_operator_key

log = logging.getLogger("calibration")
router = APIRouter()

# Serializuje odczyt-modyfikuj-zapisz progów, żeby dwa równoczesne
# /api/calibration/apply nie mogły się przeplatać.
_apply_lock = threading.Lock()


def compute_suggested_thresholds(min_samples=10):
    """Oblicza sugerowane progi na podstawie zebranych zdjęć z analizą AI."""
    photos = db.get_photos_for_calibration()

    vegetation = [p for p in photos
                  if p.get("ai_verdict") == "vegetation_present"
                  and p.get("ai_coverage_pct") is not None
                  and p.get("ai_coverage_pct") >= 30
                  and p.get("ndvi_at_alarm") is not None]

    bare = [p for p in photos
            if (p.get("ai_verdict") == "bare_soil"
                or (p.get("ai_coverage_pct") is not None and p.get("ai_coverage_pct") < 15))
            and p.get("ndvi_at_alarm") is not None]

    total = len(vegetation) + len(bare)
    if total < min_samples:
        return None

    import core
    current_high = core.NDVI_COVER_HIGH
    current_mid = core.NDVI_COVER_MID

    # percentyl 25% z NDVI dla zdjęć z poplonem
    veg_ndvi = sorted([p["ndvi_at_alarm"] for p in vegetation])
    bare_ndvi = sorted([p["ndvi_at_alarm"] for p in bare])

    def percentile(lst, pct):
        if not lst:
            return None
        idx = int(len(lst) * pct / 100)
        idx = max(0, min(idx, len(lst) - 1))
        return lst[idx]

    new_high = percentile(veg_ndvi, 25) if veg_ndvi else current_high
    new_mid_raw = percentile(bare_ndvi, 75) if bare_ndvi else current_mid
    new_mid = (new_mid_raw + 0.02) if new_mid_raw is not None else current_mid

    # ogranicz do sensownych zakresów
    new_high = max(0.30, min(0.65, round(new_high, 2))) if new_high else current_high
    new_mid = max(0.20, min(0.50, round(new_mid, 2))) if new_mid else current_mid

    # upewnij się że jest odpowiednia separacja
    if new_high < new_mid + 0.05:
        new_high = round(new_mid + 0.08, 2)

    change_magnitude = max(abs(new_high - current_high), abs(new_mid - current_mid))

    return {
        "suggested_cover_high": new_high,
        "suggested_cover_mid": new_mid,
        "current_cover_high": current_high,
        "current_cover_mid": current_mid,
        "vegetation_samples": len(vegetation),
        "bare_samples": len(bare),
        "total_samples": total,
        "change_magnitude": round(change_magnitude, 3),
    }


_INDICATOR_FIELD_MAP = {
    "ndvi": "ndvi_at_alarm", "evi": "evi_at_alarm", "savi": "savi_at_alarm",
    "ndre": "ndre_at_alarm", "ndwi": "ndwi_at_alarm", "rvi": "rvi_at_alarm",
}

# Mnożnik wagi dla wskaźnika z wystarczającą liczbą próbek, ale zerową/ujemną
# separacją wegetacja-vs-gleba (potwierdzony słaby/odwrotny sygnał, nie brak
# danych) — waga jest obniżana, nie zachowywana bez zmian.
NEGATIVE_SEPARATION_PENALTY = 0.5


def _normalize(val, lo, hi):
    if val is None:
        return None
    return min(1.0, max(0.0, (val - lo) / (hi - lo + 1e-9)))


def compute_suggested_weights(min_samples=10, min_per_indicator=3, phase="peak"):
    """Sugeruje korektę wag Confidence Score dla danej fazy fenologicznej,
    analogicznie do compute_suggested_thresholds() — ale zamiast progów NDVI,
    koryguje relatywne wagi NDVI/EVI/SAVI/NDRE/NDWI/RVI na podstawie tego, jak
    dobrze każdy wskaźnik (znormalizowany do [0,1]) rozróżnia zdjęcia
    zweryfikowane jako 'żywa okrywa' od 'goła gleba'.

    Wskaźniki z za małą liczbą próbek (< min_per_indicator na stronę) zachowują
    obecną wagę — nie są korygowane bez wystarczających danych.

    Wskaźniki z WYSTARCZAJĄCĄ liczbą próbek, ale zerową/ujemną separacją
    (średnia normalizowana wartość niższa dla wegetacji niż dla gołej gleby —
    sygnał odwrotny do oczekiwanego) są traktowane inaczej niż brak danych:
    dostają karę NEGATIVE_SEPARATION_PENALTY (waga obniżona, nie zachowana) —
    to potwierdzony, słaby/odwrotny sygnał przy realnej próbie, a nie zwykły
    brak danych, więc nie powinien zatrzymywać pełnej dotychczasowej wagi.

    Klucz "ndre" ocenia separację ZBLENDOWANEGO sygnału NDRE+REIP (patrz
    analytics.blend_red_edge_norm), nie samego surowego NDRE — bo dokładnie
    to jest ważone w compute_confidence_score pod tym kluczem od czasu
    integracji REIP. Gdyby kalibracja liczyła separację czystego NDRE,
    sugestie wag dryfowałyby od tego, co faktycznie ocenia scoring."""
    photos = db.get_photos_for_calibration()

    vegetation = [p for p in photos
                  if p.get("ai_verdict") == "vegetation_present"
                  and p.get("ai_coverage_pct") is not None
                  and p.get("ai_coverage_pct") >= 30]

    bare = [p for p in photos
            if (p.get("ai_verdict") == "bare_soil"
                or (p.get("ai_coverage_pct") is not None and p.get("ai_coverage_pct") < 15))]

    total = len(vegetation) + len(bare)
    if total < min_samples:
        return None

    import analytics
    norm_bounds = {
        # ndvi: NIE z zamrożonej THRESHOLDS["ndvi"]["high"] — patrz
        # analytics.ndvi_norm_bounds() i UWAGA przy analytics.THRESHOLDS.
        # Inaczej kalibracja oceniałaby separację NDVI względem innego progu
        # niż ten, który faktycznie normalizuje compute_confidence_score po
        # zatwierdzeniu kalibracji NDVI operatora.
        "ndvi": analytics.ndvi_norm_bounds(),
        "evi":  (analytics.THRESHOLDS["evi"]["bare"],  analytics.THRESHOLDS["evi"]["high"]),
        "savi": (analytics.THRESHOLDS["savi"]["bare"], analytics.THRESHOLDS["savi"]["high"]),
        "ndre": (analytics.THRESHOLDS["ndre"]["bare"], analytics.THRESHOLDS["ndre"]["high"]),
        "ndwi": (analytics.THRESHOLDS["ndwi"]["bare"], analytics.THRESHOLDS["ndwi"]["high"]),
        "rvi":  analytics.RVI_NORM_BOUNDS,
    }

    def _collect_norm_values(records, ind):
        """Zwraca listę znormalizowanych [0,1] wartości wskaźnika `ind` z
        listy zdjęć. Dla 'ndre' NIE normalizuje surowego ndre_at_alarm samo
        w sobie — używa analytics.blend_red_edge_norm(ndre, reip), czyli
        DOKŁADNIE tego samego scalania NDRE+REIP, które faktycznie jest
        ważone w compute_confidence_score pod kluczem 'ndre' (patrz audyt:
        bez tego kalibracja oceniałaby separację innego sygnału niż ten,
        którego wagę koryguje — wagi dryfowałyby od rzeczywistości w miarę
        jak przybywa scen z dostępnym REIP)."""
        if ind == "ndre":
            out = []
            for p in records:
                val, _sources = analytics.blend_red_edge_norm(
                    p.get("ndre_at_alarm"), p.get("reip_at_alarm"), phase)
                if val is not None:
                    out.append(val)
            return out
        lo, hi = norm_bounds[ind]
        field = _INDICATOR_FIELD_MAP[ind]
        return [_normalize(p[field], lo, hi) for p in records if p.get(field) is not None]

    separations = {}
    sample_counts = {}
    for ind in norm_bounds:
        veg_norm = _collect_norm_values(vegetation, ind)
        bare_norm = _collect_norm_values(bare, ind)
        sample_counts[ind] = len(veg_norm) + len(bare_norm)
        if len(veg_norm) < min_per_indicator or len(bare_norm) < min_per_indicator:
            separations[ind] = None
            continue
        separations[ind] = (sum(veg_norm) / len(veg_norm)) - (sum(bare_norm) / len(bare_norm))

    current = dict(analytics.CONFIDENCE_WEIGHTS_BY_PHASE.get(
        phase, analytics.CONFIDENCE_WEIGHTS_BY_PHASE["peak"]))

    # wskaźniki dobrze próbkowane i dodatnio dyskryminujące -> kandydaci do
    # podniesienia wagi
    positive = {k: v for k, v in separations.items() if v is not None and v > 0}
    if len(positive) < 2:
        return None

    # wskaźniki dobrze próbkowane, ale z separacją <=0 (potwierdzony słaby
    # lub odwrotny sygnał) -> kara, NIE zachowanie obecnej wagi
    penalized = {k: current[k] * NEGATIVE_SEPARATION_PENALTY
                for k, v in separations.items() if v is not None and v <= 0}

    # wskaźniki bez wystarczających danych -> obecna waga zachowana bez zmian
    insufficient_data = {k for k, v in separations.items() if v is None}

    reserved = sum(current[k] for k in insufficient_data) + sum(penalized.values())
    total_sep = sum(positive.values())

    suggested = dict(current)
    for k, w in penalized.items():
        suggested[k] = w
    for k, sep in positive.items():
        suggested[k] = (sep / total_sep) * (1.0 - reserved)

    s = sum(suggested.values())
    suggested = {k: round(v / s, 4) if s > 1e-9 else current[k] for k, v in suggested.items()}

    change_magnitude = max(abs(suggested[k] - current[k]) for k in suggested)

    return {
        "phase": phase,
        "current_weights": {k: round(v, 4) for k, v in current.items()},
        "suggested_weights": suggested,
        "indicator_separation": {k: (round(v, 4) if v is not None else None)
                                 for k, v in separations.items()},
        "sample_counts": sample_counts,
        "vegetation_samples": len(vegetation),
        "bare_samples": len(bare),
        "total_samples": total,
        "change_magnitude": round(change_magnitude, 4),
    }


def apply_weights(phase: str, weights: dict):
    """Zapisuje nowe wagi Confidence Score dla danej fazy w tabeli `settings`
    (jako nadpisanie w JSON obok pozostałych faz) i przeładowuje je w pamięci
    procesu przez analytics.reload_weights() — operator zatwierdza ręcznie,
    tak jak przy progach NDVI.

    Scala nowe wagi z ewentualnym istniejącym nadpisaniem tej samej fazy
    (zamiast go całkowicie zastępować) — naprawa: dawniej `overrides[phase] =
    weights` kasowało wszystkie wcześniej zatwierdzone wagi dla innych
    wskaźników tej fazy, jeśli nowe wywołanie nie obejmowało kompletu 6
    wskaźników (w praktyce ograniczone walidacją sumy≈1.0 w API, ale to
    krucha ochrona — merge jest poprawny niezależnie od tego, co przyśle
    wywołujący)."""
    import json
    with _apply_lock:
        try:
            import analytics
            old_weights = dict(analytics.CONFIDENCE_WEIGHTS_BY_PHASE.get(phase, {}))

            raw = db.get_setting("confidence_weights_override")
            overrides = json.loads(raw) if raw else {}
            overrides.setdefault(phase, {})
            overrides[phase].update({k: float(v) for k, v in weights.items()})
            db.set_setting("confidence_weights_override", json.dumps(overrides))
            analytics.reload_weights()

            log.info(f"Wagi Confidence Score ({phase}) zaktualizowane: "
                     f"{old_weights} -> {weights}")
            return True
        except Exception as e:
            log.error(f"Błąd aktualizacji wag: {e}")
            return False


def apply_thresholds(new_cover_high: float, new_cover_mid: float):
    """Zapisuje nowe progi w tabeli `settings` i uaktualnia je w pamięci
    procesu przez core.reload_thresholds() — nie modyfikujemy pliku
    źródłowego core.py w runtime (to był błąd: dawniej ta funkcja
    nadpisywała core.py na dysku bez blokady, co przy dwóch równoczesnych
    wywołaniach mogło uszkodzić plik i wymagało pełnego importlib.reload)."""
    with _apply_lock:
        try:
            import core
            old_high = core.NDVI_COVER_HIGH
            old_mid = core.NDVI_COVER_MID

            db.set_setting("ndvi_cover_high", f"{new_cover_high:.2f}")
            db.set_setting("ndvi_cover_mid", f"{new_cover_mid:.2f}")
            core.reload_thresholds()

            photos = db.get_photos_for_calibration()
            db.log_threshold_change(old_high, new_cover_high, old_mid, new_cover_mid,
                                    len(photos))
            log.info(f"Progi zaktualizowane: HIGH {old_high}->{new_cover_high}, "
                     f"MID {old_mid}->{new_cover_mid}")
            return True
        except Exception as e:
            log.error(f"Błąd aktualizacji progów: {e}")
            return False


def get_calibration_status():
    import core
    import json
    photos = db.get_photos_for_calibration()
    all_photos = db.get_photos_for_calibration(min_samples=0)
    analyzed = [p for p in photos if p.get("ai_verdict") and p["ai_verdict"] != "uncertain"]
    last = db.get_last_calibration()

    # Czy wagi Confidence Score zostały kiedykolwiek zatwierdzone na
    # podstawie realnych danych zdjęciowych (a nie tylko domyślnych wartości
    # dobranych "z literatury")? Progi ocen A-F NIE zostały ponownie
    # zwalidowane pod nowe, zależne-od-fazy wagi (patrz komentarz w
    # analytics.compute_confidence_score) — dopóki żadna faza nie ma
    # zatwierdzonego nadpisania, oceny B/C blisko granic progów warto
    # traktować z ostrożnością, zwłaszcza dla fazy "winter".
    raw_override = db.get_setting("confidence_weights_override")
    calibrated_phases = []
    if raw_override:
        try:
            calibrated_phases = sorted(json.loads(raw_override).keys())
        except (ValueError, TypeError):
            pass

    return {
        "current_cover_high": core.NDVI_COVER_HIGH,
        "current_cover_mid": core.NDVI_COVER_MID,
        "current_bare_max": core.NDVI_BARE_MAX,
        "total_photos": len(all_photos),
        "analyzed_photos": len(analyzed),
        "usable_for_calibration": len(photos),
        "min_samples_required": 10,
        "last_calibration": dict(last) if last else None,
        "weights_calibrated_phases": calibrated_phases,
        "weights_never_calibrated_warning": (
            len(calibrated_phases) == 0
        ),
    }


# --- API endpoints ---

@router.get("/api/calibration/status")
def calibration_status():
    return get_calibration_status()


@router.get("/api/calibration/suggest")
def calibration_suggest():
    result = compute_suggested_thresholds()
    if result is None:
        photos = db.get_photos_for_calibration()
        return JSONResponse({"status": "not_enough_data",
                             "min_samples": 10,
                             "current_samples": len(photos)})
    return result


class CalibrationApplyRequest(BaseModel):
    new_cover_high: float = Field(..., ge=0.25, le=0.70)
    new_cover_mid: float = Field(..., ge=0.20, le=0.60)

    @model_validator(mode="after")
    def _check_separation(self):
        if self.new_cover_high <= self.new_cover_mid:
            raise ValueError("new_cover_high musi być większe niż new_cover_mid")
        return self


@router.post("/api/calibration/apply")
def calibration_apply(body: CalibrationApplyRequest, _auth=Depends(require_operator_key)):
    ok = apply_thresholds(body.new_cover_high, body.new_cover_mid)
    if ok:
        return {"status": "ok",
                "applied": {"cover_high": body.new_cover_high, "cover_mid": body.new_cover_mid}}
    return JSONResponse({"status": "error", "detail": "Nie udało się zapisać progów"}, status_code=500)


@router.get("/api/calibration/suggest_weights")
def calibration_suggest_weights(phase: str = "peak"):
    if phase not in ("early", "peak", "winter"):
        return JSONResponse({"status": "error", "detail": "Nieznana faza"}, status_code=422)
    result = compute_suggested_weights(phase=phase)
    if result is None:
        photos = db.get_photos_for_calibration()
        return JSONResponse({"status": "not_enough_data",
                             "min_samples": 10,
                             "current_samples": len(photos)})
    return result


class WeightsApplyRequest(BaseModel):
    phase: str = Field(..., pattern="^(early|peak|winter)$")
    weights: dict[str, float]

    @model_validator(mode="after")
    def _check_weights(self):
        allowed = {"ndvi", "evi", "savi", "ndre", "rvi", "ndwi"}
        if not set(self.weights.keys()) <= allowed:
            raise ValueError(f"Nieznany wskaźnik w wagach — dozwolone: {sorted(allowed)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Wagi nie mogą być ujemne")
        total = sum(self.weights.values())
        if not (0.9 <= total <= 1.1):
            raise ValueError(f"Suma wag ({total:.3f}) musi być bliska 1.0")
        return self


@router.post("/api/calibration/apply_weights")
def calibration_apply_weights(body: WeightsApplyRequest, _auth=Depends(require_operator_key)):
    ok = apply_weights(body.phase, body.weights)
    if ok:
        return {"status": "ok", "applied": {"phase": body.phase, "weights": body.weights}}
    return JSONResponse({"status": "error", "detail": "Nie udało się zapisać wag"}, status_code=500)
