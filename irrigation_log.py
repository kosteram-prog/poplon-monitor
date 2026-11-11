"""Moduł teledetekcji i wody — log nawodnienia (Krok 1 planu).

Rejestr pól monitorowanych pod kątem stresu wodnego i zdarzeń nawadniania,
niezależny od monitoringu poplonów (własne tabele water_monitored_fields /
irrigation_events, patrz db.py — osobna tabela i lista pól, bo cykl życia
pola wodnego nie jest przywiązany do sezonu poplonu).

irrigation_mode to atrybut KONKRETNEGO pola ('irrigation' | 'rainfed'), nie
globalna flaga aplikacji — jedna instancja obsługuje jednocześnie pola
nawadniane (Uzbekistan) i uprawy deszczowe (Europa/Lower Silesia 360).

Na tym etapie moduł tylko zbiera dane (pola + log nawodnienia). Silnik
anomalii stresu wodnego, kolejka triage doradcy i panel rolnika to kolejne
kroki tego samego planu (patrz notatki z rozmowy o module wodnym).
"""
import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

import db
import water_stress
from auth import require_operator_key

log = logging.getLogger("irrigation_log")
router = APIRouter(prefix="/api/water", tags=["water"])


class WaterFieldAddRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    irrigation_mode: str = Field(..., pattern="^(irrigation|rainfed)$")
    geometry_wkt: str = Field(
        ..., description="Geometria pola w WKT, WGS84 (EPSG:4326). Wymagana "
        "wprost (nie doczytywana z ULDK jak w monitored_parcels) — pola poza "
        "Polską (np. Uzbekistan) nie mają numeru działki w GUGiK.")
    region: Optional[str] = Field(
        None, description="Klucz kohorty do porównań międzypolowych w silniku "
        "anomalii (patrz water_stress.get_cohort_fields) — pola z tym samym "
        "region i irrigation_mode są traktowane jako 'podobne'.")
    crop_type: Optional[str] = Field(
        None, max_length=40, description="Uprawa na polu (np. bawełna, pszenica, "
        "kukurydza) — część klucza kohorty: pola porównujemy tylko z polami tej samej uprawy.")
    soil_taw_mm: Optional[float] = Field(
        None, gt=0, le=400, description="Retencja użyteczna strefy korzeniowej [mm] "
        "do bilansu wodnego (np. piasek ~50, glina pylasta ~150). Puste = 100 mm.")
    sowing_date: Optional[str] = Field(None, description="RRRR-MM-DD — start bilansu wodnego")
    farmer_phone: Optional[str] = None
    farmer_name: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("sowing_date")
    @classmethod
    def _validate_sowing_date(cls, v):
        if v in (None, ""):
            return None
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("sowing_date musi być w formacie RRRR-MM-DD")
        return v

    @field_validator("geometry_wkt")
    @classmethod
    def _validate_geometry_wkt(cls, v):
        from shapely import wkt as shapely_wkt
        try:
            geom = shapely_wkt.loads(v)
        except Exception as e:
            raise ValueError(f"Nieprawidlowa geometria WKT: {e}")
        if geom.is_empty:
            raise ValueError("geometry_wkt nie moze byc puste")
        return v


class WaterFieldUpdateRequest(BaseModel):
    """Zmiana atrybutów istniejącego pola (np. uprawa w nowym sezonie)."""
    region: Optional[str] = Field(None, max_length=80)
    crop_type: Optional[str] = Field(None, max_length=40)
    soil_taw_mm: Optional[float] = Field(None, gt=0, le=400)
    sowing_date: Optional[str] = None
    farmer_phone: Optional[str] = None
    farmer_name: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("sowing_date")
    @classmethod
    def _validate_sowing_date(cls, v):
        if v in (None, ""):
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("sowing_date musi być w formacie RRRR-MM-DD")
        return v


class IrrigationEventRequest(BaseModel):
    event_date: str = Field(..., description="RRRR-MM-DD")
    method: str = Field(..., pattern="^(zalewowe|kroplowe|deszczowanie)$")
    amount_or_duration: Optional[float] = Field(None, gt=0)
    amount_unit: Optional[str] = Field(None, max_length=20)
    amount_mm: Optional[float] = Field(
        None, gt=0, le=500, description="Dawka w mm (gdy pusta — przeliczana z "
        "amount_or_duration dla jednostek mm, l/m², m³/ha).")
    notes: Optional[str] = None

    @field_validator("event_date")
    @classmethod
    def _validate_event_date(cls, v):
        try:
            d = date.fromisoformat(v)
        except ValueError:
            raise ValueError("event_date musi być w formacie RRRR-MM-DD")
        if d > date.today():
            raise ValueError("event_date nie może być z przyszłości")
        return v


class IrrigationPlanAddRequest(BaseModel):
    planned_date: str = Field(..., description="RRRR-MM-DD")
    method: Optional[str] = Field(None, pattern="^(zalewowe|kroplowe|deszczowanie)$")
    notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("planned_date")
    @classmethod
    def _validate_planned_date(cls, v):
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("planned_date musi być w formacie RRRR-MM-DD")
        return v


class IrrigationPlanCompleteRequest(BaseModel):
    """Oznaczenie planu jako 'wykonano'. Opcjonalnie od razu loguje realne
    zdarzenie w irrigation_events (sekcja 5.1) -- z parametrami z planu jako
    domyślnymi, ale nadpisywalnymi (np. planowano kroplowe, wyszlo zalewowe)."""
    event_date: Optional[str] = Field(None, description="RRRR-MM-DD, domyślnie dzisiaj")
    method: Optional[str] = Field(None, pattern="^(zalewowe|kroplowe|deszczowanie)$")
    amount_or_duration: Optional[float] = Field(None, gt=0)
    amount_unit: Optional[str] = Field(None, max_length=20)
    notes: Optional[str] = None
    log_event: bool = Field(True, description="Czy od razu utworzyć wpis w logu nawodnienia")

    @field_validator("event_date")
    @classmethod
    def _validate_event_date(cls, v):
        if v is None:
            return v
        try:
            d = date.fromisoformat(v)
        except ValueError:
            raise ValueError("event_date musi być w formacie RRRR-MM-DD")
        if d > date.today():
            raise ValueError("event_date nie może być z przyszłości")
        return v


class IrrigationPlanSkipRequest(BaseModel):
    notes: Optional[str] = Field(None, max_length=2000)


def _get_field_or_404(water_field_id: int) -> dict:
    field = db.get_water_field_by_id(water_field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Nie znaleziono pola.")
    return field


def _get_plan_or_404(plan_id: int) -> dict:
    plan = db.get_irrigation_plan_by_id(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Nie znaleziono planu nawodnienia.")
    return plan


@router.post("/field/add")
def water_field_add(req: WaterFieldAddRequest, _auth=Depends(require_operator_key)):
    kwargs = {k: v for k, v in req.model_dump().items()
              if k not in ("parcel_id", "irrigation_mode", "geometry_wkt") and v is not None}
    field_id = db.add_water_field(req.parcel_id, req.irrigation_mode, req.geometry_wkt, **kwargs)
    return {"id": field_id, "parcel_id": req.parcel_id, "status": "added"}


@router.get("/field/list")
def water_field_list(status: str = "active"):
    fields = db.get_water_fields(status=status if status != "all" else None)
    result = []
    for f in fields:
        events = db.get_irrigation_events(f["id"])
        result.append({**f, "recent_irrigation_events": events[-10:]})
    return result


@router.patch("/field/{water_field_id}")
def water_field_update(water_field_id: int, req: WaterFieldUpdateRequest,
                       _auth=Depends(require_operator_key)):
    """Zmiana atrybutów pola (np. nowa uprawa po zmianie sezonu). Pusty
    string czyści wartość; pola pominięte w żądaniu zostają bez zmian."""
    _get_field_or_404(water_field_id)
    changes = {k: (v.strip() or None) if isinstance(v, str) else v
               for k, v in req.model_dump(exclude_unset=True).items()}
    db.update_water_field(water_field_id, **changes)
    return db.get_water_field_by_id(water_field_id)


@router.delete("/field/{water_field_id}")
def water_field_delete(water_field_id: int, _auth=Depends(require_operator_key)):
    _get_field_or_404(water_field_id)
    db.update_water_field_status(water_field_id, "completed")
    return {"status": "deactivated"}


@router.post("/field/{water_field_id}/irrigation")
def irrigation_event_add(water_field_id: int, req: IrrigationEventRequest,
                          _auth=Depends(require_operator_key)):
    _get_field_or_404(water_field_id)
    event_id = db.add_irrigation_event(
        water_field_id, req.event_date, req.method,
        amount_or_duration=req.amount_or_duration,
        amount_unit=req.amount_unit, notes=req.notes, amount_mm=req.amount_mm)
    return {"id": event_id, "water_field_id": water_field_id, "status": "added"}


@router.get("/field/{water_field_id}/irrigation")
def irrigation_event_list(water_field_id: int):
    _get_field_or_404(water_field_id)
    return db.get_irrigation_events(water_field_id)


# ---------------------------------------------------------------------------
# Planowanie nawadniania (sekcja 5.2 planu, "faza 2") -- prosty kalendarz
# planowanych nawodnien per pole z mozliwoscia oznaczenia "wykonano" po
# fakcie. To narzedzie ORGANIZACYJNE, nie system rekomendacji dawek: nie ma
# tu zadnej logiki sugerujacej KIEDY nawadniac, tylko rejestr tego, co
# rolnik/doradca sam zaplanowal. Osobne od irrigation_events (log realnych
# zdarzen, sekcja 5.1) -- plan moze zostac pominiety (status 'skipped') albo
# wykonany inaczej niz planowano.
# ---------------------------------------------------------------------------

@router.post("/field/{water_field_id}/plan")
def irrigation_plan_add(water_field_id: int, req: IrrigationPlanAddRequest,
                         _auth=Depends(require_operator_key)):
    _get_field_or_404(water_field_id)
    plan_id = db.add_irrigation_plan(water_field_id, req.planned_date,
                                      method=req.method, notes=req.notes)
    return {"id": plan_id, "water_field_id": water_field_id, "status": "planned"}


@router.get("/field/{water_field_id}/plan")
def irrigation_plan_list(water_field_id: int, status: str = "all"):
    _get_field_or_404(water_field_id)
    return db.get_irrigation_plans(water_field_id=water_field_id,
                                    status=status if status != "all" else None)


@router.get("/plan/upcoming")
def irrigation_plan_upcoming(status: str = "planned"):
    """Widok kalendarza nad wszystkimi polami naraz -- doradca/rolnik nie
    musi wchodzic w kazde pole osobno, zeby zobaczyc co jest zaplanowane
    najblizej. Dokleja parcel_id pola do kazdego wpisu (surowy plan zna tylko
    water_field_id)."""
    plans = db.get_irrigation_plans(status=status if status != "all" else None)
    field_cache: dict = {}
    result = []
    for p in plans:
        fid = p["water_field_id"]
        if fid not in field_cache:
            field_cache[fid] = db.get_water_field_by_id(fid)
        field = field_cache[fid]
        result.append({**p, "parcel_id": field["parcel_id"] if field else None,
                       "region": field.get("region") if field else None})
    return result


@router.post("/plan/{plan_id}/complete")
def irrigation_plan_complete(plan_id: int, req: IrrigationPlanCompleteRequest,
                              _auth=Depends(require_operator_key)):
    plan = _get_plan_or_404(plan_id)
    event_id = None
    if req.log_event:
        method = req.method or plan["method"]
        if not method:
            raise HTTPException(status_code=400,
                                 detail="Podaj metode nawodnienia (plan jej nie mial).")
        event_date = req.event_date or date.today().isoformat()
        event_id = db.add_irrigation_event(
            plan["water_field_id"], event_date, method,
            amount_or_duration=req.amount_or_duration,
            amount_unit=req.amount_unit, source="plan", notes=req.notes)
    db.set_irrigation_plan_status(plan_id, "done", irrigation_event_id=event_id)
    return {"id": plan_id, "status": "done", "irrigation_event_id": event_id}


@router.post("/plan/{plan_id}/skip")
def irrigation_plan_skip(plan_id: int, req: IrrigationPlanSkipRequest,
                          _auth=Depends(require_operator_key)):
    _get_plan_or_404(plan_id)
    db.set_irrigation_plan_status(plan_id, "skipped", notes=req.notes)
    return {"id": plan_id, "status": "skipped"}


@router.delete("/plan/{plan_id}")
def irrigation_plan_delete(plan_id: int, _auth=Depends(require_operator_key)):
    _get_plan_or_404(plan_id)
    db.delete_irrigation_plan(plan_id)
    return {"id": plan_id, "status": "deleted"}


@router.get("/field/{water_field_id}/observations")
def water_observations_list(water_field_id: int):
    """Historia obserwacji NDWI/S1/pogoda dla pola — surowe dane pod przyszły
    wykres w panelu rolnika (Krok 4 planu)."""
    _get_field_or_404(water_field_id)
    return db.get_water_observations(water_field_id)


class BackfillRequest(BaseModel):
    days: int = Field(90, ge=7, le=365, description="Ile dni wstecz uzupełnić")


@router.post("/field/{water_field_id}/backfill")
def water_field_backfill(water_field_id: int, req: BackfillRequest,
                         _auth=Depends(require_operator_key)):
    """Uzupełnia historię obserwacji pola wstecz (water_replay.backfill_field),
    żeby anomalia względem własnej historii działała od razu, a nie po kilku
    tygodniach. Kosztuje PU (jedno zapytanie S2 + jedno S1 na cały okres)."""
    field = _get_field_or_404(water_field_id)
    import water_replay
    t_end = date.today()
    try:
        return water_replay.backfill_field(field, t_end - timedelta(days=req.days), t_end)
    except Exception as e:
        log.exception("Blad backfill dla pola wodnego id=%s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Blad uzupelniania historii: {e}")


@router.get("/field/{water_field_id}/replay")
def water_field_replay(water_field_id: int, with_balance: bool = False):
    """Suchy przebieg silnika na zapisanej historii pola: jakie alarmy
    powstałyby w kolejnych dniach (nic nie zapisuje do kolejki triage).
    with_balance=true dokłada bilans FAO-56 na pogodzie z archiwum ERA5."""
    field = _get_field_or_404(water_field_id)
    import water_replay
    wx = []
    if with_balance:
        obs = db.get_water_observations(water_field_id)
        if obs:
            import weather as weather_mod
            from shapely import wkt as shapely_wkt
            try:
                wx = weather_mod.fetch_weather_series(
                    shapely_wkt.loads(field["geometry_wkt"]),
                    date.fromisoformat(obs[0]["obs_date"]), date.today())
            except Exception as e:
                log.warning("Pogoda niedostepna dla replay pola id=%s: %s", water_field_id, e)
    return water_replay.replay_field(field, weather_series=wx)


@router.get("/field/{water_field_id}/balance")
def water_field_balance(water_field_id: int):
    """Bilans wodny strefy korzeniowej (FAO-56) z sugestią terminu i dawki
    nawodnienia — narzędzie doradcze, nic nie zapisuje (plan dodaje
    człowiek). Źródła: Open-Meteo (bez klucza, bez PU), NDVI z historii
    obserwacji pola, log nawodnień."""
    field = _get_field_or_404(water_field_id)
    try:
        import water_balance
        return water_balance.compute_field_balance(field)
    except Exception as e:
        log.exception("Blad bilansu wodnego dla pola id=%s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Blad bilansu wodnego: {e}")


@router.post("/field/{water_field_id}/check_now")
def water_field_check_now(water_field_id: int, _auth=Depends(require_operator_key)):
    """Sprawdzenie 'na żądanie' silnikiem anomalii (water_stress.py) —
    kosztowne (odpytuje Sentinel Hub/Open-Meteo), dlatego chronione tak samo
    jak monitor_check_now w module poplonów. Celowo NIE zapisuje do kolejki
    triage (patrz water_stress.check_field_now) — to podgląd dla operatora,
    kolejkę zapełnia tylko scheduler (Krok 3)."""
    field = _get_field_or_404(water_field_id)
    try:
        return water_stress.check_field_now(field)
    except Exception as e:
        log.exception("Blad check_now dla pola wodnego id=%s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Blad silnika anomalii: {e}")


# ---------------------------------------------------------------------------
# Kolejka triage doradcy (Krok 3) — alerty z water_stress.py trafiają tu
# najpierw (sekcja 4 planu), NIE bezpośrednio do rolnika. Rolnik zobaczy
# potwierdzone alerty dopiero w panelu z Kroku 4.
# ---------------------------------------------------------------------------

class TriageRequest(BaseModel):
    triage_status: str = Field(..., pattern="^(confirmed|rejected|needs_visit)$")
    triage_note: Optional[str] = Field(None, max_length=2000)
    triage_by: Optional[str] = Field(None, max_length=80)


@router.get("/triage_queue")
def water_triage_queue(status: str = "pending"):
    return db.get_water_alarms(triage_status=status if status != "all" else None)


@router.get("/alarm/{alarm_id}")
def water_alarm_get(alarm_id: int):
    alarm = db.get_water_alarm_by_id(alarm_id)
    if not alarm:
        raise HTTPException(status_code=404, detail="Nie znaleziono alarmu.")
    return alarm


@router.post("/alarm/{alarm_id}/triage")
def water_alarm_triage(alarm_id: int, req: TriageRequest, _auth=Depends(require_operator_key)):
    alarm = db.get_water_alarm_by_id(alarm_id)
    if not alarm:
        raise HTTPException(status_code=404, detail="Nie znaleziono alarmu.")
    db.set_water_alarm_triage(alarm_id, req.triage_status,
                              triage_note=req.triage_note, triage_by=req.triage_by)
    return {"id": alarm_id, "triage_status": req.triage_status}


# ---------------------------------------------------------------------------
# Petla zwrotna / rekalibracja progu anomalii (patrz water_stress.py:
# compute_suggested_water_threshold) -- wzorem /api/calibration/* w module
# poplonow (calibration.py), ale osobno pod /api/water, bo kalibruje inny
# silnik na innych danych (decyzje triage, nie zdjecia terenowe).
# ---------------------------------------------------------------------------

@router.get("/calibration/status")
def water_calibration_status():
    return water_stress.get_water_calibration_status()


@router.get("/calibration/suggest")
def water_calibration_suggest():
    result = water_stress.compute_suggested_water_threshold()
    if result is None:
        return {"status": "not_enough_data", "min_samples": 10}
    return result


class WaterThresholdApplyRequest(BaseModel):
    new_threshold: float = Field(..., ge=-4.0, le=-0.5)


@router.post("/calibration/apply")
def water_calibration_apply(body: WaterThresholdApplyRequest, _auth=Depends(require_operator_key)):
    ok = water_stress.apply_water_threshold(body.new_threshold)
    if not ok:
        raise HTTPException(status_code=500, detail="Nie udalo sie zapisac progu.")
    return {"status": "ok", "applied": {"z_score_risk_threshold": body.new_threshold}}
