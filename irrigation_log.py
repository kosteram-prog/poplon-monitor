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
    geometry_wkt: Optional[str] = Field(
        None, description="Geometria pola w WKT, WGS84 (EPSG:4326). Może być "
        "pusta, gdy parcel_id to numer(y) działek ewidencyjnych z Polski "
        "(oddzielone przecinkiem) — wtedy obrys pobierany jest z ULDK. Dla pól "
        "poza Polską (np. Uzbekistan) wymagana.")
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
    split_parcels: bool = Field(
        False, description="Kilka numerów działek w parcel_id = kilka OSOBNYCH pól "
        "(zamiast jednego pola złożonego z kilku działek)")
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
        if v is None or not str(v).strip():
            return None
        try:
            geom = shapely_wkt.loads(v)
        except Exception as e:
            raise ValueError(f"Nieprawidlowa geometria WKT: {e}")
        if geom.is_empty:
            raise ValueError("geometry_wkt nie moze byc puste")
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("geometry_wkt musi byc poligonem (POLYGON albo MULTIPOLYGON)")
        if not geom.is_valid:
            # Np. obrys narysowany "ósemką" albo nachodzące na siebie części —
            # naprawiamy zamiast odrzucać; zostają tylko części powierzchniowe.
            from shapely import make_valid
            from shapely.geometry import MultiPolygon, Polygon
            fixed = make_valid(geom)
            parts = [g for g in getattr(fixed, "geoms", [fixed]) if isinstance(g, (Polygon, MultiPolygon))]
            if not parts:
                raise ValueError("Nie da sie naprawic geometrii pola")
            from shapely.ops import unary_union
            return unary_union(parts).wkt
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
    import water_geometry as wg
    from shapely import wkt as shapely_wkt
    kwargs = {k: v for k, v in req.model_dump().items()
              if k not in ("parcel_id", "irrigation_mode", "geometry_wkt", "split_parcels")
              and v is not None}
    active = {f["parcel_id"] for f in db.get_water_fields(status="active")}

    if req.split_parcels and not req.geometry_wkt and len(wg.split_parcel_ids(req.parcel_id)) > 1:
        # Kilka numerów działek -> kilka osobnych pól (obrys każdej z ULDK).
        added, skipped = [], []
        for pid in wg.split_parcel_ids(req.parcel_id):
            if pid in active:
                skipped.append(pid)
                continue
            wkt = _geometry_from_uldk_or_422(pid)
            added.append({"id": db.add_water_field(pid, req.irrigation_mode, wkt, **kwargs),
                          "parcel_id": pid})
            active.add(pid)
        return {"status": "added", "added": added, "skipped_existing": skipped,
                "geometry_source": "uldk"}

    pid = req.parcel_id.strip()
    if pid in active:
        raise HTTPException(status_code=409, detail=f"Pole „{pid}” jest już w module wodnym.")
    geometry_wkt, source = req.geometry_wkt, "manual"
    if not geometry_wkt:
        geometry_wkt = _geometry_from_uldk_or_422(pid)
        source = "uldk"
    try:
        wg.check_field_extent(shapely_wkt.loads(geometry_wkt))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    field_id = db.add_water_field(pid, req.irrigation_mode, geometry_wkt, **kwargs)
    return {"id": field_id, "parcel_id": pid, "status": "added",
            "geometry_source": source}


def _geometry_from_uldk_or_422(parcel_id: str) -> str:
    import core
    import water_geometry as wg
    if not wg.is_polish_parcel_ids(parcel_id):
        raise HTTPException(status_code=422, detail=(
            "Brak geometrii pola. Dla polskich działek podaj numer TERYT "
            "(np. 141201_1.0001.6509, kilka oddziel przecinkiem), dla innych pól "
            "zaznacz obrys na mapie albo wklej WKT."))
    try:
        return wg.geometry_from_parcel_ids(parcel_id)["geometry_wkt"]
    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/parcel_lookup")
def water_parcel_lookup(ids: str):
    """Obrys pola z ULDK dla numeru (numerów) działek — podgląd przed dodaniem."""
    import core
    import water_geometry as wg
    if not wg.is_polish_parcel_ids(ids):
        raise HTTPException(status_code=422, detail="To nie wygląda na numer działki TERYT "
                            "(np. 141201_1.0001.6509).")
    try:
        return wg.geometry_from_parcel_ids(ids)
    except core.AppError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/parcel_at")
def water_parcel_at(lat: float, lon: float):
    """Działka ewidencyjna w klikniętym punkcie mapy (ULDK, tylko Polska)."""
    import core
    import water_geometry as wg
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Nieprawidłowe współrzędne.")
    try:
        geom, meta = core.fetch_parcel_at_point(lon, lat)
    except core.AppError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"parcel_id": meta.get("parcel_id"), "geometry_wkt": geom.wkt,
            "area_ha": wg.area_ha(geom), "gmina": meta.get("gmina")}


@router.get("/field_at")
def water_field_at(lat: float, lon: float, source: str = "auto"):
    """Gotowy obrys pola w klikniętym punkcie mapy.
    source=auto: w Polsce działka ewidencyjna z ULDK, poza Polską pole z
    Fields of The World (globalna mapa granic pól z Sentinel-2);
    source=ftw: zawsze FTW (także w Polsce — pole uprawne zamiast działki);
    source=uldk: zawsze ULDK."""
    import core
    import water_geometry as wg
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Nieprawidłowe współrzędne.")
    if source not in ("auto", "ftw", "uldk"):
        raise HTTPException(status_code=422, detail="source: auto | ftw | uldk")
    use_uldk = source == "uldk" or (source == "auto" and wg.in_poland_bbox(lon, lat))
    if use_uldk:
        try:
            geom, meta = core.fetch_parcel_at_point(lon, lat)
        except core.AppError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"source": "uldk", "parcel_id": meta.get("parcel_id"), "geometry_wkt": geom.wkt,
                "area_ha": wg.area_ha(geom), "gmina": meta.get("gmina")}
    countries = wg.ftw_candidate_countries(lon, lat)
    if not countries:
        raise HTTPException(status_code=404, detail=(
            "Gotowe obrysy pól są dostępne w Polsce (ULDK) oraz w Uzbekistanie, Tadżykistanie, "
            "Kirgistanie, Turkmenistanie i Kazachstanie (Fields of The World). "
            "Tutaj narysuj obrys ręcznie."))
    try:
        hit = wg.ftw_field_at(lon, lat)
    except Exception as e:
        log.warning("FTW niedostępne (%s, %s): %s", lat, lon, e)
        raise HTTPException(status_code=502, detail=f"Baza obrysów pól (FTW) niedostępna: {e}")
    if hit is None:
        raise HTTPException(status_code=404, detail=(
            "W tym miejscu baza Fields of The World nie ma pola (zabudowa, droga albo pole "
            "nierozpoznane przez model). Narysuj obrys ręcznie."))
    return hit


class WaterFieldImportRequest(BaseModel):
    filename: str = Field(..., max_length=200)
    content: str = Field(..., max_length=20_000_000)
    irrigation_mode: str = Field("rainfed", pattern="^(irrigation|rainfed)$",
                                 description="Domyślny tryb dla pól bez kolumny irrigation_mode")
    region: Optional[str] = Field(None, max_length=80)
    crop_type: Optional[str] = Field(None, max_length=40)


@router.post("/field/import")
def water_field_import(req: WaterFieldImportRequest, _auth=Depends(require_operator_key)):
    """Import wielu pól z GeoJSON/CSV (water_geometry.parse_fields_text).
    Pola bez geometrii z polskim numerem działki dostają obrys z ULDK.
    Pola, których parcel_id już jest aktywne w module, są pomijane."""
    import core
    import water_geometry as wg
    try:
        parsed = wg.parse_fields_text(req.filename, req.content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Nie udało się odczytać pliku: {e}")
    active = {f["parcel_id"] for f in db.get_water_fields(status="active")}
    added, skipped, errors = [], [], []
    for f in parsed:
        pid = f["parcel_id"]
        if pid in active:
            skipped.append({"parcel_id": pid, "reason": "już jest w module"})
            continue
        wkt = f.get("geometry_wkt")
        try:
            if not wkt:
                if not wg.is_polish_parcel_ids(pid):
                    raise core.AppError("brak geometrii, a ID nie jest numerem działki z ULDK")
                wkt = wg.geometry_from_parcel_ids(pid)["geometry_wkt"]
            extra = {k: f[k] for k in wg.FIELD_PROPS if f.get(k) not in (None, "")}
            if req.region and "region" not in extra:
                extra["region"] = req.region
            if req.crop_type and "crop_type" not in extra:
                extra["crop_type"] = req.crop_type
            from shapely import wkt as shapely_wkt
            wg.check_field_extent(shapely_wkt.loads(wkt), label=f"Pole {pid}")
            fid = db.add_water_field(pid, f.get("irrigation_mode") or req.irrigation_mode, wkt, **extra)
            added.append({"id": fid, "parcel_id": pid})
            active.add(pid)
        except Exception as e:
            errors.append({"parcel_id": pid, "error": str(e)})
    return {"added": added, "skipped": skipped, "errors": errors}


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


@router.delete("/irrigation/{event_id}")
def irrigation_event_delete(event_id: int, _auth=Depends(require_operator_key)):
    """Usuwa błędny wpis z dziennika (np. pomyłka w dacie albo dawce)."""
    if not db.delete_irrigation_event(event_id):
        raise HTTPException(status_code=404, detail="Nie ma takiego wpisu nawodnienia")
    return {"status": "deleted"}


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


class RainReadingRequest(BaseModel):
    reading_date: str = Field(..., description="RRRR-MM-DD")
    precip_mm: float = Field(..., ge=0, le=500)
    notes: Optional[str] = Field(None, max_length=500)

    @field_validator("reading_date")
    @classmethod
    def _validate_reading_date(cls, v):
        try:
            d = date.fromisoformat(v)
        except ValueError:
            raise ValueError("reading_date musi być w formacie RRRR-MM-DD")
        if d > date.today():
            raise ValueError("reading_date nie może być z przyszłości")
        return v


@router.post("/field/{water_field_id}/rain")
def rain_reading_add(water_field_id: int, req: RainReadingRequest,
                     _auth=Depends(require_operator_key)):
    """Odczyt deszczomierza rolnika. W bilansie wodnym zastępuje opad z
    modelu pogodowego w danym dniu (patrz water_balance.apply_rain_gauge)."""
    _get_field_or_404(water_field_id)
    rid = db.add_rain_reading(water_field_id, req.reading_date, req.precip_mm, req.notes)
    return {"id": rid, "status": "added"}


@router.get("/field/{water_field_id}/rain")
def rain_reading_list(water_field_id: int):
    _get_field_or_404(water_field_id)
    return db.get_rain_readings(water_field_id)


@router.delete("/rain/{reading_id}")
def rain_reading_delete(reading_id: int, _auth=Depends(require_operator_key)):
    if not db.delete_rain_reading(reading_id):
        raise HTTPException(status_code=404, detail="Nie ma takiego odczytu")
    return {"status": "deleted"}


class FieldSoilRequest(BaseModel):
    """Gleba pola ustalona przez doradcę. Pusty string / null czyści wartość."""
    texture_class: Optional[str] = Field(None, max_length=30)
    root_limit_cm: Optional[int] = Field(None, ge=10, le=300)
    salinity_class: Optional[str] = Field(None, max_length=20)
    ec_value: Optional[float] = Field(None, ge=0, le=100, description="dS/m (mS/cm)")
    ec_method: Optional[str] = Field(None, max_length=10, description="ece | ec_1_5 | ec_water")
    gw_depth_m: Optional[float] = Field(None, ge=0, le=30)


def _soil_payload(field: dict, fetch: bool = True, quick: bool = True) -> dict:
    import salinity_scan
    import soil_water
    t = soil_water.field_taw(field, fetch=fetch, quick=quick)
    scan = db.get_salinity_scan(field["id"])
    row = db.get_field_soil(field["id"]) or {}
    return {**t, "sg_fetched_at": row.get("sg_fetched_at"), "sg_error": row.get("sg_error"),
            "salinity": soil_water.field_salinity(field, row, t.get("texture")),
            "manual_taw_mm": field.get("soil_taw_mm"),
            "salinity_scan": scan[0] if scan else None,
            "salinity_scan_pu": salinity_scan.estimate_pu(field),
            "classes": [{"key": k, "awc_mm_m": round(soil_water.awc_mm_per_m(s, c))}
                        for k, (s, c) in soil_water.TEXTURE_CLASSES.items()],
            "attribution": soil_water.SOILGRIDS_ATTRIBUTION if t.get("texture_from") == "soilgrids" or t.get("sand") else None}


@router.get("/field/{water_field_id}/soil")
def field_soil_get(water_field_id: int):
    """Pojemność wodna gleby pola i jej źródło (soil_water.field_taw)."""
    return _soil_payload(_get_field_or_404(water_field_id))


@router.put("/field/{water_field_id}/soil")
def field_soil_set(water_field_id: int, req: FieldSoilRequest, _auth=Depends(require_operator_key)):
    import soil_water
    field = _get_field_or_404(water_field_id)
    changes = req.model_dump(exclude_unset=True)
    if "texture_class" in changes:
        tc = (changes["texture_class"] or "").strip() or None
        if tc is not None and tc not in soil_water.TEXTURE_CLASSES:
            raise HTTPException(status_code=422, detail=f"Nieznany gatunek gleby: {tc}")
        changes["texture_class"] = tc
    if "salinity_class" in changes:
        sc = (changes["salinity_class"] or "").strip() or None
        if sc is not None and sc not in soil_water.SALINITY_CLASSES:
            raise HTTPException(status_code=422, detail=f"Nieznana klasa zasolenia: {sc}")
        changes["salinity_class"] = sc
    if "ec_method" in changes:
        em = (changes["ec_method"] or "").strip() or None
        if em is not None and em not in soil_water.EC_METHODS:
            raise HTTPException(status_code=422, detail=f"Nieznana metoda pomiaru EC: {em}")
        changes["ec_method"] = em
    if changes.get("ec_value") is not None and not (changes.get("ec_method") or
                                                    (db.get_field_soil(water_field_id) or {}).get("ec_method")):
        raise HTTPException(status_code=422, detail="Podaj metodę pomiaru EC (ece, ec_1_5, ec_water).")
    if changes:
        db.save_field_soil(water_field_id, **changes)
    return _soil_payload(field)


@router.post("/field/{water_field_id}/soil/refresh")
def field_soil_refresh(water_field_id: int, _auth=Depends(require_operator_key)):
    """Pobiera profil SoilGrids na nowo (np. po zmianie granic pola)."""
    field = _get_field_or_404(water_field_id)
    db.save_field_soil(water_field_id, sg_profile_json=None, sg_fetched_at=None, sg_error=None, sg_error_at=None)
    return _soil_payload(field, quick=False)


@router.post("/field/{water_field_id}/salinity_scan")
def field_salinity_scan(water_field_id: int, _auth=Depends(require_operator_key)):
    """Podejrzenie zasolenia z 6 sezonów Sentinel-2 (salinity_scan.py). ~1 PU/rok, do ~1 min."""
    import salinity_scan
    field = _get_field_or_404(water_field_id)
    if not field.get("geometry_wkt"):
        raise HTTPException(status_code=422, detail="Pole nie ma obrysu.")
    try:
        return salinity_scan.scan_field(field)
    except Exception as e:
        log.exception("Skan zasolenia pola %s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Skan zasolenia nie powiódł się: {e}")


@router.get("/field/{water_field_id}/salinity_scan.png")
def field_salinity_scan_png(water_field_id: int):
    from fastapi.responses import Response
    got = db.get_salinity_scan(water_field_id, with_png=True)
    if not got or not got[1]:
        raise HTTPException(status_code=404, detail="Brak skanu zasolenia dla tego pola.")
    return Response(content=got[1], media_type="image/png", headers={"Cache-Control": "no-cache"})


class SoilCheckRequest(BaseModel):
    check_date: str = Field(..., description="RRRR-MM-DD")
    level: Optional[str] = Field(None, pattern="^(wet|moist|medium|dry|very_dry)$",
                                 description="ocena „gleba w dłoni” (FAO/USDA)")
    fill_pct: Optional[float] = Field(None, ge=0, le=100, description="% wody dostępnej (np. z czujnika)")
    method: str = Field("feel", pattern="^(feel|sensor|other)$")
    depth_cm: Optional[int] = Field(None, ge=5, le=200)
    notes: Optional[str] = Field(None, max_length=500)

    @field_validator("check_date")
    @classmethod
    def _validate_check_date(cls, v):
        try:
            d = date.fromisoformat(v)
        except ValueError:
            raise ValueError("check_date musi być w formacie RRRR-MM-DD")
        if d > date.today():
            raise ValueError("check_date nie może być z przyszłości")
        return v


def soil_check_fill(level: Optional[str], fill_pct: Optional[float]) -> float:
    import water_balance
    if fill_pct is not None:
        return float(fill_pct)
    if level is None:
        raise HTTPException(status_code=422, detail="Podaj ocenę (level) albo % wody dostępnej (fill_pct).")
    return water_balance.SOIL_FEEL_LEVELS[level]


@router.post("/field/{water_field_id}/soil_check")
def soil_check_add(water_field_id: int, req: SoilCheckRequest, _auth=Depends(require_operator_key)):
    """Punkt kontrolny wilgotności: ocena w polu ustawia stan profilu
    w bilansie wodnym w danym dniu (water_balance.run_balance)."""
    _get_field_or_404(water_field_id)
    fill = soil_check_fill(req.level, req.fill_pct)
    cid = db.add_soil_check(water_field_id, req.check_date, fill, method=req.method, level=req.level,
                            depth_cm=req.depth_cm, source="manual", notes=req.notes)
    return {"id": cid, "fill_pct": fill}


@router.get("/field/{water_field_id}/soil_checks")
def soil_check_list(water_field_id: int):
    _get_field_or_404(water_field_id)
    return db.get_soil_checks(water_field_id)


@router.delete("/soil_check/{check_id}")
def soil_check_delete(check_id: int, _auth=Depends(require_operator_key)):
    if not db.delete_soil_check(check_id):
        raise HTTPException(status_code=404, detail="Nie ma takiego punktu kontrolnego")
    return {"status": "deleted"}


class BackfillRequest(BaseModel):
    days: int = Field(90, ge=7, le=365, description="Ile dni wstecz uzupełnić (gdy brak dat)")
    date_from: Optional[str] = Field(None, description="RRRR-MM-DD — początek okresu")
    date_to: Optional[str] = Field(None, description="RRRR-MM-DD — koniec okresu (domyślnie dziś)")

    @field_validator("date_from", "date_to")
    @classmethod
    def _validate_dates(cls, v):
        if v in (None, ""):
            return None
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("Data musi być w formacie RRRR-MM-DD")
        return v


BACKFILL_MAX_DAYS = 400


@router.post("/field/{water_field_id}/backfill")
def water_field_backfill(water_field_id: int, req: BackfillRequest,
                         _auth=Depends(require_operator_key)):
    """Uzupełnia historię obserwacji pola (water_replay.backfill_field) —
    ostatnie `days` dni albo dowolny okres date_from..date_to (np. cały
    sezon kwiecień–wrzesień do powtórki). Kosztuje PU (jedno zapytanie S2 +
    jedno S1 na cały okres)."""
    field = _get_field_or_404(water_field_id)
    import water_replay
    t_end = date.fromisoformat(req.date_to) if req.date_to else date.today()
    t_end = min(t_end, date.today())
    t_start = date.fromisoformat(req.date_from) if req.date_from else t_end - timedelta(days=req.days)
    if t_start >= t_end:
        raise HTTPException(status_code=422, detail="Początek okresu musi być przed końcem.")
    if (t_end - t_start).days > BACKFILL_MAX_DAYS:
        raise HTTPException(status_code=422, detail=f"Okres dłuższy niż {BACKFILL_MAX_DAYS} dni — podziel go na części.")
    try:
        res = water_replay.backfill_field(field, t_start, t_end)
        res.update({"date_from": t_start.isoformat(), "date_to": t_end.isoformat()})
        return res
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


class ReplayRatingRequest(BaseModel):
    obs_date: str = Field(..., description="RRRR-MM-DD")
    rating: str = Field(..., pattern="^(confirmed|rejected|missed|unsure)$")
    risk_type: Optional[str] = Field(None, max_length=60)
    note: Optional[str] = Field(None, max_length=1000)


@router.post("/field/{water_field_id}/replay_rating")
def water_replay_rate(water_field_id: int, req: ReplayRatingRequest,
                      _auth=Depends(require_operator_key)):
    """Ocena wyniku powtórki: alarm trafny / fałszywy albo przeoczony problem."""
    field = _get_field_or_404(water_field_id)
    import water_replay
    try:
        return water_replay.rate_replay(field, req.obs_date, req.rating,
                                        risk_type=req.risk_type or "", note=req.note)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/replay_ratings")
def water_replay_ratings(water_field_id: Optional[int] = None):
    return db.get_replay_ratings(water_field_id)


@router.delete("/replay_rating/{rating_id}")
def water_replay_rating_delete(rating_id: int, _auth=Depends(require_operator_key)):
    if not db.delete_replay_rating(rating_id):
        raise HTTPException(status_code=404, detail="Nie ma takiej oceny")
    return {"status": "deleted"}


@router.get("/calibration/suggest_from_replay")
def water_calibration_suggest_from_replay():
    """Sugestia progu z ocen powtórki (oba rodzaje błędów). Zastosowanie —
    jak dotąd: POST /api/water/calibration/apply."""
    import water_replay
    return water_replay.suggest_threshold_from_ratings()


@router.get("/field/{water_field_id}/balance")
def water_field_balance(water_field_id: int):
    """Bilans wodny strefy korzeniowej (FAO-56) z sugestią terminu i dawki
    nawodnienia — narzędzie doradcze, nic nie zapisuje (plan dodaje
    człowiek). Źródła: Open-Meteo (bez klucza, bez PU), NDVI z historii
    obserwacji pola, log nawodnień."""
    field = _get_field_or_404(water_field_id)
    try:
        import water_balance
        return water_balance.compute_field_balance(field, quick_soil=True)
    except Exception as e:
        log.exception("Blad bilansu wodnego dla pola id=%s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Blad bilansu wodnego: {e}")


class OptramCalibrateRequest(BaseModel):
    months: int = Field(12, ge=3, le=24, description="Długość okresu kalibracji (miesiące wstecz od dziś).")
    buffer_m: float = Field(1500.0, ge=300, le=3000, description="Zapas regionu kalibracji wokół pola (m).")


@router.post("/field/{water_field_id}/optram/calibrate")
def water_optram_calibrate(water_field_id: int, req: OptramCalibrateRequest,
                           _auth=Depends(require_operator_key)):
    """Kalibracja trapezu OPTRAM dla regionu wokół pola + seria wilgotności
    W za ten sam okres (optram.py). Kosztowne: jedno zapytanie Process API
    (do 60 scen) i jedno Statistical API."""
    import optram
    field = _get_field_or_404(water_field_id)
    end = date.today()
    start = end - timedelta(days=int(req.months * 30.4))
    try:
        return optram.calibrate_field(field, start, end, buffer_m=req.buffer_m)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Blad kalibracji OPTRAM dla pola id=%s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Blad kalibracji OPTRAM: {e}")


@router.get("/field/{water_field_id}/optram")
def water_optram_get(water_field_id: int):
    import optram
    _get_field_or_404(water_field_id)
    return optram.get_field_optram(water_field_id)


@router.post("/field/{water_field_id}/optram/refresh")
def water_optram_refresh(water_field_id: int, _auth=Depends(require_operator_key)):
    import optram
    field = _get_field_or_404(water_field_id)
    if not db.get_optram_params(water_field_id):
        raise HTTPException(status_code=409, detail="Pole nie ma jeszcze kalibracji OPTRAM.")
    try:
        n = optram.update_recent(field, days=45, force=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Blad OPTRAM: {e}")
    return {"added": n, **optram.get_field_optram(water_field_id)}


@router.delete("/field/{water_field_id}/optram")
def water_optram_delete(water_field_id: int, _auth=Depends(require_operator_key)):
    _get_field_or_404(water_field_id)
    db.delete_optram(water_field_id)
    return {"deleted": True}


class ThermalFetchRequest(BaseModel):
    months: int = Field(6, ge=1, le=24, description="Ile miesięcy wstecz pobrać (Landsat na CDSE od 2021).")


@router.post("/field/{water_field_id}/thermal/fetch")
def water_thermal_fetch(water_field_id: int, req: ThermalFetchRequest,
                        _auth=Depends(require_operator_key)):
    """Termika Landsat 8/9 dla pola (thermal.py): jedno zapytanie Process API
    (do 60 scen) dla kwadratu ~3 km wokół pola."""
    import thermal
    field = _get_field_or_404(water_field_id)
    end = date.today()
    try:
        rows = thermal.fetch_field(field, end - timedelta(days=int(req.months * 30.4)), end)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Blad termiki dla pola id=%s", water_field_id)
        raise HTTPException(status_code=502, detail=f"Blad termiki Landsat: {e}")
    return {"added": len(rows), **thermal.get_field_thermal(field)}


@router.get("/field/{water_field_id}/thermal")
def water_thermal_get(water_field_id: int):
    import thermal
    return thermal.get_field_thermal(_get_field_or_404(water_field_id))


@router.delete("/field/{water_field_id}/thermal")
def water_thermal_delete(water_field_id: int, _auth=Depends(require_operator_key)):
    _get_field_or_404(water_field_id)
    db.delete_thermal(water_field_id)
    return {"deleted": True}


class PrepareStepRequest(BaseModel):
    step: str = Field(..., pattern="^(backfill|optram|thermal|soil|check_now)$")


@router.get("/field/{water_field_id}/prepare_plan")
def water_field_prepare_plan(water_field_id: int):
    """Co trzeba zrobić, żeby „Pełny obraz” miał wszystkie źródła, i ile to
    kosztuje (szacunek PU) — bez wykonywania czegokolwiek."""
    import water_overview
    return water_overview.prepare_plan(_get_field_or_404(water_field_id))


@router.post("/field/{water_field_id}/prepare_step")
def water_field_prepare_step(water_field_id: int, req: PrepareStepRequest,
                             _auth=Depends(require_operator_key)):
    import water_overview
    return water_overview.run_prepare_step(_get_field_or_404(water_field_id), req.step)


@router.get("/field/{water_field_id}/overview")
def water_field_overview(water_field_id: int):
    """„Pełny obraz pola”: ocena z bilansu, satelity, OPTRAM, termiki i pogody
    + wspólna oś czasu 0–100 (water_overview.py). Bez zapytań do satelitów."""
    import water_overview
    return water_overview.build_overview(_get_field_or_404(water_field_id))


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
    notify_farmer: bool = Field(
        False, description="Przy potwierdzeniu: wyślij rolnikowi SMS z linkiem do "
        "jego panelu (farmer_panel.notify_confirmed_alarm).")


@router.get("/triage_queue")
def water_triage_queue(status: str = "pending"):
    alarms = db.get_water_alarms(triage_status=status if status != "all" else None)
    # Czy da się wysłać SMS rolnikowi przy potwierdzeniu (checkbox w UI).
    phones = {f["id"]: bool((f.get("farmer_phone") or "").strip()) for f in db.get_water_fields()}
    for a in alarms:
        a["has_farmer_phone"] = phones.get(a["water_field_id"], False)
    return alarms


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
    out = {"id": alarm_id, "triage_status": req.triage_status}
    if req.triage_status == "confirmed" and req.notify_farmer:
        import farmer_panel
        try:
            out["sms"] = farmer_panel.notify_confirmed_alarm(alarm_id)
        except Exception as e:  # SMS nie może zablokować decyzji doradcy
            log.error(f"SMS po potwierdzeniu alertu {alarm_id}: {e}")
            out["sms"] = {"status": "failed", "error": str(e)}
    return out


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
