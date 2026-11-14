"""API analiz wieloletnich stref plonu (/api/yield/analysis).

Logika w yield_multi.py, raport PDF w yield_report.py. Liczenie (N lat ×
jedno zapytanie Process API) trwa do kilku minut, więc POST startuje
wątek w tle, a UI odpytuje listę o status (wzorzec status pending/
running/done/error jak w yield_maps).
"""
import json
import logging
import os
import re
from threading import Thread
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

import db
import yield_multi as ymu
import yield_mvi as ym
from auth import require_operator_key

log = logging.getLogger("yield_multi_api")
router = APIRouter(prefix="/api/yield/analysis", tags=["yield"])


class YearEntry(BaseModel):
    year: int = Field(..., ge=2017, le=2100)
    crop: str = Field(..., min_length=2, max_length=40)
    yield_t_ha: float = Field(..., gt=0, le=200)
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class FertParams(BaseModel):
    fert_name: Optional[str] = Field(None, max_length=80, description="np. NPK 6-20-30, mocznik")
    avg_dose_kg_ha: float = Field(..., gt=0, le=5000)
    strategy: str = Field("potencjal")
    rounding_kg: int = Field(5, ge=1, le=100)
    dose_min_kg_ha: Optional[float] = Field(None, ge=0)
    dose_max_kg_ha: Optional[float] = Field(None, gt=0)

    @field_validator("strategy")
    @classmethod
    def _s(cls, v):
        if v not in ymu.STRATEGIES:
            raise ValueError(f"strategia: {' albo '.join(ymu.STRATEGIES)}")
        return v


class AnalysisRequest(BaseModel):
    label: Optional[str] = Field(None, max_length=120)
    parcel_id: Optional[str] = Field(None, max_length=600)
    geometry_wkt: Optional[str] = None
    years: List[YearEntry] = Field(..., min_length=2, max_length=10)
    fertilizer: Optional[FertParams] = None

    @field_validator("years")
    @classmethod
    def _unique_years(cls, v):
        ys = [e.year for e in v]
        if len(set(ys)) != len(ys):
            raise ValueError("lata w płodozmianie muszą być różne")
        return v


def _resolve_geometry(parcel_id, geometry_wkt):
    import core
    import water_geometry as wg
    from shapely import wkt as shapely_wkt
    if geometry_wkt:
        geom = shapely_wkt.loads(geometry_wkt)
    elif parcel_id and wg.is_polish_parcel_ids(parcel_id):
        try:
            geom = shapely_wkt.loads(wg.geometry_from_parcel_ids(parcel_id)["geometry_wkt"])
        except core.AppError as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        raise HTTPException(status_code=422, detail="Podaj numer działki TERYT albo obrys pola (mapa / WKT).")
    try:
        wg.check_field_extent(geom)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return geom


def _run_analysis(aid: int, geom, years: list, fert: Optional[dict]):
    try:
        stats = ymu.build_analysis(geom, years, aid)
        if fert:
            try:
                f = ymu.fertilizer_doses(stats["zones"], fert["avg_dose_kg_ha"], fert["strategy"],
                                         fert.get("rounding_kg", 5), fert.get("dose_min_kg_ha"),
                                         fert.get("dose_max_kg_ha"))
                f["fert_name"] = fert.get("fert_name")
                stats["fertilizer"] = f
                _write_prescription(aid, f)
            except ValueError as e:
                stats["fertilizer_error"] = str(e)
        db.update_yield_analysis(aid, status="done", stats_json=json.dumps(stats))
    except Exception as e:
        log.exception("Analiza wieloletnia %s nieudana", aid)
        db.update_yield_analysis(aid, status="error", error=str(e))


def _write_prescription(aid: int, fert: dict):
    adir = os.path.join(ymu.ANALYSES_DIR, str(aid))
    gj = ymu.prescription_geojson(adir, fert)
    with open(os.path.join(adir, "dawki.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f)
    ymu.prescription_shapefile_zip(gj, os.path.join(adir, "dawki_shp.zip"))


@router.post("")
def analysis_create(req: AnalysisRequest, _auth=Depends(require_operator_key)):
    geom = _resolve_geometry(req.parcel_id, req.geometry_wkt)
    years = [e.model_dump() for e in req.years]
    fert = req.fertilizer.model_dump() if req.fertilizer else None
    aid = db.add_yield_analysis(label=req.label or req.parcel_id, parcel_id=req.parcel_id,
                                geometry_wkt=geom.wkt, years_json=json.dumps(years),
                                fert_json=json.dumps(fert) if fert else None)
    db.update_yield_analysis(aid, status="running")
    Thread(target=_run_analysis, args=(aid, geom, years, fert), daemon=True).start()
    return {"id": aid, "status": "running"}


def _a_or_404(aid: int) -> dict:
    a = db.get_yield_analysis(aid)
    if not a:
        raise HTTPException(status_code=404, detail="Nie ma takiej analizy")
    return a


def _public(a: dict, with_geojson=False) -> dict:
    out = {k: a[k] for k in ("id", "label", "parcel_id", "status", "error", "created_at")}
    out["years"] = json.loads(a["years_json"]) if a.get("years_json") else []
    out["fertilizer_params"] = json.loads(a["fert_json"]) if a.get("fert_json") else None
    out["stats"] = json.loads(a["stats_json"]) if a.get("stats_json") else None
    soil = json.loads(a["soil_json"]) if a.get("soil_json") else None
    out["soil"] = soil
    if with_geojson:
        out["geometry_wkt"] = a.get("geometry_wkt")
        p = os.path.join(ymu.ANALYSES_DIR, str(a["id"]), "strefy.geojson")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out["zones_geojson"] = json.load(f)
    return out


@router.get("")
def analysis_list():
    return [_public(a) for a in db.get_yield_analyses()]


@router.get("/{aid}")
def analysis_get(aid: int):
    return _public(_a_or_404(aid), with_geojson=True)


@router.post("/{aid}/fertilizer")
def analysis_set_fertilizer(aid: int, req: FertParams, _auth=Depends(require_operator_key)):
    a = _a_or_404(aid)
    if a["status"] != "done" or not a.get("stats_json"):
        raise HTTPException(status_code=409, detail="Analiza jeszcze nie jest policzona.")
    stats = json.loads(a["stats_json"])
    try:
        f = ymu.fertilizer_doses(stats["zones"], req.avg_dose_kg_ha, req.strategy,
                                 req.rounding_kg, req.dose_min_kg_ha, req.dose_max_kg_ha)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    f["fert_name"] = req.fert_name
    stats["fertilizer"] = f
    stats.pop("fertilizer_error", None)
    _write_prescription(aid, f)
    db.update_yield_analysis(aid, stats_json=json.dumps(stats), fert_json=req.model_dump_json())
    return {"id": aid, "fertilizer": f}


@router.post("/{aid}/soil")
async def analysis_upload_soil(aid: int, file: UploadFile = File(...),
                               _auth=Depends(require_operator_key)):
    a = _a_or_404(aid)
    data = await file.read()
    if len(data) > 2_000_000:
        raise HTTPException(status_code=413, detail="Plik za duży (limit 2 MB).")
    try:
        parsed = ymu.parse_soil_csv(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    samples = parsed["samples"]
    zones_tif = os.path.join(ymu.ANALYSES_DIR, str(aid), "strefy.tif")
    if a["status"] == "done" and os.path.exists(zones_tif):
        samples = ymu.assign_sample_zones(samples, zones_tif)
    soil = {"filename": file.filename, "samples": samples, "warnings": parsed["warnings"],
            "zone_summary": ymu.soil_zone_summary(samples)}
    db.update_yield_analysis(aid, soil_json=json.dumps(soil))
    return {"id": aid, "n_samples": len(samples), "warnings": parsed["warnings"],
            "zone_summary": soil["zone_summary"]}


@router.get("/{aid}/report.pdf")
def analysis_report(aid: int):
    a = _a_or_404(aid)
    if a["status"] != "done":
        raise HTTPException(status_code=409, detail="Analiza jeszcze nie jest policzona.")
    import yield_report
    path = yield_report.build_analysis_pdf(_public(a, with_geojson=True))
    label = re.sub(r"[^\w\-]+", "_", (a.get("label") or f"analiza_{aid}"))[:60]
    return FileResponse(path, media_type="application/pdf",
                        filename=f"strefy_plonu_{label}.pdf")


_FILE_RE = re.compile(r"^(nakladka\.png|strefy\.geojson|strefy\.tif|rel_sredni\.tif|"
                      r"dawki\.geojson|dawki_shp\.zip|rok_\d{4}\.png|rel_\d{4}\.tif)$")
_MIME = {".png": "image/png", ".geojson": "application/geo+json", ".tif": "image/tiff",
         ".zip": "application/zip", ".pdf": "application/pdf"}


@router.get("/{aid}/file/{name}")
def analysis_file(aid: int, name: str):
    _a_or_404(aid)
    if not _FILE_RE.match(name):
        raise HTTPException(status_code=404, detail="Nieznany plik")
    path = os.path.join(ymu.ANALYSES_DIR, str(aid), name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Plik nie istnieje")
    ext = os.path.splitext(name)[1]
    return FileResponse(path, media_type=_MIME.get(ext, "application/octet-stream"),
                        filename=f"analiza_{aid}_{name}")


@router.delete("/{aid}")
def analysis_delete(aid: int, _auth=Depends(require_operator_key)):
    import shutil
    _a_or_404(aid)
    db.delete_yield_analysis(aid)
    shutil.rmtree(os.path.join(ymu.ANALYSES_DIR, str(aid)), ignore_errors=True)
    return {"status": "deleted"}
