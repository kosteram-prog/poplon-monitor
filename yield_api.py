"""API map stref plonu MVI (/api/yield). Logika w yield_mvi.py."""
import json
import logging
import os
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

import db
import yield_mvi
from auth import require_operator_key

log = logging.getLogger("yield_api")
router = APIRouter(prefix="/api/yield", tags=["yield"])


class YieldMapRequest(BaseModel):
    parcel_id: Optional[str] = Field(None, max_length=600, description="Numer(y) działek TERYT albo nazwa pola")
    geometry_wkt: Optional[str] = Field(None, description="Obrys pola WKT (WGS84); pusty = z ULDK po numerze działki")
    crop_type: str = Field(..., min_length=2, max_length=40)
    season_year: int = Field(..., ge=2017, le=2100)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    reported_yield_t_ha: float = Field(..., gt=0, le=200)
    label: Optional[str] = Field(None, max_length=120)

    @field_validator("date_from", "date_to")
    @classmethod
    def _d(cls, v):
        if v in (None, ""):
            return None
        date.fromisoformat(v)
        return v


@router.get("/crop_window")
def yield_crop_window(crop_type: str, season_year: int):
    a, b = yield_mvi.crop_window(crop_type, season_year)
    return {"date_from": a.isoformat(), "date_to": b.isoformat()}


@router.post("/mvi")
def yield_map_create(req: YieldMapRequest, _auth=Depends(require_operator_key)):
    """Liczy mapę stref plonu (synchronicznie — kilka do kilkunastu sekund,
    jedno zapytanie Process API). Kosztuje PU proporcjonalnie do liczby scen."""
    import core
    import water_geometry as wg
    from shapely import wkt as shapely_wkt
    if req.geometry_wkt:
        geom = shapely_wkt.loads(req.geometry_wkt)
    elif req.parcel_id and wg.is_polish_parcel_ids(req.parcel_id):
        try:
            geom = shapely_wkt.loads(wg.geometry_from_parcel_ids(req.parcel_id)["geometry_wkt"])
        except core.AppError as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        raise HTTPException(status_code=422, detail="Podaj numer działki TERYT albo obrys pola (mapa / WKT).")
    try:
        wg.check_field_extent(geom)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    d0, d1 = yield_mvi.crop_window(req.crop_type, req.season_year)
    t_start = date.fromisoformat(req.date_from) if req.date_from else d0
    t_end = date.fromisoformat(req.date_to) if req.date_to else d1
    if t_start >= t_end:
        raise HTTPException(status_code=422, detail="Początek okresu musi być przed końcem.")
    map_id = db.add_yield_map(label=req.label or req.parcel_id, parcel_id=req.parcel_id,
                              crop_type=req.crop_type, season_year=req.season_year,
                              date_from=t_start.isoformat(), date_to=t_end.isoformat(),
                              reported_yield_t_ha=req.reported_yield_t_ha, geometry_wkt=geom.wkt)
    try:
        stats = yield_mvi.build_yield_map(geom, req.reported_yield_t_ha, t_start, t_end, map_id)
    except ValueError as e:
        db.update_yield_map(map_id, "error", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Blad mapy plonu id=%s", map_id)
        db.update_yield_map(map_id, "error", error=str(e))
        raise HTTPException(status_code=502, detail=f"Błąd liczenia mapy plonu: {e}")
    db.update_yield_map(map_id, "done", stats_json=json.dumps(stats))
    return {"id": map_id, "status": "done", "stats": stats}


def _map_or_404(map_id: int) -> dict:
    m = db.get_yield_map(map_id)
    if not m:
        raise HTTPException(status_code=404, detail="Nie ma takiej mapy")
    return m


@router.get("/maps")
def yield_maps_list():
    out = []
    for m in db.get_yield_maps():
        stats = json.loads(m["stats_json"]) if m.get("stats_json") else None
        out.append({k: m[k] for k in ("id", "label", "parcel_id", "crop_type", "season_year", "date_from",
                                      "date_to", "reported_yield_t_ha", "status", "error", "created_at")}
                   | {"stats": stats})
    return out


@router.get("/map/{map_id}")
def yield_map_get(map_id: int):
    m = _map_or_404(map_id)
    m["stats"] = json.loads(m["stats_json"]) if m.get("stats_json") else None
    m.pop("stats_json", None)
    path = os.path.join(yield_mvi.YIELD_MAPS_DIR, str(map_id), "strefy.geojson")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            m["zones_geojson"] = json.load(f)
    return m


_FILES = {"overlay.png": ("nakladka.png", "image/png"),
          "plon_mvi.tif": ("plon_mvi.tif", "image/tiff"),
          "strefy.geojson": ("strefy.geojson", "application/geo+json")}


@router.get("/map/{map_id}/{name}")
def yield_map_file(map_id: int, name: str):
    _map_or_404(map_id)
    if name not in _FILES:
        raise HTTPException(status_code=404, detail="Nieznany plik")
    fname, mime = _FILES[name]
    path = os.path.join(yield_mvi.YIELD_MAPS_DIR, str(map_id), fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Plik nie istnieje")
    return FileResponse(path, media_type=mime, filename=f"mapa_{map_id}_{fname}")


@router.delete("/map/{map_id}")
def yield_map_delete(map_id: int, _auth=Depends(require_operator_key)):
    import shutil
    _map_or_404(map_id)
    db.delete_yield_map(map_id)
    shutil.rmtree(os.path.join(yield_mvi.YIELD_MAPS_DIR, str(map_id)), ignore_errors=True)
    return {"status": "deleted"}
