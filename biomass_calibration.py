"""Kalibracja modelu biomasy i pobrania N przez poplon z CIred-edge na
próbkach z pola (wizyty doradców LENs).

Dlaczego CIred-edge (CIre = B07/B05 − 1):
  * jest liniowy względem zawartości chlorofilu i azotu w łanie (Clevers &
    Gitelson 2013, pasma Sentinel-2), więc nadaje się do szacowania N;
  * w odróżnieniu od NDVI/NDRE nie nasyca się przy bujnym poplonie — w pracy
    Jennewein i in. 2022 (27 indeksów, ozime poplony zbożowe, Maryland) modele
    optyczne nasycały się ok. 1,9 t s.m./ha, a najlepsze były indeksy red-edge.

Dlaczego KALIBRACJA, a nie gotowe współczynniki: w literaturze nie ma
współczynników dla polskich mieszanek poplonowych (gorczyca, facelia,
rzodkiew, owies, wyka…), a obecny model w analytics.py pochodzi z kukurydzy.
Ten moduł pozwala zebrać pary (próbka z ramki ↔ najbliższa czysta scena S2)
i dopasować model na własnych danych:
  * biomasa:  liniowy  DM = a + b·CIre   albo  log-liniowy  ln DM = a + b·CIre
              (forma z Jennewein i in.; z poprawką Duana przy powrocie do t/ha),
  * azot:     N = a + b·CIre (liniowo, jak u Clevers & Gitelson) albo log.
Wybór formy i ocena: walidacja krzyżowa „leave-one-out” (RMSE_CV), dla
porównania — ten sam test dla NDRE, NDVI i obecnego modelu literaturowego.
Zatwierdzony model trafia do settings i od tej chwili analytics.estimate_biomass
używa go, gdy obserwacja ma CIre.
"""
import json
import logging
import math
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

import db
from auth import require_operator_key

log = logging.getLogger("biomass_calibration")
router = APIRouter(prefix="/api/biomass", tags=["biomass"])

MATCH_DAYS = 5               # max. odstęp próbka ↔ scena S2
MATCH_SEARCH_DAYS = 8        # okno pobierania scen wokół daty próbki
MIN_VALID_FRAC = 0.6
MIN_SAMPLES_FIT = 8
GOOD_SAMPLES = 15


# ---------------------------------------------------------------------------
# Dopasowanie modeli (bez I/O)
# ---------------------------------------------------------------------------

def _linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("Brak zmienności wskaźnika w próbkach.")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _fit_one(xs, ys, form):
    """(a, b, smear) — dla 'log' dopasowanie ln y i czynnik Duana."""
    if form == "log":
        ly = [math.log(y) for y in ys]
        a, b = _linfit(xs, ly)
        smear = sum(math.exp(l - (a + b * x)) for x, l in zip(xs, ly)) / len(xs)
        return a, b, smear
    a, b = _linfit(xs, ys)
    return a, b, 1.0


def _predict(a, b, smear, form, x):
    if form == "log":
        return max(0.0, math.exp(a + b * x) * smear)
    return max(0.0, a + b * x)


def evaluate_form(xs, ys, form) -> Optional[dict]:
    """Dopasowanie na całości + LOOCV. None, gdy forma nie pasuje do danych."""
    n = len(xs)
    if n < 3 or (form == "log" and min(ys) <= 0):
        return None
    try:
        a, b, smear = _fit_one(xs, ys, form)
    except ValueError:
        return None
    pred = [_predict(a, b, smear, form, x) for x in xs]
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum((y - p) ** 2 for y, p in zip(ys, pred)) / ss_tot if ss_tot > 0 else None
    cv_err = []
    for i in range(n):
        xi = xs[:i] + xs[i + 1:]
        yi = ys[:i] + ys[i + 1:]
        try:
            ai, bi, si = _fit_one(xi, yi, form)
        except ValueError:
            return None
        cv_err.append(ys[i] - _predict(ai, bi, si, form, xs[i]))
    rmse_cv = math.sqrt(sum(e * e for e in cv_err) / n)
    return {"form": form, "a": round(a, 5), "b": round(b, 5), "smear": round(smear, 5),
            "n": n, "r2": None if r2 is None else round(r2, 3), "rmse_cv": round(rmse_cv, 3),
            "x_min": round(min(xs), 4), "x_max": round(max(xs), 4),
            "mean_y": round(my, 3)}


def best_cire_model(xs, ys) -> Optional[dict]:
    cands = [m for m in (evaluate_form(xs, ys, "linear"), evaluate_form(xs, ys, "log")) if m]
    return min(cands, key=lambda m: m["rmse_cv"]) if cands else None


def literature_rmse(samples) -> Optional[dict]:
    """Błąd obecnego modelu literaturowego (EVI+NDRE, bez RVI) na próbkach."""
    import analytics
    pairs = []
    for s in samples:
        if s.get("evi") is None or s.get("ndre") is None:
            continue
        c = analytics.BIOMASS_COEFFICIENTS
        p = max(0.0, min(analytics.BIOMASS_MAX_T_HA, c["intercept"] + c["evi"] * s["evi"] + c["ndre"] * s["ndre"]))
        pairs.append((s["dm_t_ha"], p))
    if len(pairs) < 3:
        return None
    rmse = math.sqrt(sum((y - p) ** 2 for y, p in pairs) / len(pairs))
    bias = sum(p - y for y, p in pairs) / len(pairs)
    return {"n": len(pairs), "rmse": round(rmse, 3), "bias": round(bias, 3)}


def fit_report(samples: list) -> dict:
    """Wszystkie porównania na zebranych próbkach (te z dopasowaną sceną)."""
    usable = [s for s in samples if s.get("cire") is not None and s.get("dm_t_ha") is not None]
    out = {"n_samples": len(samples), "n_usable": len(usable), "min_samples": MIN_SAMPLES_FIT,
           "good_samples": GOOD_SAMPLES, "biomass": None, "n_uptake": None, "compare": [],
           "points": [{"id": s["id"], "cire": s["cire"], "dm_t_ha": s["dm_t_ha"],
                       "n_kg_ha": s.get("n_kg_ha"), "parcel_id": s["parcel_id"],
                       "sample_date": s["sample_date"]} for s in usable]}
    if len(usable) >= 3:
        xs = [s["cire"] for s in usable]
        ys = [s["dm_t_ha"] for s in usable]
        out["biomass"] = best_cire_model(xs, ys)
        out["biomass_all_forms"] = [m for m in (evaluate_form(xs, ys, "linear"),
                                                evaluate_form(xs, ys, "log")) if m]
        for key, label in (("ndre", "NDRE"), ("ndvi", "NDVI"), ("evi", "EVI2")):
            sub = [s for s in usable if s.get(key) is not None]
            if len(sub) >= 3:
                m = best_cire_model([s[key] for s in sub], [s["dm_t_ha"] for s in sub])
                if m:
                    out["compare"].append({"index": label, "n": m["n"], "form": m["form"],
                                           "rmse_cv": m["rmse_cv"], "r2": m["r2"]})
        out["literature"] = literature_rmse(usable)
        nsub = [s for s in usable if s.get("n_kg_ha") is not None]
        if len(nsub) >= 3:
            out["n_uptake"] = best_cire_model([s["cire"] for s in nsub], [s["n_kg_ha"] for s in nsub])
    out["can_apply_biomass"] = bool(out["biomass"]) and out["biomass"]["n"] >= MIN_SAMPLES_FIT
    out["can_apply_n"] = bool(out["n_uptake"]) and out["n_uptake"]["n"] >= MIN_SAMPLES_FIT
    out["warnings"] = _warnings(out)
    return out


def _warnings(rep: dict) -> list:
    w = []
    if rep["n_usable"] < MIN_SAMPLES_FIT:
        w.append(f"Za mało próbek z dopasowaną sceną ({rep['n_usable']}) — model można zatwierdzić "
                 f"od {MIN_SAMPLES_FIT}, wiarygodny jest od ok. {GOOD_SAMPLES}.")
    b = rep.get("biomass")
    if b:
        if b["x_max"] - b["x_min"] < 1.0:
            w.append("Próbki obejmują wąski zakres CIre — zbierz też próbki z bardzo słabych i "
                     "bardzo bujnych poplonów, inaczej model będzie ekstrapolował.")
        if b["r2"] is not None and b["r2"] < 0.4:
            w.append("Słaba zależność biomasy od CIre w próbkach (R² < 0,4) — sprawdź, czy daty "
                     "próbek i scen się zgadzają oraz czy ramki są reprezentatywne dla pola.")
    return w


# ---------------------------------------------------------------------------
# Dopasowanie próbki do sceny Sentinel-2
# ---------------------------------------------------------------------------

def _sample_geometry(sample: dict):
    from shapely import wkt as shapely_wkt
    import core
    if sample.get("geometry_wkt"):
        return shapely_wkt.loads(sample["geometry_wkt"])
    geom, _ = core.fetch_parcel_geometry(sample["parcel_id"])
    return geom


def match_scene(sample: dict) -> dict:
    """Najbliższa czysta scena S2 (±MATCH_DAYS) -> wskaźniki pola."""
    import analytics
    import core
    d = date.fromisoformat(sample["sample_date"])
    geom = _sample_geometry(sample)
    g_buf, *_ = core.inner_buffer(geom, 10.0)
    recs = analytics.fetch_multi_series_strict(g_buf, d - timedelta(days=MATCH_SEARCH_DAYS),
                                               d + timedelta(days=MATCH_SEARCH_DAYS))
    best = None
    for r in recs:
        if (r.get("valid_frac") or 0) < MIN_VALID_FRAC or r.get("cire") is None:
            continue
        rd = r["date"] if isinstance(r["date"], date) else date.fromisoformat(str(r["date"]))
        delta = abs((rd - d).days)
        if delta <= MATCH_DAYS and (best is None or delta < best[0]):
            best = (delta, rd, r)
    if best is None:
        return {"matched": False}
    delta, rd, r = best
    return {"matched": True, "scene_date": rd.isoformat(), "delta_days": delta,
            "cire": r.get("cire"), "ndre": r.get("ndre"), "ndvi": r.get("ndvi"),
            "evi": r.get("evi"), "valid_frac": r.get("valid_frac")}


def rematch(sample_id: int) -> dict:
    s = db.get_biomass_sample(sample_id)
    if not s:
        raise LookupError("Nie ma takiej próbki.")
    m = match_scene(s)
    db.update_biomass_sample_match(sample_id, m if m["matched"] else {})
    return db.get_biomass_sample(sample_id)


def apply_model(target: str, form: Optional[str] = None) -> dict:
    import analytics
    rep = fit_report(db.get_biomass_samples())
    key = "biomass" if target == "biomass" else "n_uptake"
    if not rep["can_apply_" + ("biomass" if target == "biomass" else "n")]:
        raise ValueError(f"Za mało próbek do zatwierdzenia modelu (potrzeba {MIN_SAMPLES_FIT}).")
    model = rep[key]
    if form and form != model["form"]:
        usable = [p for p in rep["points"] if (p["dm_t_ha"] if target == "biomass" else p["n_kg_ha"]) is not None]
        ys = [p["dm_t_ha"] if target == "biomass" else p["n_kg_ha"] for p in usable]
        model = evaluate_form([p["cire"] for p in usable], ys, form)
        if not model:
            raise ValueError("Ta forma modelu nie pasuje do danych.")
    from datetime import datetime
    model = {**model, "created_at": datetime.now().isoformat(timespec="seconds")}
    db.set_setting("biomass_cire_model" if target == "biomass" else "n_cire_model", json.dumps(model))
    analytics.reload_biomass_models()
    n_updated = analytics.recompute_stored_biomass() if target == "biomass" else 0
    return {"applied": target, "model": model, "recomputed_observations": n_updated}


def clear_model(target: str) -> None:
    import analytics
    db.set_setting("biomass_cire_model" if target == "biomass" else "n_cire_model", "")
    analytics.reload_biomass_models()
    if target == "biomass":
        analytics.recompute_stored_biomass()


def active_models() -> dict:
    import analytics
    analytics.reload_biomass_models()
    return {"biomass": analytics.BIOMASS_CIRE_MODEL, "n_uptake": analytics.N_CIRE_MODEL}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SampleRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=80)
    sample_date: str
    dm_t_ha: float = Field(..., ge=0, le=20, description="Sucha masa nadziemna, t/ha")
    n_kg_ha: Optional[float] = Field(None, ge=0, le=500, description="Azot w biomasie nadziemnej, kg/ha")
    mix: Optional[str] = Field(None, max_length=200)
    fresh_g: Optional[float] = Field(None, ge=0)
    frame_m2: Optional[float] = Field(None, gt=0, le=10)
    n_frames: Optional[int] = Field(None, ge=1, le=50)
    dm_pct: Optional[float] = Field(None, gt=0, le=100)
    n_pct: Optional[float] = Field(None, gt=0, le=10)
    geometry_wkt: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=1000)
    match_now: bool = True

    @field_validator("sample_date")
    @classmethod
    def _v_date(cls, v):
        try:
            d = date.fromisoformat(v)
        except ValueError:
            raise ValueError("sample_date w formacie RRRR-MM-DD")
        if d > date.today():
            raise ValueError("Data próbki nie może być z przyszłości")
        return v


class ApplyRequest(BaseModel):
    target: str = Field(..., pattern="^(biomass|n)$")
    form: Optional[str] = Field(None, pattern="^(linear|log)$")


@router.post("/samples")
def biomass_sample_add(req: SampleRequest, _auth=Depends(require_operator_key)):
    data = req.model_dump(exclude={"match_now"})
    sid = db.add_biomass_sample(**data)
    match = None
    if req.match_now:
        try:
            s = rematch(sid)
            match = {"matched": s.get("scene_date") is not None}
        except Exception as e:
            log.warning("Dopasowanie sceny dla próbki %s nieudane: %s", sid, e)
            match = {"matched": False, "error": str(e)[:300]}
    return {"id": sid, "match": match, "sample": db.get_biomass_sample(sid)}


@router.get("/samples")
def biomass_sample_list():
    return db.get_biomass_samples()


@router.delete("/samples/{sample_id}")
def biomass_sample_delete(sample_id: int, _auth=Depends(require_operator_key)):
    if not db.get_biomass_sample(sample_id):
        raise HTTPException(404, "Nie ma takiej próbki.")
    db.delete_biomass_sample(sample_id)
    return {"deleted": True}


@router.post("/samples/{sample_id}/rematch")
def biomass_sample_rematch(sample_id: int, _auth=Depends(require_operator_key)):
    try:
        return rematch(sample_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Nie udało się pobrać sceny Sentinel-2: {e}")


@router.get("/fit")
def biomass_fit():
    rep = fit_report(db.get_biomass_samples())
    rep["active"] = active_models()
    return rep


@router.post("/model/apply")
def biomass_model_apply(req: ApplyRequest, _auth=Depends(require_operator_key)):
    try:
        return apply_model(req.target, req.form)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.delete("/model/{target}")
def biomass_model_clear(target: str, _auth=Depends(require_operator_key)):
    if target not in ("biomass", "n"):
        raise HTTPException(404, "Nieznany model.")
    clear_model(target)
    return {"cleared": target}
