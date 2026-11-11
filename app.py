"""Serwer aplikacji weryfikacji poplonów.

Uruchomienie:
    export CDSE_CLIENT_ID="..."
    export CDSE_CLIENT_SECRET="..."
    uvicorn app:app --reload
Aplikacja: http://127.0.0.1:8000
"""
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import core
import db
from auth import require_operator_key
from photo_handler import router as photo_router
from calibration import router as calibration_router
from irrigation_log import router as irrigation_router
from yield_api import router as yield_router

Path("uploads/photos").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- podstawowa baza (monitored_parcels, ndvi_observations, alarms...) ---
    db.init_db()

    # --- kalibrowalne progi NDVI z bazy (jeśli operator je kiedyś zmienił) ---
    core.reload_thresholds()

    # --- kalibrowalny prog anomalii wodnej (jesli operator go kiedys zmienil) ---
    try:
        from water_stress import reload_water_threshold
        reload_water_threshold()
    except Exception as e:
        print(f"[WARN] reload_water_threshold: {e}")

    # --- kalibrowalne wagi Confidence Score z bazy (jeśli operator je zmienił) ---
    from analytics import reload_weights
    reload_weights()

    # --- tabele analityczne (multi_observations, agro_events, chart_data VIEW) ---
    try:
        from analytics import init_analytics_tables
        init_analytics_tables()
        print("[OK] Tabele analityczne (analytics) gotowe")
    except Exception as e:
        print(f"[WARN] init_analytics_tables: {e}")

    # --- tabele SAR Sentinel-1 (s1_observations, sar_events) ---
    try:
        from sentinel1 import init_s1_tables
        init_s1_tables()
        print("[OK] Tabele SAR (sentinel1) gotowe")
    except Exception as e:
        print(f"[WARN] init_s1_tables: {e}")

    # --- scheduler monitoringu (co 48h) ---
    try:
        import scheduler
        scheduler.start_scheduler()
        print("[OK] Scheduler uruchomiony")
    except Exception as e:
        print(f"[WARN] Scheduler nie uruchomiony: {e}")

    # --- cykliczne czyszczenie starych zadań batch ---
    purge_thread = threading.Thread(target=_purge_loop, daemon=True)
    purge_thread.start()

    yield

    _purge_thread_stop.set()

    try:
        import scheduler
        scheduler.stop_scheduler()
    except Exception:
        pass


app = FastAPI(title="Weryfikacja poplonów NDVI", lifespan=lifespan)
app.include_router(photo_router)
app.include_router(calibration_router)
app.include_router(irrigation_router)
app.include_router(yield_router)

log = logging.getLogger("app")


def _internal_error(e: Exception) -> HTTPException:
    """Buduje bezpieczną odpowiedź 500: pełny wyjątek (z traceback) trafia
    do logów serwera, a klient dostaje tylko typ wyjątku — bez treści
    komunikatu, która mogła zawierać wewnętrzne szczegóły (URL-e, ścieżki
    plików, fragmenty zapytań SQL/HTTP)."""
    log.exception("Nieobsłużony wyjątek w endpoincie API")
    return HTTPException(
        status_code=500,
        detail=f"Błąd wewnętrzny serwera ({type(e).__name__}). Sprawdź logi serwera.",
    )


# ---------------------------------------------------------------------------
# Jednorazowe sprawdzenie działki
# ---------------------------------------------------------------------------

class CheckRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    typ: str = Field(..., pattern="^(scierniskowy|ozimy)$")
    rok: int = Field(..., ge=2016, le=2100)
    sowing_date: Optional[str] = Field(None, description="RRRR-MM-DD")
    termination_date: Optional[str] = Field(None, description="RRRR-MM-DD")
    termination_method: Optional[str] = Field(
        None, pattern="^(walowanie|talerzowanie|przyoranie|oprysk|inne)$")


@app.post("/api/check")
def api_check(req: CheckRequest):
    try:
        return core.check_parcel(req.parcel_id, req.typ, req.rok,
                                 req.sowing_date, req.termination_date,
                                 req.termination_method)
    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error(e)


class SatelliteImageRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    date: str = Field(..., description="RRRR-MM-DD, zwykle verdict.peak_date")


@app.post("/api/satellite_image")
def api_satellite_image(req: SatelliteImageRequest):
    """Dociąga zdjęcie prawdziwokolorowe on-demand — używane, gdy operator
    otwiera szczegóły działki z listy wsadowej (batch celowo pomija ten
    fetch dla każdej pozycji, patrz core.check_parcel(fetch_image=False))."""
    try:
        from datetime import date as date_cls
        geom, _ = core.fetch_parcel_geometry(req.parcel_id)
        img = core.fetch_true_color_image(geom, date_cls.fromisoformat(req.date))
        if img is None:
            raise HTTPException(status_code=404,
                detail="Brak zdjęcia satelitarnego Sentinel-2 dla tej daty.")
        return img
    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=422, detail="Nieprawidłowy format daty (oczekiwano RRRR-MM-DD).")
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(e)


# ---------------------------------------------------------------------------
# Tryb wsadowy
# ---------------------------------------------------------------------------

MAX_BATCH_ITEMS = 200
# Konfigurowalne przez env. Optymalizacja (listopad 2026): domyślnie 6
# (dawniej 3). Działka kosztuje teraz ~2 zapytania i ~2 PU zamiast 3 zapytań
# i 5-14 PU (NDVI liczony z zapytania wielowskaźnikowego, UTM, wyjścia
# 16-bit), a sentinel1._post_with_retry sam pilnuje budżetu PU/zapytań na
# minutę i honoruje Retry-After — więcej równoległych działek nie grozi już
# seriami 429. Liczbę zapytań "w locie" i tak ogranicza
# SENTINEL_HUB_MAX_CONCURRENT.
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "6"))
JOB_TTL_SECONDS = 3600
MAX_CONCURRENT_BATCH_JOBS = 3       # ile batchy naraz mogą walić w Sentinel Hub
PURGE_INTERVAL_SECONDS = 300        # czyść stare zadania nawet gdy nikt nowych nie startuje

_jobs: dict = {}
_jobs_lock = threading.Lock()
_purge_thread_stop = threading.Event()


class BatchItem(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    typ: str = Field(..., pattern="^(scierniskowy|ozimy)$")
    rok: int = Field(..., ge=2016, le=2100)
    sowing_date: Optional[str] = Field(None, description="RRRR-MM-DD")
    termination_date: Optional[str] = Field(None, description="RRRR-MM-DD")
    termination_method: Optional[str] = Field(
        None, pattern="^(walowanie|talerzowanie|przyoranie|oprysk|inne)$")


class BatchStartRequest(BaseModel):
    items: List[BatchItem]


def _purge_old_jobs():
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items()
                   if now - j["created"] > JOB_TTL_SECONDS]
        for jid in expired:
            del _jobs[jid]


def _purge_loop():
    """Czyści przeterminowane zadania batch cyklicznie, niezależnie od tego,
    czy ktoś w międzyczasie startuje nowe zadania. Bez tego, zakończone
    zadania (z pełnymi seriami NDVI dla do 200 działek) mogły siedzieć w
    pamięci pełną godzinę — albo bezterminowo, jeśli nikt więcej nie
    wystartował nowego batcha."""
    while not _purge_thread_stop.wait(PURGE_INTERVAL_SECONDS):
        try:
            _purge_old_jobs()
        except Exception as e:
            print(f"[purge_loop] błąd: {e}")


def _count_active_jobs() -> int:
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if not j["finished"])


def _process_one(job_id: str, idx: int):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        item = job["items"][idx]
        item["state"] = "running"
    try:
        multi_payload = MultiCheckRequest(
            parcel_id=item["parcel_id"],
            typ=item["typ"],
            rok=item["rok"],
            sowing_date=item.get("sowing_date"),
            termination_date=item.get("termination_date"),
            include_s1=True,
        )

        # core.check_parcel (NDVI) i check_multi (EVI/SAVI/NDRE/NDWI + S1) to
        # dwa niezależne zestawy wywołań do Sentinel Hub (różne evalscripty) —
        # dotąd czekały na siebie sekwencyjnie, mimo że nic w drugim nie
        # zależy od wyniku pierwszego. Odpalamy je równolegle, co dla
        # pojedynczej pozycji batcha realnie tnie czas mniej więcej o połowę.
        # fetch_image=False: w liście wyników wsadowych zdjęcie satelitarne
        # nie jest renderowane (patrz showSatelliteImage w static/index.html
        # — używane dopiero po wejściu w szczegóły jednej działki), więc nie
        # ma sensu ciągnąć go dla każdej z np. 20 działek. Dociągane on-demand
        # przez /api/satellite_image, gdy operator faktycznie otworzy detale.
        with ThreadPoolExecutor(max_workers=2) as item_pool:
            fut_ndvi = item_pool.submit(
                core.check_parcel, item["parcel_id"], item["typ"], item["rok"],
                item.get("sowing_date"), item.get("termination_date"),
                item.get("termination_method"), fetch_image=False)
            fut_multi = item_pool.submit(check_multi, multi_payload)

            result = fut_ndvi.result()

            # Dokończ pełną analizę wielowskaźnikową i dopiero wtedy ustal kategorię.
            try:
                multi = fut_multi.result()
                if isinstance(multi, dict):
                    fused_verdict = multi.get("fused_verdict")
                    fused_category = multi.get("fused_category") or multi.get("category")
                    if fused_verdict:
                        # Zachowaj metryki NDVI i kontekst z szybkiej analizy,
                        # ale status/kategorię weź z pełnej analizy.
                        ndvi_verdict = result.get("verdict", {})
                        result["verdict"] = {
                            **ndvi_verdict,
                            **fused_verdict,
                            "metrics": ndvi_verdict.get("metrics", {}),
                            "termination": ndvi_verdict.get("termination"),
                        }
                    if fused_category:
                        result["category"] = fused_category
                    # Jawne powody, dla których działka nie jest zielona —
                    # wyświetlane na karcie w wynikach wsadowych.
                    result["category_reasons"] = multi.get("category_reasons") or []
                    # Zachowaj skrót wyników multi dla ewentualnego debug/podglądu.
                    result["multi_summary"] = multi.get("summary")
                    # Zachowaj PEŁNĄ odpowiedź check_multi — jest już policzona
                    # tutaj (i tak samo droga jak przy pojedynczym zapytaniu),
                    # więc karta szczegółów działki może ją odtworzyć od razu
                    # z pamięci zamiast wywoływać /api/check_multi ponownie
                    # (patrz static/index.html _startMultiLoad: multi_full).
                    # Ten sam dict jest już JSON-serializowalny (check_multi
                    # serializuje daty przez isoformat() przed zwróceniem).
                    result["multi_full"] = multi
                    result["multi_ready"] = True
            except Exception as multi_ex:
                # Fuzja S1/multi jest wzbogaceniem werdyktu NDVI, nie jego
                # zależnością blokującą. Awaria pełnej analizy (np. timeout/rate
                # limit Sentinel Hub) nie może niszczyć już policzonego,
                # poprawnego werdyktu NDVI — zachowujemy go bez zmian i tylko
                # dopisujemy notatkę informacyjną.
                v = result.get("verdict", {})
                details = list(v.get("details", []))
                details.append(
                    "Pełna analiza wielowskaźnikowa niedostępna "
                    f"({type(multi_ex).__name__}) — werdykt oparty wyłącznie na NDVI."
                )
                result["verdict"] = {
                    **v,
                    "details": details,
                }
                # Kategoria zostaje ta z NDVI (result["category"] się nie zmienia),
                # chyba że oryginalny werdykt NDVI już sam był niepewny.
                if v.get("status") in ("PRAWDOPODOBNY", "NIEROZSTRZYGNIETE"):
                    result["category"] = "yellow"
                result["multi_ready"] = False
                # KLUCZOWA informacja diagnostyczna: kategoria pochodzi z
                # samego NDVI, bo pełna analiza się nie powiodła — jeśli to
                # występuje masowo, to jest odpowiedź na pytanie "dlaczego
                # nic nie wpada w zielone / czemu liczy się tylko NDVI".
                result["category_reasons"] = [
                    f"Pełna analiza wielowskaźnikowa NIEDOSTĘPNA ({type(multi_ex).__name__}) "
                    "— kategoria wyłącznie z NDVI"
                ]
                log.warning(
                    "batch: pełna analiza niedostępna dla %s (%s: %s) — kategoria z samego NDVI",
                    item["parcel_id"], type(multi_ex).__name__, multi_ex)

        with _jobs_lock:
            item["state"] = "done"
            item["result"] = result
    except core.AppError as e:
        with _jobs_lock:
            item["state"] = "error"
            item["error"] = str(e)
    except Exception as e:
        with _jobs_lock:
            item["state"] = "error"
            item["error"] = f"{type(e).__name__}: {e}"
    with _jobs_lock:
        job["done_count"] += 1


def _run_batch(job_id: str):
    from sentinel1 import pu_stats
    with _jobs_lock:
        job = _jobs[job_id]
        n = len(job["items"])
    t0 = time.time()
    stats0 = pu_stats()
    with ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as pool:
        futures = [pool.submit(_process_one, job_id, i) for i in range(n)]
        for f in futures:
            f.result()
    with _jobs_lock:
        _jobs[job_id]["finished"] = True
    # Podsumowanie do monitoring.log — ile PU i czasu kosztował batch (przy
    # równoległych batchach liczniki są wspólne dla procesu, więc to górne
    # oszacowanie).
    stats1 = pu_stats()
    log.info("batch %s: %d działek w %.0f s, %.1f PU, %d zapytań do Sentinel Hub, %d odpowiedzi 429",
             job_id[:8], n, time.time() - t0, stats1["pu"] - stats0["pu"],
             stats1["requests"] - stats0["requests"],
             stats1["rate_limited"] - stats0["rate_limited"])


@app.post("/api/batch/start")
def batch_start(req: BatchStartRequest):
    _purge_old_jobs()
    if not req.items:
        raise HTTPException(status_code=422, detail="Brak działek do sprawdzenia.")
    if len(req.items) > MAX_BATCH_ITEMS:
        raise HTTPException(status_code=422,
            detail=f"Zbyt dużo działek naraz (max {MAX_BATCH_ITEMS}).")
    if _count_active_jobs() >= MAX_CONCURRENT_BATCH_JOBS:
        raise HTTPException(status_code=429,
            detail=f"Zbyt wiele równoczesnych zadań wsadowych (max {MAX_CONCURRENT_BATCH_JOBS}). "
                   "Poczekaj aż poprzednie się zakończą — to ochrona przed przekroczeniem "
                   "limitów Sentinel Hub.")
    job_id = uuid.uuid4().hex
    items = [{"parcel_id": it.parcel_id, "typ": it.typ, "rok": it.rok,
              "sowing_date": it.sowing_date,
              "termination_date": it.termination_date,
              "termination_method": it.termination_method,
              "state": "pending", "result": None, "error": None}
             for it in req.items]
    with _jobs_lock:
        _jobs[job_id] = {"created": time.time(), "items": items,
                         "done_count": 0, "finished": False}
    thread = threading.Thread(target=_run_batch, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id, "total": len(items)}


@app.get("/api/batch/status/{job_id}")
def batch_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404,
                detail="Nieznane zadanie (mogło wygasnąć — zadania żyją godzinę).")
        items_copy = [dict(it) for it in job["items"]]
        return {"total": len(items_copy), "done_count": job["done_count"],
                "finished": job["finished"], "items": items_copy}


# ---------------------------------------------------------------------------
# Monitoring — endpointy
# ---------------------------------------------------------------------------

class MonitorAddRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    typ: str = Field(..., pattern="^(scierniskowy|ozimy)$")
    rok: int = Field(..., ge=2016, le=2100)
    sowing_date: Optional[str] = None
    termination_date: Optional[str] = None
    termination_method: Optional[str] = None
    farmer_phone: Optional[str] = None
    farmer_name: Optional[str] = None
    notes: Optional[str] = None


@app.post("/api/monitor/add")
def monitor_add(req: MonitorAddRequest, _auth=Depends(require_operator_key)):
    kwargs = {k: v for k, v in req.model_dump().items()
              if k not in ("parcel_id", "typ", "rok") and v is not None}
    pid = db.add_monitored_parcel(req.parcel_id, req.typ, req.rok, **kwargs)
    return {"id": pid, "parcel_id": req.parcel_id, "status": "added"}


@app.get("/api/monitor/list")
def monitor_list(status: str = "active"):
    parcels = db.get_monitored_parcels(status=status if status != "all" else None)
    result = []
    for p in parcels:
        obs = db.get_observations(p["id"])
        alarms = db.get_alarms(p["id"], resolved=False)
        result.append({**p, "recent_observations": obs[-10:],
                       "active_alarms": alarms})
    return result


@app.delete("/api/monitor/{parcel_db_id}")
def monitor_delete(parcel_db_id: int, _auth=Depends(require_operator_key)):
    db.update_parcel_status(parcel_db_id, "completed")
    return {"status": "deactivated"}


@app.get("/api/monitor/{parcel_db_id}/observations")
def monitor_observations(parcel_db_id: int):
    return db.get_observations(parcel_db_id)


@app.post("/api/monitor/{parcel_db_id}/alarm/{alarm_id}/resolve")
def monitor_resolve_alarm(parcel_db_id: int, alarm_id: int, _auth=Depends(require_operator_key)):
    db.resolve_alarm(alarm_id)
    return {"status": "resolved"}


@app.post("/api/monitor/check_now")
def monitor_check_now(_auth=Depends(require_operator_key)):
    def run():
        try:
            import scheduler
            scheduler.check_all_monitored_parcels()
        except Exception as e:
            print(f"[check_now] błąd: {e}")
    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Statyczne pliki i strona główna
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sentinel-1 SAR — endpointy
# ---------------------------------------------------------------------------

class S1CheckRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    typ: str = Field(..., pattern="^(scierniskowy|ozimy)$")
    rok: int = Field(..., ge=2016, le=2100)
    sowing_date: Optional[str] = None
    termination_date: Optional[str] = None
    orbit_direction: Optional[str] = Field("DESCENDING", pattern="^(ASCENDING|DESCENDING)$")


@app.post("/api/s1/check")
def s1_check(req: S1CheckRequest):
    """Pobiera dane Sentinel-1 i wykrywa zdarzenia agrotechniczne dla działki."""
    try:
        from sentinel1 import (fetch_s1_series, assess_with_sar,
                               detect_tillage, init_s1_tables, save_s1_observations)
        from datetime import date as date_cls
        import core as core_mod

        geom, parcel_info = core_mod.fetch_parcel_geometry(req.parcel_id)
        geom_buf, _, _ = core_mod.inner_buffer(geom, 10.0)

        cfg = core_mod.CFG[req.typ]
        t_start = cfg["t_start"](req.rok)
        t_end   = min(cfg["t_end"](req.rok) if req.typ == "scierniskowy"
                      else cfg["t_end"](req.rok), date_cls.today())

        sowing_date = date_cls.fromisoformat(req.sowing_date) if req.sowing_date else None
        term_date   = date_cls.fromisoformat(req.termination_date) if req.termination_date else None

        s1_records = fetch_s1_series(
            geom_buf, t_start, t_end,
            orbit_direction=req.orbit_direction or "DESCENDING"
        )
        events = detect_tillage(s1_records, sowing_date, term_date)

        return {
            "parcel_id": req.parcel_id,
            "parcel_info": parcel_info,
            "s1_records": [
                {**r, "date": r["date"].isoformat() if hasattr(r["date"], "isoformat") else r["date"]}
                for r in s1_records
            ],
            "tillage_events": events,
            "summary": {
                "observations": len(s1_records),
                "alarm_events": sum(1 for e in events if e["confidence"] in ("high","medium") and not e["within_termination_window"]),
                "last_rvi": s1_records[-1]["rvi"] if s1_records else None,
                "last_vv_db": s1_records[-1]["vv_db"] if s1_records else None,
            }
        }
    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error(e)


@app.post("/api/check_fused")
def check_fused(req: CheckRequest):
    """Jednoczesne sprawdzenie NDVI (S2) + SAR (S1) z fuzją werdyktu."""
    try:
        from sentinel1 import fetch_s1_series, assess_with_sar
        from datetime import date as date_cls

        # 1. Standardowy werdykt NDVI
        ndvi_result = core.check_parcel(
            req.parcel_id, req.typ, req.rok,
            req.sowing_date, req.termination_date, req.termination_method
        )

        # 2. Pobierz S1 (nie blokuj jeśli brak danych)
        s1_records = []
        s1_error   = None
        try:
            geom, _ = core.fetch_parcel_geometry(req.parcel_id)
            geom_buf, _, _ = core.inner_buffer(geom, 10.0)
            cfg     = core.CFG[req.typ]
            t_start = cfg["t_start"](req.rok)
            t_end   = min(cfg["t_end"](req.rok), date_cls.today())
            from sentinel1 import fetch_s1_series
            s1_records = fetch_s1_series(geom_buf, t_start, t_end)
        except Exception as e:
            s1_error = str(e)

        # 3. Fuzja
        sowing_d = date_cls.fromisoformat(req.sowing_date) if req.sowing_date else None
        term_d   = date_cls.fromisoformat(req.termination_date) if req.termination_date else None
        fused = assess_with_sar(ndvi_result["verdict"], s1_records, sowing_d, term_d)

        return {**ndvi_result, "verdict": fused, "s1_records": [
            {**r, "date": r["date"].isoformat() if hasattr(r["date"], "isoformat") else r["date"]}
            for r in s1_records
        ], "s1_error": s1_error}

    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error(e)


@app.get("/api/monitor/{parcel_db_id}/s1")
def monitor_s1_history(parcel_db_id: int):
    """Historia obserwacji S1 dla działki z monitoringu."""
    try:
        from sentinel1 import get_s1_observations
        return get_s1_observations(parcel_db_id)
    except Exception as e:
        raise _internal_error(e)


@app.get("/api/monitor/{parcel_db_id}/chart")
def monitor_chart_data(parcel_db_id: int):
    """
    Dane do wykresu wielowskaźnikowego — wszystkie indeksy S2 + S1 per dzień.
    Format: [{date, ndvi, evi, savi, ndre, ndwi, confidence_score, vv_db, rvi, ...}]
    """
    try:
        from analytics import get_chart_data, init_analytics_tables
        init_analytics_tables()
        return get_chart_data(parcel_db_id)
    except Exception as e:
        raise _internal_error(e)


class MultiCheckRequest(BaseModel):
    parcel_id: str = Field(..., min_length=3, max_length=60)
    typ: str = Field(..., pattern="^(scierniskowy|ozimy)$")
    rok: int = Field(..., ge=2016, le=2100)
    sowing_date: Optional[str] = None
    termination_date: Optional[str] = None
    include_s1: bool = False   # czy pobierać dane SAR (wolniejsze)


@app.post("/api/check_multi")
def check_multi(req: MultiCheckRequest):
    """
    Pełna analiza wielowskaźnikowa: NDVI+EVI+SAVI+NDRE+NDWI + opcjonalnie S1.
    Zwraca Confidence Score, biomassę i zdarzenia agrotechniczne.
    """
    try:
        from analytics import (fetch_multi_series, compute_confidence_score,
                               estimate_biomass, detect_agro_events_fused,
                               select_primary_index, get_season_phase,
                               init_analytics_tables, save_multi_observation,
                               SCORE_THRESHOLDS)
        from datetime import date as date_cls

        geom, parcel_info = core.fetch_parcel_geometry(req.parcel_id)
        geom_buf, _, _ = core.inner_buffer(geom, 10.0)

        cfg     = core.CFG[req.typ]
        t_start = cfg["t_start"](req.rok)
        t_end   = min(cfg["t_end"](req.rok), date_cls.today())

        sowing_d = date_cls.fromisoformat(req.sowing_date) if req.sowing_date else None
        term_d   = date_cls.fromisoformat(req.termination_date) if req.termination_date else None

        # 1-2c. Pobranie danych: S2 (multi), opcjonalnie S1, pogoda i teren.
        # Optymalizacja (listopad 2026): dawniej szły po kolei (multi ~2,4 s
        # + S1 ~2,5 s + Open-Meteo ~1,2 s), choć żadne nie zależy od
        # pozostałych — teraz równolegle, czas ≈ najwolniejsze z nich.
        # Semantyka błędów bez zmian: błąd multi przerywa analizę, błędy
        # S1/pogody/terenu są NIEBLOKUJĄCE (patrz komentarze niżej).
        # 2b. Kontekst pogodowy (Open-Meteo, bezpłatne, bez klucza) —
        # NIEBLOKUJĄCY, dokładnie jak S1 powyżej: brak danych pogodowych nie
        # powinien uniemożliwiać oceny, tylko ją zubażać o jeden pomocniczy
        # sygnał. Używane w czterech miejscach: (a) detect_agro_events_fused
        # — odróżnienie przymrozku/posuchy od realnej orki, (b)
        # get_season_phase — realny (nie tylko kalendarzowy) start fazy
        # "winter" dla ozimych, (c) get_season_phase — fenologia licząca
        # się w GDD zamiast tygodni od siewu, (d) growing_conditions —
        # czysto informacyjne podsumowanie warunków wzrostu dla operatora
        # (patrz niżej, konstrukcja odpowiedzi — NIGDY nie zmienia
        # score/statusu, patrz weather.py).
        # 2c. Kontekst rzeźby terenu (Open-Meteo Elevation, bezpłatne, bez
        # klucza) — CZYSTO INFORMACYJNY, nie wchodzi do Confidence Score ani
        # do fused_status (patrz terrain.py — inny typ dowodu: nie "czy jest
        # okrywa", tylko "jak ryzykowna erozyjnie jest ta działka
        # niezależnie od wyniku"). Nieblokujący jak wyżej.
        import weather as weather_mod
        import terrain as terrain_mod
        token = core.cdse_token()
        with ThreadPoolExecutor(max_workers=4) as fetch_pool:
            fut_multi = fetch_pool.submit(fetch_multi_series, geom_buf, t_start, t_end, token=token)
            fut_s1 = None
            if req.include_s1:
                from sentinel1 import fetch_s1_series
                fut_s1 = fetch_pool.submit(fetch_s1_series, geom_buf, t_start, t_end)
            fut_weather = fetch_pool.submit(weather_mod.fetch_weather_series, geom_buf, t_start, t_end)
            fut_terrain = fetch_pool.submit(terrain_mod.fetch_terrain_context, geom_buf)

            # 1. Dane wielowskaźnikowe S2 (błąd = błąd całej analizy, jak dawniej)
            multi_records = fut_multi.result()

            # 2. S1 (nieblokujące)
            s1_records = []
            s1_error   = None
            if fut_s1 is not None:
                try:
                    s1_records = fut_s1.result()
                except Exception as e:
                    s1_error = str(e)

            # 2b. Pogoda (nieblokujące)
            weather_error = None
            weather_series: list = []
            frost_dates: set = set()
            dry_dates: set = set()
            first_frost = None
            try:
                weather_series = fut_weather.result()
                frost_dates = weather_mod.detect_frost_dates(weather_series)
                dry_dates = weather_mod.detect_dry_spell_dates(weather_series)
                # "after" = 1 września, żeby nie złapać przypadkowego zimnego
                # poranka wczesnym latem jako "pierwszy przymrozek zimy".
                first_frost = weather_mod.first_frost_date(
                    weather_series, after=date_cls(req.rok, 9, 1))
            except Exception as e:
                weather_error = str(e)

            # 2c. Teren (nieblokujące, czysto informacyjne)
            try:
                terrain_context = fut_terrain.result()
            except Exception:
                terrain_context = None

        def _gdd_since_sowing(obs_date):
            """Stopniodnie wzrostu (GDD) od daty siewu do obs_date — None,
            gdy brak zadeklarowanej daty siewu albo danych pogodowych.
            get_season_phase w takim wypadku sama spada z powrotem na
            klasyfikację tygodniową (patrz jej docstring)."""
            if not sowing_d or not weather_series:
                return None
            return weather_mod.cumulative_gdd(weather_series, sowing_d, obs_date)

        # 3. Oblicz Confidence Score dla ostatniej obserwacji w oknie oceny
        # Filtrujemy po valid_frac tak samo jak core.assess() — scena prawie
        # w całości zachmurzona (valid_frac poniżej MIN_VALID_FRAC) nie może
        # zostać wybrana jako "peak_record" do wyliczeń CS/biomasy, bo to
        # praktycznie szum, nie realny pomiar.
        cover_window = cfg["cover"](req.rok)
        cover_recs   = [r for r in multi_records
                        if cover_window[0] <= r["date"] <= cover_window[1]
                        and (r.get("valid_frac") or 0) >= core.MIN_VALID_FRAC]

        cs_result     = None
        biomass       = None
        peak_record   = None
        agro_events   = []

        if cover_recs:
            # Wybór dnia reprezentatywnego dla okna oceny.
            #
            # Naprawa: dotąd był to po prostu dzień z NAJWYŻSZYM NDVI
            # (max(cover_recs, key=ndvi)), a dopiero POTEM dla tego dnia
            # doliczano EVI/SAVI/NDRE/NDWI/RVI. To sprawiało, że cała analiza
            # wielowskaźnikowa była zakotwiczona w dacie wybranej wyłącznie
            # przez NDVI — pozostałe wskaźniki nie miały żadnego wpływu na to,
            # KTÓRY dzień w ogóle jest oceniany. Problem jest tym istotniejszy,
            # że NDVI saturuje na gęstej biomasie (dokładnie po to istnieje
            # EVI), więc "max NDVI" wśród kilku nasyconych dni to wybór niemal
            # arbitralny.
            #
            # Teraz liczymy pełny Confidence Score dla KAŻDEGO kandydata (z
            # jego własnym dopasowanym RVI i własną fazą fenologiczną) i
            # wybieramy dzień o najwyższym score. To czysta arytmetyka na już
            # pobranych danych — zero dodatkowych zapytań do Sentinel Hub.
            def _rvi_for(rec):
                if not s1_records:
                    return None
                def _dist(s):
                    sd = s["date"] if isinstance(s["date"], date_cls) else date_cls.fromisoformat(str(s["date"]))
                    return abs((sd - rec["date"]).days)
                closest = min(s1_records, key=_dist)
                return closest.get("rvi") if _dist(closest) <= 6 else None

            def _evaluate(rec):
                """Zwraca (cs_result, rvi, faza) dla pojedynczego dnia."""
                rvi = _rvi_for(rec)
                phase = get_season_phase(rec["date"], sowing_d, req.typ, first_frost_date=first_frost,
                                          cumulative_gdd_since_sowing=_gdd_since_sowing(rec["date"]))
                if term_d:
                    near_term = abs((rec["date"] - term_d).days) <= 21
                else:
                    near_term = (cover_window[1] - rec["date"]).days <= 21
                cs = compute_confidence_score(
                    ndvi=rec.get("ndvi"),
                    evi=rec.get("evi"),
                    savi=rec.get("savi"),
                    ndre=rec.get("ndre"),
                    rvi=rvi,
                    ndwi=rec.get("ndwi"),
                    reip=rec.get("reip"),
                    valid_frac=rec.get("valid_frac", 1.0),
                    season_phase=phase,
                    near_termination=near_term,
                )
                return cs, rvi

            evaluated = [(rec, *_evaluate(rec)) for rec in cover_recs]
            # Kandydaci z policzalnym score; jeśli żaden nie ma score (skrajnie
            # ubogie dane), spadamy na dawne kryterium NDVI, żeby nie zwracać
            # pustego wyniku tam, gdzie wcześniej coś było.
            scored = [e for e in evaluated if e[1].get("score") is not None]

            # Naprawa z audytu: wybór "dnia o najwyższym score" mógł wskazać
            # dzień słabej jakości (valid_frac 0.35-0.50) jako reprezentatywny
            # — kara za jakość w samym score jest łagodna (max 7.5% przy
            # dolnej granicy), więc scena z marginalną jakością, ale akurat
            # korzystnymi odczytami, potrafiła wygrać z solidniejszą, lepiej
            # zachmurzoną sceną. Naprawiamy u źródła SELEKCJI, nie samego
            # wyniku: spośród kandydatów o POLICZALNYM score preferujemy
            # WIARYGODNE sceny (valid_frac >= MIN_VALID_FRAC_AUTO) i wybieramy
            # najlepszą z nich; do słabszych sięgamy tylko wtedy, gdy w oknie
            # nie ma ŻADNEJ wiarygodnej sceny. Dzięki temu peak_record może
            # wypaść na scenie słabej jakości wyłącznie wtedy, gdy WSZYSTKIE
            # sceny w oknie są słabej jakości — a to już jest przypadek, który
            # sekcja 4d (kontrola jakości, "większość scen słabej jakości")
            # i tak wychwytuje i kieruje do operatora.
            reliable = [e for e in scored
                        if (e[0].get("valid_frac") or 0) >= core.MIN_VALID_FRAC_AUTO]
            pool = reliable or scored

            if pool:
                peak_record, cs_result, rvi_at_peak = max(
                    pool, key=lambda e: e[1]["score"])
            else:
                peak_record, cs_result, rvi_at_peak = max(
                    evaluated, key=lambda e: e[0].get("ndvi") or 0)

            # Przejrzystość wyboru: obok wyniku z preferencją wiarygodności
            # pokazujemy też, jaki wynik dałby wybór BEZ tego wymogu (czysto
            # po najwyższym score) — żeby było widać, ile "kosztowało"
            # odrzucenie słabszej jakościowo sceny, zamiast to ukrywać.
            if scored:
                alt_rec, alt_cs, _alt_rvi = max(scored, key=lambda e: e[1]["score"])
                if alt_rec is not peak_record and alt_cs["score"] != cs_result["score"]:
                    cs_result = {
                        **cs_result,
                        "best_unfiltered_score": alt_cs["score"],
                        "best_unfiltered_date": alt_rec["date"].isoformat(),
                        "best_unfiltered_valid_frac": alt_rec.get("valid_frac"),
                    }

            biomass = estimate_biomass(
                peak_record.get("evi"), peak_record.get("ndre"), rvi_at_peak
            )
            if rvi_at_peak is not None:
                peak_record = {**peak_record, "rvi": rvi_at_peak}

        # 4. Detekcja zdarzeń
        if multi_records:
            agro_events = detect_agro_events_fused(
                multi_records, s1_records, sowing_d, term_d,
                weather_frost_dates=frost_dates, weather_dry_dates=dry_dates)

        # 4b. Werdykt fuzji: NDVI + wskaźniki wielospektralne + jakość danych
        low_quality_cover = [
            r for r in cover_recs
            if core.MIN_VALID_FRAC <= (r.get("valid_frac") or 0) < core.MIN_VALID_FRAC_AUTO
        ]
        fused_details = []
        requires_operator_approval = False
        confidence_label = None
        fused_status = "NIEROZSTRZYGNIETE"
        # Jawna lista powodów, dla których działka NIE trafiła do zielonej
        # kategorii — po jednym krótkim wpisie na blokadę. Wyświetlana na
        # karcie w wynikach wsadowych, żeby było widać czarno na białym CO
        # zablokowało automatyczne zatwierdzenie (zamiast zgadywać).
        category_reasons = []

        # (Kontrola jakości danych przeniesiona NIŻEJ — musi znać już ustalony
        # fused_status, patrz sekcja 4d.)

        if cs_result and cs_result.get("score") is not None:
            score = cs_result["score"]
            # Progi score->status/label czytane z analytics.SCORE_THRESHOLDS —
            # JEDYNEGO źródła prawdy, dzielonego z grade (A-F) liczonym w
            # compute_confidence_score. Naprawa z audytu: dawniej te same
            # liczby (70/50/30/15) były zakodowane tu NIEZALEŻNIE, więc
            # przyszła rekalibracja progów mogła rozjechać grade i
            # fused_status/confidence_label bez żadnego ostrzeżenia.
            confidence_label = (
                "wysoka" if score >= SCORE_THRESHOLDS["A"] else
                "średnia" if score >= SCORE_THRESHOLDS["B"] else
                "niska"
            )

            if score >= SCORE_THRESHOLDS["A"]:
                fused_status = "OBECNY"
            elif score >= SCORE_THRESHOLDS["C"]:
                fused_status = "PRAWDOPODOBNY"
                category_reasons.append(
                    f"Wynik łączny {score:.0f}% < {SCORE_THRESHOLDS['A']}% "
                    "(próg automatycznego potwierdzenia)")
            elif score >= SCORE_THRESHOLDS["D"]:
                fused_status = "BRAK"
                category_reasons.append(f"Wynik łączny {score:.0f}% — brak okrywy")
            else:
                fused_status = "BRAK"
                category_reasons.append(f"Wynik łączny {score:.0f}% — brak okrywy")

            # Kontrola rozbieżności NDVI vs. wynik wielowskaźnikowy.
            #
            # Naprawa: dawniej stało tu nadpisanie werdyktu przez samo NDVI
            # ("ndvi >= NDVI_COVER_HIGH and score >= 50 -> wymuś OBECNY"),
            # które łamało cały sens oceny wielowskaźnikowej na dwa sposoby:
            #   1. NDVI dostawało DRUGI głos — najpierw jako składnik o
            #      największej wadze w Confidence Score, a potem jeszcze raz
            #      jako prawo weta nad wynikiem tego score'u. Działka ze score
            #      50-69 (czyli "poplon prawdopodobny" wg wszystkich sześciu
            #      wskaźników razem) była awansowana do "OBECNY" wyłącznie
            #      dlatego, że NDVI przekroczyło próg — nawet jeśli EVI, SAVI,
            #      NDRE i RVI wskazywały słabszą okrywę.
            #   2. Nadpisanie było ASYMETRYCZNE — NDVI mogło werdykt tylko
            #      PODNIEŚĆ, nigdy obniżyć. Druga gałąź (ndvi <= NDVI_BARE_MAX
            #      and score < 30 -> BRAK) była martwym kodem: score < 30 i
            #      tak już zawsze dawał BRAK, więc nie zmieniała niczego.
            #
            # Teraz status ustala wyłącznie Confidence Score (wszystkie
            # wskaźniki, z wagami zależnymi od fazy). NDVI nie nadpisuje
            # niczego — zamiast tego SILNA ROZBIEŻNOŚĆ między NDVI a resztą
            # wskaźników jest tym, czym faktycznie jest: sygnałem niepewności,
            # który kieruje sprawę do operatora, a nie powodem, by jeden
            # wskaźnik wygrał z pozostałymi pięcioma.
            ndvi_peak = peak_record.get("ndvi") if peak_record else None
            if ndvi_peak is not None:
                fused_details.append(
                    f"Szczyt NDVI: {ndvi_peak:.3f} (okno oceny: {cover_window[0].isoformat()}–{cover_window[1].isoformat()})."
                )
                ndvi_says_cover = ndvi_peak >= core.NDVI_COVER_HIGH
                ndvi_says_bare  = ndvi_peak <= core.NDVI_BARE_MAX
                # Uwaga: gdy score < SCORE_THRESHOLDS["D"], fused_status jest
                # już wymuszony na "BRAK" (patrz blok wyżej), a categorize_verdict()
                # i tak zawsze zwraca czerwony dla status=="BRAK" — flaga
                # requires_operator_approval/komunikat rozbieżności niczego by
                # tam nie zmieniły, tylko myliłyby operatora sugestią, że to
                # "rozbieżność" zdecydowała o kategorii. Pokazujemy ten
                # komunikat tylko, gdy realnie ma znaczenie dla werdyktu.
                if ndvi_says_cover and score < 50 and fused_status != "BRAK":
                    requires_operator_approval = True
                    category_reasons.append(
                        f"Rozbieżność NDVI ({ndvi_peak:.2f}) vs wynik łączny ({score:.0f}%)")
                    fused_details.append(
                        f"⚠ Rozbieżność: NDVI ({ndvi_peak:.3f}) wskazuje okrywę, ale pozostałe "
                        f"wskaźniki dają niski wynik łączny ({score:.1f}%). Możliwa saturacja NDVI, "
                        "zachwaszczenie lub resztki pożniwne — wymaga weryfikacji operatora."
                    )
                elif ndvi_says_bare and score >= 50:
                    requires_operator_approval = True
                    category_reasons.append(
                        f"Rozbieżność NDVI ({ndvi_peak:.2f}) vs wynik łączny ({score:.0f}%)")
                    fused_details.append(
                        f"⚠ Rozbieżność: NDVI ({ndvi_peak:.3f}) wskazuje glebę bez okrywy, ale pozostałe "
                        f"wskaźniki dają wysoki wynik łączny ({score:.1f}%) — wymaga weryfikacji operatora."
                    )

            comps = cs_result.get("raw_components", {}) or cs_result.get("components", {})
            # REIP jest w nanometrach (~700-728), nie w skali [-1,1] jak reszta
            # surowych wskaźników — ten sam .3f dawałby mylące "REIP=718.400"
            # tuż obok "NDVI=0.550". Formatujemy go osobno, z jednostką, i
            # pomijamy w tej liście (REIP jest już i tak scalony z NDRE w
            # Confidence Score — patrz blend_red_edge_norm — więc pokazywanie
            # go tu jako gdyby był kolejnym, osobnym składnikiem tylko myli).
            comp_txt = ", ".join(
                f"{k.upper()}={v:.3f}" for k, v in comps.items()
                if k != "reip" and isinstance(v, (int, float))
            )
            if comp_txt:
                fused_details.append(f"Składowe oceny: {comp_txt}.")
            reip_raw = comps.get("reip")
            if isinstance(reip_raw, (int, float)):
                fused_details.append(f"REIP (scalony z NDRE powyżej): {reip_raw:.1f}nm.")
            best_unfiltered = cs_result.get("best_unfiltered_score")
            unfiltered_note = (
                f" (najwyższy wynik bez wymogu wiarygodnej sceny: "
                f"{best_unfiltered:.1f}% z {cs_result.get('best_unfiltered_date')}, "
                f"valid_frac={cs_result.get('best_unfiltered_valid_frac')})"
                if best_unfiltered is not None else ""
            )
            fused_details.append(
                f"Confidence Score: {score:.1f}% ({cs_result.get('grade', '?')}) — "
                f"{cs_result.get('label', '').strip()}.{unfiltered_note}"
            )
        else:
            fused_status = "NIEROZSTRZYGNIETE"
            confidence_label = None
            category_reasons.append("Brak wystarczających danych wielowskaźnikowych")
            fused_details.append(
                "Brak wystarczających danych wielowskaźnikowych do potwierdzenia obecności poplonu."
            )

        # 4c. Uwzględnij wykryte zdarzenia agrotechniczne (orka/likwidacja) w
        # werdykcie fuzji. Dotąd `agro_events` trafiały wyłącznie do listy
        # informacyjnej w odpowiedzi i nie wpływały na `fused_status` — co
        # było niespójne z osobną ścieżką `/api/check_fused`
        # (sentinel1.assess_with_sar), która na niezadeklarowaną likwidację
        # reaguje obniżeniem werdyktu. Tu robimy to samo: niezadeklarowane
        # zdarzenie (poza oknem deklarowanej likwidacji) nigdy nie podnosi
        # werdyktu, co najwyżej obniża "OBECNY" do "PRAWDOPODOBNY" i zawsze
        # wymusza przegląd operatora — nie nadpisujemy "BRAK"/"NIEROZSTRZYGNIETE".
        #
        # Naprawa: zdarzenie musi paść W OKNIE OCENY ZGODNOŚCI, nie w
        # rozszerzonym oknie DANYCH/WYŚWIETLANIA. Dla ozimego t_end (dane) to
        # 10.03, ale wymóg programu LEN to utrzymanie okrywy tylko do 15.02
        # (OZIMY_MAINTENANCE_END) — orka po tej dacie jest w pełni zgodna z
        # przepisami, a mimo to każde zdarzenie w oknie 15.02–10.03 było tu
        # traktowane jako "niezadeklarowane" i zawsze zrzucało werdykt do
        # PRAWDOPODOBNY + wymagało przeglądu operatora. To właśnie ta luka
        # sprawiała, że praktycznie żadna działka ozima nie mogła wpaść w
        # kategorię "OK" — każda scena po terminie zaliczała się jako
        # "nieoczekiwana likwidacja", nawet gdy poplon spełnił wymóg w pełni.
        assessment_end = (core.OZIMY_MAINTENANCE_END(req.rok) if req.typ == "ozimy"
                          else cfg["t_end"](req.rok))

        def _event_date(e):
            try:
                return date_cls.fromisoformat(e["date"])
            except (KeyError, ValueError, TypeError):
                return None

        unexpected_events = [
            e for e in agro_events
            if not e.get("within_declared")
            and e.get("confidence") in ("high", "medium")
            and e.get("type") in ("confirmed_tillage", "possible_tillage", "radar_only_event")
            and (_event_date(e) is None or _event_date(e) <= assessment_end)
        ]

        # Dowód "biomasa jednak nadal na polu" PO dacie incydentu.
        #
        # Detekcja zdarzenia (analytics.detect_agro_events_fused) patrzy tylko
        # na PARĘ kolejnych obserwacji (spadek NDVI między dniem A i B, ew.
        # potwierdzenie skokiem VV) — nie sprawdza, co pokazują POZOSTAŁE
        # wskaźniki tego samego dnia, ani co się dzieje PÓŹNIEJ. Jeśli więc
        # istnieje wiarygodna, PÓŹNIEJSZA obserwacja (nadal w oknie oceny),
        # której PEŁNY, wielowskaźnikowy Confidence Score niezależnie wskazuje
        # wyraźną okrywę, to "incydent" był najwyraźniej przejściowy (chwilowy
        # szum sceny, cień chmury, przymrozek) albo okrywa odrosła/nie została
        # trwale usunięta — a blokowanie pozytywnej decyzji mimo takiego
        # dowodu byłoby ignorowaniem właśnie tych wskaźników, po które
        # sięgamy zamiast polegać na samym NDVI.
        POST_EVENT_RECOVERY_MIN_SCORE = 70

        def _rvi_near(d):
            if not s1_records:
                return None
            def _dist(s):
                sd = s["date"] if isinstance(s["date"], date_cls) else date_cls.fromisoformat(str(s["date"]))
                return abs((sd - d).days)
            closest = min(s1_records, key=_dist)
            return closest.get("rvi") if _dist(closest) <= 6 else None

        def _post_event_recovery(event_date):
            candidates = [
                r for r in multi_records
                if r["date"] > event_date and r["date"] <= assessment_end
                and (r.get("valid_frac") or 0) >= core.MIN_VALID_FRAC
            ]
            for rec in candidates:
                # Naprawa z audytu: _evaluate() (wybór peak-day) liczy i
                # przekazuje near_termination, ale ta funkcja tego nie robiła
                # — mimo że jest równie istotne TU: rozróżnienie żywego
                # odrostu od suchych resztek pożniwnych ma największe
                # znaczenie właśnie tuż po zdarzeniu (orka/likwidacja),
                # dokładnie tam, gdzie sprawdzamy dowód odrostu.
                if term_d:
                    near_term = abs((rec["date"] - term_d).days) <= 21
                else:
                    near_term = (cover_window[1] - rec["date"]).days <= 21
                cs = compute_confidence_score(
                    ndvi=rec.get("ndvi"), evi=rec.get("evi"), savi=rec.get("savi"),
                    ndre=rec.get("ndre"), rvi=_rvi_near(rec["date"]), ndwi=rec.get("ndwi"),
                    reip=rec.get("reip"),
                    valid_frac=rec.get("valid_frac", 1.0),
                    season_phase=get_season_phase(rec["date"], sowing_d, req.typ,
                                                  first_frost_date=first_frost,
                                                  cumulative_gdd_since_sowing=_gdd_since_sowing(rec["date"])),
                    near_termination=near_term,
                )
                if cs.get("score") is not None and cs["score"] >= POST_EVENT_RECOVERY_MIN_SCORE:
                    return rec, cs["score"]
            return None, None

        if unexpected_events:
            # Naprawa: dawniej "worst" wybierał pierwsze zdarzenie o wysokiej
            # pewności (a przy braku takiego — po prostu PIERWSZY element
            # listy) i tylko WZGLĘDEM NIEGO sprawdzał odrost. Przy DWÓCH
            # incydentach — wczesnym z odrostem po nim i późniejszym BEZ
            # odrostu (czyli faktycznej, trwałej likwidacji) — kod potrafił
            # znaleźć dowód odrostu leżący MIĘDZY dwoma incydentami i uznać
            # całą sprawę za "przejściową", całkowicie ignorując drugi,
            # nierozwiązany incydent. Zweryfikowane reprodukcją: dwa
            # incydenty (dzień 5 z odrostem dzień 10, dzień 20 bez odrostu)
            # dawały status OBECNY/green/bez zatwierdzenia — mimo trwałej
            # likwidacji w dniu 20.
            #
            # Poprawka: liczy się NAJPÓŹNIEJSZY incydent w oknie oceny — to on
            # reprezentuje aktualny stan pola. Sprawdzamy odrost WZGLĘDEM
            # NIEGO; jeśli jest potwierdzony, to z definicji "pokrywa" też
            # wszystkie wcześniejsze incydenty (odrost jest chronologicznie
            # po nich wszystkich).
            worst = max(unexpected_events, key=lambda e: _event_date(e) or date_cls.min)
            worst_date = _event_date(worst)
            recovery_rec, recovery_score = (
                _post_event_recovery(worst_date) if worst_date else (None, None))

            if recovery_rec is not None:
                fused_details.append(
                    f"ℹ Wykryto incydent ({worst['date']}, pewność: {worst['confidence']}) — {worst['message']} "
                    f"Jednak późniejsza obserwacja ({recovery_rec['date'].isoformat()}, wynik "
                    f"{recovery_score:.0f}%) niezależnie potwierdza wyraźną okrywę — "
                    "incydent uznany za przejściowy, bez blokowania decyzji."
                )
            else:
                requires_operator_approval = True
                category_reasons.append(
                    f"Wykryty incydent {worst['date']} (pewność: {worst['confidence']})")
                if fused_status == "OBECNY":
                    fused_status = "PRAWDOPODOBNY"
                    confidence_label = "średnia"
                    fused_details.append(
                        f"⚠ Wykryto niezadeklarowane zdarzenie agrotechniczne "
                        f"({worst['date']}, pewność: {worst['confidence']}) — {worst['message']} "
                        "Werdykt obniżony z OBECNY do PRAWDOPODOBNY — wymaga weryfikacji operatora."
                    )
                else:
                    fused_details.append(
                        f"⚠ Wykryto niezadeklarowane zdarzenie agrotechniczne "
                        f"({worst['date']}, pewność: {worst['confidence']}) — {worst['message']} "
                        "Wymaga weryfikacji operatora."
                    )

        # 4d. Kontrola jakości danych optycznych.
        #
        # Naprawa (kolejność): ta sekcja MUSI działać PO sekcji 4c (zdarzenia
        # agrotechniczne), nie przed nią — inaczej "is_ambiguous"/"sar_corroborates"
        # liczone tu opierają się na fused_status SPRZED ewentualnego obniżenia
        # przez wykryty incydent, co potrafiło dać samosprzeczny komunikat:
        # 4d pisze "wynik jednoznaczny, bez konieczności zatwierdzenia", a
        # chwilę później 4c i tak ustawia wymóg zatwierdzenia z powodu
        # incydentu. requires_operator_approval był finalnie poprawny (4c
        # ustawia go niezależnie), ale wyświetlany operatorowi tekst wprowadzał
        # w błąd co do PRZYCZYNY.
        #
        # Naprawa (reguła): dotąd stało tu twarde "peak_record.valid_frac <
        # MIN_VALID_FRAC_AUTO -> requires_operator_approval = True", czyli
        # jedna częściowo zachmurzona scena bezwarunkowo blokowała
        # automatyczne zatwierdzenie — nawet przy Confidence Score 94 (ocena A,
        # "poplon potwierdzony"). Były w tym dwa realne błędy:
        #
        #   1. PODWÓJNE KARANIE za to samo. compute_confidence_score() już
        #      obniża wynik proporcjonalnie do valid_frac (quality_penalty).
        #      Score 94 to wynik JUŻ uwzględniający zachmurzenie — a mimo to
        #      ta sama informacja była tu użyta drugi raz, tym razem jako
        #      bezwarunkowe weto nad wynikiem, który sam ją wliczył.
        #
        #   2. NIESPÓJNOŚĆ Z ISTNIEJĄCĄ, BARDZIEJ PRZEMYŚLANĄ REGUŁĄ. Ścieżka
        #      NDVI (core.assess) rozstrzyga to od dawna mądrzej: wymaga
        #      zatwierdzenia tylko gdy wynik jest NIEJEDNOZNACZNY albo gdy
        #      WIĘKSZOŚĆ scen w oknie jest słabej jakości; przy jednoznacznym
        #      werdykcie i przewadze dobrych scen wprost stwierdza "dominują
        #      sceny wiarygodne — bez konieczności zatwierdzenia". Ścieżka
        #      fuzji nadpisuje werdykt NDVI, więc porzucając tę logikę
        #      cofała zachowanie do surowszego niż to, co zastępowała.
        #
        # Stosujemy więc tę samą regułę co core.assess — jedna, spójna polityka
        # jakości danych w całej aplikacji, zamiast dwóch rozjeżdżających się.
        if low_quality_cover:
            n_low = len(low_quality_cover)
            n_all = len(cover_recs)
            is_ambiguous = fused_status in ("PRAWDOPODOBNY", "NIEROZSTRZYGNIETE")
            majority_low = (n_low / n_all) > 0.5 if n_all else False

            # POTWIERDZENIE RADAROWE znosi blokadę "większość scen optycznych
            # słabej jakości". Uzasadnienie: cały sens fuzji S1 polega na tym,
            # że radar NIE jest blokowany przez chmury — jeśli więc problemem
            # optyki było wyłącznie zachmurzenie, a niezależny od chmur RVI
            # sam z siebie potwierdza obecność wyraźnej okrywy, to wysyłanie
            # sprawy do ręcznego zatwierdzenia "z powodu chmur" ignorowałoby
            # dokładnie ten pomiar, który chmur nie widzi.
            #
            # Warunki zniesienia blokady (wszystkie naraz):
            #   1. wynik JEDNOZNACZNY (OBECNY, score >= 70) — niejednoznaczny
            #      zawsze idzie do operatora, niezależnie od radaru;
            #   2. RVI faktycznie zmierzone dla dnia oceny (nie brak danych);
            #   3. RVI po normalizacji >= 0.5 — radar niezależnie wskazuje
            #      wyraźną strukturę roślinną, a nie tylko "jakiś sygnał".
            # Bez spełnienia tych warunków zasada działa jak dotąd.
            rvi_norm = (cs_result.get("components", {}) or {}).get("rvi") if cs_result else None
            sar_corroborates = (
                fused_status == "OBECNY"
                and rvi_norm is not None
                and rvi_norm >= 0.5
            )

            if is_ambiguous or (majority_low and not sar_corroborates):
                requires_operator_approval = True
                if majority_low:
                    category_reasons.append(
                        f"Większość scen słabej jakości ({n_low}/{n_all})"
                        + ("" if rvi_norm is not None else ", brak pomiaru RVI do potwierdzenia"))
                fused_details.append(
                    f"⚠ Dane częściowe: {n_low}/{n_all} scen obniżonej jakości "
                    f"(valid_frac {core.MIN_VALID_FRAC:.2f}–{core.MIN_VALID_FRAC_AUTO:.2f}). "
                    + ("Wynik niejednoznaczny — " if is_ambiguous else "")
                    + "Werdykt wymaga zatwierdzenia operatora."
                )
            elif majority_low and sar_corroborates:
                fused_details.append(
                    f"ℹ {n_low}/{n_all} scen optycznych obniżonej jakości (zachmurzenie), "
                    f"ale radar Sentinel-1 (RVI, niewrażliwy na chmury) niezależnie "
                    "potwierdza wyraźną okrywę — bez konieczności zatwierdzenia."
                )
            else:
                fused_details.append(
                    f"ℹ {n_low}/{n_all} scen obniżonej jakości "
                    f"(valid_frac {core.MIN_VALID_FRAC:.2f}–{core.MIN_VALID_FRAC_AUTO:.2f}), "
                    "ale wynik jest jednoznaczny i dominują sceny wiarygodne — "
                    "bez konieczności zatwierdzenia."
                )

        # Notatka informacyjna o ryzyku erozyjnym z kontekstu rzeźby terenu —
        # CELOWO nie wpływa na status/requires_operator_approval (patrz
        # terrain.py: inny typ dowodu, nie "czy jest okrywa").
        if terrain_context and terrain_context.get("erosion_risk") == "wysokie":
            fused_details.append(
                f"ℹ Kontekst terenu: nachylenie {terrain_context['slope_deg']:.1f}° "
                "— wysokie ryzyko erozyjne działki (niezależnie od wyniku oceny okrywy)."
            )

        fused_verdict = {
            "status": fused_status,
            "confidence": confidence_label,
            "details": fused_details,
            "detail_codes": [],
            "compliance_issue": False,
            "requires_operator_approval": requires_operator_approval,
            "low_quality_scenes": len(low_quality_cover),
        }
        fused_category = core.categorize_verdict(fused_verdict)
        if fused_category == "green":
            category_reasons = []

        # 5. Serializacja dat
        def _ser(records):
            return [{**r, "date": r["date"].isoformat()
                     if hasattr(r["date"], "isoformat") else str(r["date"])}
                    for r in records]

        return {
            "parcel_id":      req.parcel_id,
            "parcel_info":    parcel_info,
            "confidence":     cs_result,
            "biomass":        biomass,
            "peak_record":    {**peak_record, "date": peak_record["date"].isoformat()}
                              if peak_record else None,
            "multi_records":  _ser(multi_records),
            "s1_records":     _ser(s1_records),
            "agro_events":    agro_events,
            "fused_verdict":  fused_verdict,
            "fused_category": fused_category,
            "category":       fused_category,
            "category_reasons": category_reasons,
            "s1_error":       s1_error,
            "weather_context": {
                "frost_days": len(frost_dates),
                "dry_spell_days": len(dry_dates),
                "first_frost_date": first_frost.isoformat() if first_frost else None,
            },
            "weather_error":  weather_error,
            "terrain_context": terrain_context,
            # Czysto informacyjne podsumowanie warunków wzrostu (GDD, opad,
            # nasłonecznienie) w oknie oceny — NIGDY nie wpływa na score ani
            # status, patrz weather.growing_conditions_summary docstring.
            "growing_conditions": weather_mod.growing_conditions_summary(
                weather_series, cover_window[0], cover_window[1]),
            "summary": {
                "observations_s2":    len(multi_records),
                "observations_s1":    len(s1_records),
                "agro_events_high":   sum(1 for e in agro_events if e["confidence"] == "high"),
                "confidence_score":   cs_result["score"] if cs_result else None,
                "confidence_grade":   cs_result["grade"] if cs_result else None,
                "biomass_t_ha":       biomass["biomass_t_ha"] if biomass else None,
            }
        }

    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error(e)


# ============================================================
# Koherencja InSAR — dodatkowa, RĘCZNIE ZLECANA analiza (patrz insar.py)
# ============================================================
#
# W odróżnieniu od check_multi (synchroniczny, jeden request -> pełny
# wynik), to jest DWUETAPOWE i ASYNCHRONICZNE: /request zleca zadanie w
# CDSE On-Demand Production API i zwraca order_id NATYCHMIAST (bez
# czekania na wynik — przetwarzanie InSAR nie ma gwarantowanego czasu
# ukończenia, patrz insar.py), a /status odpytuje o postęp i — po
# ukończeniu — pobiera i przetwarza wynik.
#
# _insar_orders: cache geometrii działki per order_id, żeby /status nie
# musiało jej dostawać w query params przy każdym odpytaniu. CELOWO
# tylko w pamięci procesu (nie w bazie) — to sesyjny, krótkotrwały stan
# jednej ręcznej weryfikacji operatora, nie trwałe dane; ginie przy
# restarcie serwera, co dla tego przypadku użycia jest akceptowalne
# (operator po prostu zleca ponownie).
_insar_orders: dict = {}
_insar_orders_lock = threading.Lock()


class InsarCoherenceRequest(BaseModel):
    event_date: str   # data zdarzenia (ISO), wokół której szukamy sceny SLC


@app.post("/api/parcel/{parcel_id}/insar_coherence")
def insar_coherence_request(parcel_id: str, req: InsarCoherenceRequest,
                            _auth=Depends(require_operator_key)):
    """Zleca sprawdzenie koherencji InSAR dla działki wokół podanej daty
    zdarzenia. NIE czeka na wynik — zwraca order_id do odpytania przez
    GET /api/insar_coherence/{order_id}."""
    try:
        import insar
        from datetime import date as date_cls

        event_d = date_cls.fromisoformat(req.event_date)
        geom, _parcel_info = core.fetch_parcel_geometry(parcel_id)

        product = insar.find_slc_product(geom, event_d)
        if product is None:
            raise HTTPException(
                status_code=404,
                detail=f"Nie znaleziono sceny Sentinel-1 SLC pokrywającej działkę "
                       f"w oknie ±{insar.SLC_SEARCH_TOLERANCE_DAYS} dni od {req.event_date}.")

        order_id = insar.submit_coherence_order(product["name"])
        with _insar_orders_lock:
            _insar_orders[order_id] = {
                "geom_wkt": geom.wkt, "parcel_id": parcel_id,
                "created": time.time(), "product_name": product["name"],
            }

        return {
            "order_id": order_id,
            "status": "queued",
            "product_used": product["name"],
            "note": ("Zadanie zlecone w CDSE On-Demand Production API. Przetwarzanie "
                    "InSAR nie ma gwarantowanego czasu ukończenia (darmowa, wspólna "
                    "pula zasobów) — sprawdź status za chwilę przez GET "
                    f"/api/insar_coherence/{order_id}."),
        }
    except HTTPException:
        raise
    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error(e)


@app.get("/api/insar_coherence/{order_id}")
def insar_coherence_status(order_id: str, _auth=Depends(require_operator_key)):
    """Sprawdza status zleconej analizy koherencji. Gdy ukończona, pobiera
    wynik, wycina średnią koherencję po geometrii działki i zwraca
    interpretację (patrz insar.interpret_coherence — czysto informacyjna,
    NIE wpływa na status/score check_multi)."""
    try:
        import insar
        with _insar_orders_lock:
            order_meta = _insar_orders.get(order_id)
        if order_meta is None:
            raise HTTPException(
                status_code=404,
                detail="Nieznany order_id (albo serwer został zrestartowany od "
                       "czasu zlecenia — zleć sprawdzenie ponownie).")

        status_info = insar.get_order_status(order_id)
        status = str(status_info["status"]).lower()

        if status not in ("completed", "done", "finished"):
            return {"order_id": order_id, "status": status,
                    "estimated_date": status_info.get("estimated_date")}

        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(order_meta["geom_wkt"])
        zip_bytes = insar.download_order_result(order_id)
        mean_coherence = insar.extract_mean_coherence(zip_bytes, geom)
        interpretation = insar.interpret_coherence(mean_coherence)

        return {"order_id": order_id, "status": "completed", **interpretation}
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(e)


@app.get("/api/parcel/{parcel_id}/report_pdf")
def parcel_report_pdf(parcel_id: str, typ: str, rok: int,
                       sowing_date: Optional[str] = None,
                       termination_date: Optional[str] = None):
    """Raport PDF dla pojedynczej działki — buduje się z tego samego wyniku
    check_multi, który widzi operator w UI (żadnych nowych zapytań do
    Copernicus). Bez osobnej autoryzacji, spójnie z /api/check_multi
    (te same dane, ten sam brak bramki — parcel_id i tak jest publiczny
    przez ULDK)."""
    try:
        import report_pdf
        req = MultiCheckRequest(parcel_id=parcel_id, typ=typ, rok=rok,
                                 sowing_date=sowing_date, termination_date=termination_date,
                                 include_s1=False)
        data = check_multi(req)
        pdf_bytes = report_pdf.build_parcel_report(data)
        from fastapi.responses import Response
        return Response(content=pdf_bytes, media_type="application/pdf", headers={
            "Content-Disposition": f'inline; filename="raport_{parcel_id}_{rok}.pdf"'})
    except HTTPException:
        raise
    except core.AppError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error(e)


@app.get("/api/farmer_report_pdf")
def farmer_report_pdf(farmer_phone: str, _auth=Depends(require_operator_key)):
    """Zbiorczy raport PDF dla WSZYSTKICH monitorowanych działek jednego
    rolnika (grupowanie po farmer_phone — patrz db.get_monitored_parcels_by_farmer_phone).
    W odróżnieniu od raportu pojedynczej działki, ten endpoint agreguje dane
    osobowe (numer telefonu) wielu działek naraz -> wymaga klucza operatora."""
    try:
        parcels = db.get_monitored_parcels_by_farmer_phone(farmer_phone)
        if not parcels:
            raise HTTPException(status_code=404,
                                 detail=f"Brak monitorowanych działek dla numeru {farmer_phone}.")

        import report_pdf
        parcels_data = []
        for p in parcels:
            req = MultiCheckRequest(
                parcel_id=p["parcel_id"], typ=p["typ"], rok=p["rok"],
                sowing_date=p.get("sowing_date"), termination_date=p.get("termination_date"),
                include_s1=False)
            try:
                parcels_data.append(check_multi(req))
            except Exception as e:
                # Nie blokuj całego zbiorczego raportu przez błąd jednej działki
                # (np. chwilowy błąd ULDK/Sentinel Hub) — zaznacz to w sekcji.
                parcels_data.append({
                    "parcel_id": p["parcel_id"],
                    "parcel_info": {},
                    "confidence": {},
                    "fused_verdict": {"status": f"BŁĄD ANALIZY: {e}"},
                    "fused_category": None,
                    "category_reasons": [],
                    "multi_records": [], "agro_events": [],
                })

        farmer_name = parcels[0].get("farmer_name")
        pdf_bytes = report_pdf.build_farmer_report(parcels_data, farmer_name=farmer_name,
                                                     farmer_phone=farmer_phone)
        from fastapi.responses import Response
        return Response(content=pdf_bytes, media_type="application/pdf", headers={
            "Content-Disposition": f'inline; filename="raport_zbiorczy_{farmer_phone}.pdf"'})
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(e)


class SmsSendRequest(BaseModel):
    message: str
    parcel_id: Optional[str] = None
    phone: Optional[str] = None


@app.post("/api/sms/send")
def sms_send(req: SmsSendRequest, _auth=Depends(require_operator_key)):
    """Wysyła SMS przez SMSAPI.pl. Wymaga zmiennej SMSAPI_TOKEN."""
    import os, requests as req_lib
    token = os.environ.get("SMSAPI_TOKEN")
    if not token:
        raise HTTPException(status_code=503,
            detail="SMSAPI_TOKEN nie jest ustawiony na serwerze. "
                   "Dodaj go w zakładce Secrets w Replicie.")
    phone = req.phone
    if not phone:
        raise HTTPException(status_code=422,
            detail="Brak numeru telefonu — podaj numer w polu SMS lub "
                   "dodaj telefon rolnika w danych działki.")
    if len(req.message) > 480:
        raise HTTPException(status_code=422,
            detail=f"Wiadomość za długa ({len(req.message)} znaków, max 480).")
    try:
        r = req_lib.post(
            "https://ssl.smsapi.pl/sms.do",
            headers={"Authorization": f"Bearer {token}"},
            data={"to": phone, "message": req.message,
                  "format": "json", "from": "PopMonitor"},
            timeout=15,
        )
        data = r.json()
        if r.status_code == 200 and data.get("count", 0) > 0:
            return {"status": "ok", "count": data.get("count")}
        else:
            raise HTTPException(status_code=502,
                detail=f"SMSAPI error: {data.get('message', str(data))}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd wysyłania: {e}")


@app.get("/")
def index():
    # no-cache (nie no-store): przeglądarka i tak trzyma kopię, ale musi ją
    # zweryfikować (If-None-Match/ETag) przy każdym wejściu, zamiast serwować
    # starą wersję z pamięci podręcznej po każdej zmianie static/index.html.
    return FileResponse("static/index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory="static"), name="static")
