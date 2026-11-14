"""Uzupełnianie historii i powtórka sezonu dla modułu wodnego.

Dwa zastosowania:

1. UZUPEŁNIANIE HISTORII (backfill_field) — nowe pole potrzebuje kilku
   tygodni obserwacji, zanim anomalia względem własnej historii w ogóle
   zadziała (OWN_HISTORY_MIN_SAMPLES). Backfill pobiera JEDNYM zapytaniem
   na źródło cały okres wstecz (S2, S1, orbity, pogoda) i zapisuje każdą
   czystą scenę jako obserwację — tą samą ścieżką co bieżące sprawdzenie
   (water_stress.fetch_sources / build_observation).

2. POWTÓRKA SEZONU (replay_field) — "co by się stało, gdyby moduł działał
   cały sezon": dla każdej obserwacji po kolei liczy werdykt tylko z danych
   dostępnych DO tego dnia (bez zaglądania w przyszłość), symuluje kolejkę
   alarmów z tym samym okresem karencji co scheduler i — opcjonalnie —
   bilans wodny FAO-56 na pogodzie z archiwum ERA5. NIC nie zapisuje do
   water_alarms — to suchy przebieg do oceny progów przed sezonem.

Uruchomienie z wiersza poleceń (osobna baza, produkcyjna nietknięta):

    python water_replay.py pola.geojson --od 2026-04-01 --do 2026-09-30

pola.geojson: poligony w WGS84 albo z polem "crs" (np. EPSG:2180);
właściwości (opcjonalne): parcel_id / id / name, irrigation_mode
(irrigation|rainfed, domyślnie rainfed), region, crop_type, soil_taw_mm,
sowing_date. Zamiast GeoJSON można podać CSV z kolumną "wkt" (WGS84) i tymi
samymi kolumnami; dla polskich działek kolumna wkt może być pusta — obrys
zostanie pobrany z ULDK po numerze działki (parcel_id). Wynik: katalog replay_wyniki/ z obserwacje.csv,
alarmy.csv i podsumowanie.txt.

Koszt: na pole jedno zapytanie Statistical API S2 i jedno S1 na cały okres
(PU zależy od liczby scen w okresie — sprawdź na jednym polu przed dużą
partią), plus bezpłatne Open-Meteo i STAC.
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import date, timedelta
from typing import Optional

import db
import water_stress as ws

log = logging.getLogger("water_replay")

REPLAY_ALARM_COOLDOWN_DAYS = ws.WATER_ALARM_RECENT_DAYS


# ---------------------------------------------------------------------------
# 1. Uzupełnianie historii
# ---------------------------------------------------------------------------

def backfill_field(field: dict, t_start: date, t_end: date) -> dict:
    """Zapisuje do water_observations wszystkie czyste sceny z okresu.
    Obserwacje już istniejące (ta sama data) zostają bez zmian (INSERT OR
    IGNORE). Zwraca {"scenes": n_wszystkich, "clean": n_czystych, "stored": n}."""
    sources = ws.fetch_sources(field, t_start, t_end)
    optical = sources["optical"]
    clean = [r for r in optical
             if (r.get("valid_frac") or 0.0) >= ws.WATER_MIN_VALID_FRAC and r.get("ndwi") is not None]
    existing = {o["obs_date"] for o in db.get_water_observations(field["id"])}
    stored = updated = 0
    for opt in sorted(clean, key=lambda r: r["date"]):
        obs = ws.build_observation(field, opt, sources)
        if obs["obs_date"] in existing:
            # obserwacja zapisana, zanim moduł liczył EVI2/red-edge — dopisz je
            if db.fill_water_observation(field["id"], obs["obs_date"],
                                         **{k: obs.get(k) for k in db.WATER_OBS_FILLABLE}):
                updated += 1
            continue
        if db.add_water_observation(**obs):
            stored += 1
    if stored:
        db.update_water_field_last_checked(field["id"])
    return {"scenes": len(optical), "clean": len(clean), "stored": stored, "updated": updated,
            "radar_scenes": len(sources["radar"]), "weather_days": len(sources["weather"])}


# ---------------------------------------------------------------------------
# 2. Powtórka sezonu
# ---------------------------------------------------------------------------

def _balance_by_date(field: dict, observations: list, events: list, weather_series: list) -> dict:
    """Bilans FAO-56 na pogodzie historycznej. Zwraca {data: rekord dnia}.
    Uwaga: NDVI między obserwacjami jest interpolowane, więc w dniach między
    scenami bilans "widzi" najbliższą następną scenę (kilka dni) — pomijalne
    dla oceny progów, ale to nie jest ściśle bez zaglądania w przyszłość."""
    import water_balance as wb
    if not weather_series:
        return {}
    weather_series = [dict(w) for w in weather_series]
    wb.apply_rain_gauge(weather_series, db.get_rain_readings(field["id"]))
    kcb_max, p, _ = wb.crop_params(field.get("crop_type"))
    import soil_water
    soil = soil_water.field_taw(field)
    taw = float(soil["taw_mm"])
    ks_salt = soil_water.field_salinity(field, texture=soil.get("texture"))["ks_salt"]
    ndvi_points = [(date.fromisoformat(o["obs_date"]), o["ndvi_mean"])
                   for o in observations if o.get("ndvi_mean") is not None]
    start = None
    if field.get("sowing_date"):
        try:
            start = date.fromisoformat(field["sowing_date"])
        except ValueError:
            start = None
    days = wb.run_balance(weather_series, ndvi_points, events, taw, p, kcb_max, start_date=start,
                          soil_checks=db.get_soil_checks(field["id"]), ks_salt=ks_salt)
    out = {r["date"]: r for r in days}
    raw = p * taw
    for r in days:
        r["raw_mm"] = round(raw, 1)
        r["taw_mm"] = taw
    return out


def _balance_summary_at(day: date, by_date: dict, horizon: int = 10) -> Optional[dict]:
    """Podsumowanie bilansu 'na dzień' z doskonałą prognozą (rzeczywista
    pogoda kolejnych dni) — odpowiednik water_balance.summarize_balance."""
    cur = by_date.get(day)
    if cur is None:
        return None
    raw = cur["raw_mm"]
    status = "stress" if cur["depletion_mm"] > raw else (
        "approaching" if cur["depletion_mm"] > 0.75 * raw else "ok")
    stress_day = day if status == "stress" else next(
        (day + timedelta(days=k) for k in range(1, horizon + 1)
         if by_date.get(day + timedelta(days=k), {}).get("stress")), None)
    dose = by_date.get(stress_day, cur)["depletion_mm"] if stress_day else None
    return {"status": status, "depletion_mm": cur["depletion_mm"], "raw_mm": raw,
            "taw_mm": cur["taw_mm"],
            "days_to_stress": None if stress_day is None else (stress_day - day).days,
            "suggested_irrigation_date": stress_day.isoformat() if stress_day else None,
            "suggested_dose_mm": None if dose is None else round(dose, 0),
            "suggested_dose_m3_ha": None if dose is None else round(dose * 10, -1)}


def _date_periods(days: list) -> list:
    """Sąsiednie daty skleja w przedziały {from, to} (włącznie)."""
    if not days:
        return []
    days = sorted(days)
    out, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if (d - prev).days == 1:
            prev = d
            continue
        out.append({"from": start.isoformat(), "to": prev.isoformat()})
        start = prev = d
    out.append({"from": start.isoformat(), "to": prev.isoformat()})
    return out


def replay_field(field: dict, weather_series: Optional[list] = None) -> dict:
    """Suchy przebieg silnika na zapisanej historii pola. Zwraca
    {"rows": [...], "alarms": [...], "summary": {...}}; niczego nie zapisuje."""
    observations = db.get_water_observations(field["id"])
    events = db.get_irrigation_events(field["id"])
    cohort = ws.get_cohort_fields(field)
    bal_by_date = _balance_by_date(field, observations, events, weather_series or [])

    rows, alarms = [], []
    last_alarm: dict = {}
    for i, obs in enumerate(observations):
        d = date.fromisoformat(obs["obs_date"])
        hist = observations[:i + 1]
        ev = [e for e in events if e["event_date"] <= obs["obs_date"]]
        cohort_vals = ws.cohort_values_near_date(cohort, obs["obs_date"])
        verdict = ws.classify_water_signal(field, hist, cohort_vals, ev)
        bal = _balance_summary_at(d, bal_by_date)
        if bal is not None:
            verdict["water_balance"] = bal

        emitted = None
        candidates = []
        if verdict.get("risk_flag"):
            candidates.append((verdict["risk_type"], verdict["message_pl"]))
        else:
            dec = ws.balance_alarm_decision(field, verdict)
            if dec:
                candidates.append((dec["risk_type"], dec["message_pl"]))
        for rtype, msg in candidates:
            prev = last_alarm.get(rtype)
            if prev is None or (d - prev).days >= REPLAY_ALARM_COOLDOWN_DAYS:
                last_alarm[rtype] = d
                emitted = rtype
                alarms.append({"parcel_id": field.get("parcel_id"), "obs_date": obs["obs_date"],
                               "risk_type": rtype, "message_pl": msg})

        comp = verdict.get("components", {})
        z = lambda k: (comp.get(k) or {}).get("z_score")
        rows.append({
            "parcel_id": field.get("parcel_id"), "obs_date": obs["obs_date"],
            "ndmi": obs.get("ndwi_mean"), "ndvi": obs.get("ndvi_mean"),
            "ndmi_p10": obs.get("ndwi_p10"), "ndmi_low_frac": obs.get("ndwi_low_frac"),
            "vv_db": obs.get("vv_db"), "s1_orbit": obs.get("s1_rel_orbit"),
            "cwb_14d_mm": obs.get("cwb_14d_mm"),
            "z_ndmi": z("own_ndwi"), "z_ndmi_residual": z("own_ndwi_residual"), "z_vv": z("own_vv"),
            "delta_ndmi": (comp.get("own_ndwi") or {}).get("delta"),
            "risk_flag": verdict.get("risk_flag"), "risk_type": verdict.get("risk_type"),
            "phenology_explained": verdict.get("phenology_explained"),
            "harvest_like": verdict.get("harvest_like"),
            "canopy_present": verdict.get("canopy_present"),
            "balance_depletion_mm": bal["depletion_mm"] if bal else None,
            "balance_raw_mm": bal["raw_mm"] if bal else None,
            "balance_status": bal["status"] if bal else None,
            "alarm_emitted": emitted,
        })

    by_type: dict = {}
    for a in alarms:
        by_type[a["risk_type"]] = by_type.get(a["risk_type"], 0) + 1
    summary = {
        "parcel_id": field.get("parcel_id"),
        "observations": len(observations),
        "first_obs": observations[0]["obs_date"] if observations else None,
        "last_obs": observations[-1]["obs_date"] if observations else None,
        "alarms_total": len(alarms), "alarms_by_type": by_type,
        "phenology_suppressed": sum(1 for r in rows if r["phenology_explained"]),
        "stress_days_in_balance": sum(1 for r in bal_by_date.values() if r.get("stress")),
    }
    stress_periods = _date_periods(
        [d for d, r in bal_by_date.items() if r.get("stress") and not r.get("is_forecast")])
    return {"rows": rows, "alarms": alarms, "summary": summary,
            "stress_periods": stress_periods}


# ---------------------------------------------------------------------------
# 3. Oceny powtórki i kalibracja progu z obu rodzajów błędów
# ---------------------------------------------------------------------------

RATING_MIN_POSITIVES = 3
RATING_MIN_NEGATIVES = 3
THRESHOLD_GRID = [round(-4.0 + 0.1 * i, 1) for i in range(36)]   # -4.0 .. -0.5


def rate_replay(field: dict, obs_date: str, rating: str, risk_type: str = "",
                note: Optional[str] = None) -> dict:
    """Zapisuje ocenę doradcy dla obserwacji z powtórki razem z metrykami
    silnika z tego dnia (z-score NDMI, rezyduum, NDVI, bramki łan/zbiór) —
    kalibracja liczy na tych liczbach, a nie na samym fakcie alarmu.
    Dla 'missed' bierze obserwację najbliższą wskazanej dacie (±3 dni)."""
    rows = replay_field(field)["rows"]
    if not rows:
        raise ValueError("Pole nie ma jeszcze historii obserwacji — najpierw uzupełnij historię.")
    target = date.fromisoformat(obs_date)
    row = min(rows, key=lambda r: abs((date.fromisoformat(r["obs_date"]) - target).days))
    if abs((date.fromisoformat(row["obs_date"]) - target).days) > 3:
        raise ValueError("Brak obserwacji satelitarnej w ciągu 3 dni od wskazanej daty.")
    if rating == "missed":
        if row.get("risk_flag"):
            raise ValueError(f"W dniu {row['obs_date']} silnik już sygnalizował problem — "
                             f"oceń ten alarm jako trafny zamiast dopisywać przeoczenie.")
        risk_type = ""   # przeoczenie to osobny wpis, nie nadpisuje oceny alarmu
    metrics = {k: row.get(k) for k in ("ndmi", "ndvi", "delta_ndmi", "z_ndmi", "z_ndmi_residual",
                                       "z_vv", "cwb_14d_mm", "risk_flag", "risk_type",
                                       "phenology_explained", "harvest_like", "canopy_present")}
    if rating != "missed":
        risk_type = risk_type or (row.get("risk_type") or "")
    return db.upsert_replay_rating(field["id"], row["obs_date"], rating,
                                   risk_type=risk_type,
                                   metrics_json=json.dumps(metrics), note=note)


def _confusion(pos_z: list, neg_z: list, thr: float) -> dict:
    tp = sum(1 for z in pos_z if z <= thr)
    fn = len(pos_z) - tp
    fp = sum(1 for z in neg_z if z <= thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def suggest_threshold_from_ratings(ratings: Optional[list] = None) -> dict:
    """Sugestia Z_SCORE_RISK_THRESHOLD z ocen powtórki. Pozytywy = alarmy
    trafne + przeoczone problemy, negatywy = alarmy fałszywe. Wybór progu
    o najlepszym F1 (przy remisie — bliższy obecnemu). Dotyczy tylko alarmów
    spadku wilgotności (nie równomierności ani bilansu)."""
    if ratings is None:
        ratings = db.get_replay_ratings()
    pos, neg, blocked = [], [], []
    for r in ratings:
        if r.get("risk_type") in ws.NON_DECLINE_RISK_TYPES:
            continue
        try:
            m = json.loads(r.get("metrics_json") or "{}")
        except ValueError:
            continue
        z = m.get("z_ndmi")
        if z is None:
            continue
        if r["rating"] in ("confirmed", "missed"):
            pos.append(z)
            # Przeoczony problem, choć z-score przekroczył próg: zablokowała go
            # inna reguła (brak łanu, zbiór, rezyduum, min. spadek) — próg z
            # tego nie naprawi, trzeba przejrzeć regułę.
            if r["rating"] == "missed" and z <= ws.Z_SCORE_RISK_THRESHOLD:
                blocked.append({"water_field_id": r["water_field_id"], "obs_date": r["obs_date"],
                                "z_ndmi": z, "canopy_present": m.get("canopy_present"),
                                "harvest_like": m.get("harvest_like"),
                                "phenology_explained": m.get("phenology_explained")})
        elif r["rating"] == "rejected":
            neg.append(z)
    current = ws.Z_SCORE_RISK_THRESHOLD
    out = {"positives": len(pos), "negatives": len(neg), "current_threshold": current,
           "blocked_by_other_rules": blocked,
           "min_positives": RATING_MIN_POSITIVES, "min_negatives": RATING_MIN_NEGATIVES}
    if len(pos) < RATING_MIN_POSITIVES or len(neg) < RATING_MIN_NEGATIVES:
        out["status"] = "not_enough_data"
        return out
    grid = [_confusion(pos, neg, t) for t in THRESHOLD_GRID]
    best = max(grid, key=lambda g: (g["f1"], -abs(g["threshold"] - current)))
    out.update({"status": "ok", "suggested_threshold": best["threshold"],
                "at_suggested": best, "at_current": _confusion(pos, neg, current)})
    return out


# ---------------------------------------------------------------------------
# 4. Wczytywanie pól (GeoJSON / CSV) i CLI
# ---------------------------------------------------------------------------

from water_geometry import FIELD_PROPS  # noqa: E402


def load_fields(path: str) -> list:
    """Pola z GeoJSON/CSV (water_geometry.parse_fields_text). Pola bez
    geometrii z polskim numerem działki dostają obrys z ULDK; tryb domyślnie
    'rainfed'."""
    import water_geometry as wg
    out = []
    for f in wg.load_fields(path):
        if not f.get("geometry_wkt"):
            if not wg.is_polish_parcel_ids(f["parcel_id"]):
                raise ValueError(f"Pole {f['parcel_id']}: brak geometrii i to nie jest numer działki z ULDK")
            f["geometry_wkt"] = wg.geometry_from_parcel_ids(f["parcel_id"])["geometry_wkt"]
        f["irrigation_mode"] = f.get("irrigation_mode") or "rainfed"
        out.append(f)
    return out


def _write_csv(path: str, rows: list):
    if not rows:
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def run_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Powtórka sezonu modułu wodnego (suchy przebieg).")
    ap.add_argument("pola", help="GeoJSON albo CSV z kolumną wkt")
    ap.add_argument("--od", required=True, help="RRRR-MM-DD")
    ap.add_argument("--do", required=True, help="RRRR-MM-DD")
    ap.add_argument("--baza", default=None, help="plik SQLite powtórki (domyślnie replay_<od>_<do>.db)")
    ap.add_argument("--wyniki", default="replay_wyniki", help="katalog na wyniki")
    ap.add_argument("--bez-bilansu", action="store_true", help="pomiń bilans wodny FAO-56")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t_start, t_end = date.fromisoformat(args.od), date.fromisoformat(args.do)
    db.DB_PATH = args.baza or f"replay_{args.od}_{args.do}.db"
    if os.path.abspath(db.DB_PATH) == os.path.abspath("monitoring.db"):
        print("Odmowa: powtórka nie może pracować na produkcyjnej monitoring.db.", file=sys.stderr)
        return 2
    db.init_db()
    os.makedirs(args.wyniki, exist_ok=True)

    fields_in = load_fields(args.pola)
    print(f"Pól do powtórki: {len(fields_in)} · okres {t_start}..{t_end} · baza {db.DB_PATH}")
    known = {f["parcel_id"]: f for f in db.get_water_fields()}
    fields = []
    for fin in fields_in:
        if fin["parcel_id"] in known:
            fields.append(known[fin["parcel_id"]])
            continue
        extra = {k: fin[k] for k in FIELD_PROPS if k in fin}
        fid = db.add_water_field(fin["parcel_id"], fin["irrigation_mode"], fin["geometry_wkt"], **extra)
        fields.append(db.get_water_field_by_id(fid))

    # Najpierw historia WSZYSTKICH pól — kohorta potrzebuje sąsiadów.
    for f in fields:
        try:
            st = backfill_field(f, t_start, t_end)
            print(f"  [{f['parcel_id']}] scen S2: {st['scenes']}, czystych: {st['clean']}, "
                  f"zapisanych: {st['stored']}, scen S1: {st['radar_scenes']}")
        except Exception as e:
            print(f"  [{f['parcel_id']}] BŁĄD pobierania: {e}", file=sys.stderr)

    import weather as weather_mod
    from shapely import wkt as shapely_wkt
    all_rows, all_alarms, summaries = [], [], []
    for f in fields:
        wx = []
        if not args.bez_bilansu:
            try:
                wx = weather_mod.fetch_weather_series(
                    shapely_wkt.loads(f["geometry_wkt"]), t_start - timedelta(days=1), t_end)
            except Exception as e:
                print(f"  [{f['parcel_id']}] pogoda niedostępna, bez bilansu: {e}", file=sys.stderr)
        res = replay_field(f, weather_series=wx)
        all_rows += res["rows"]
        all_alarms += res["alarms"]
        summaries.append(res["summary"])

    _write_csv(os.path.join(args.wyniki, "obserwacje.csv"), all_rows)
    _write_csv(os.path.join(args.wyniki, "alarmy.csv"), all_alarms)
    lines = [f"Powtórka sezonu {t_start}..{t_end} — progi robocze (niekalibrowane)", ""]
    for s in summaries:
        types = ", ".join(f"{k}: {v}" for k, v in sorted(s["alarms_by_type"].items())) or "brak"
        lines.append(f"{s['parcel_id']}: obserwacji {s['observations']} "
                     f"({s['first_obs']}..{s['last_obs']}), alarmów {s['alarms_total']} [{types}], "
                     f"spadków wyjaśnionych fenologią {s['phenology_suppressed']}, "
                     f"dni stresu w bilansie {s['stress_days_in_balance']}")
    total = sum(s["alarms_total"] for s in summaries)
    lines += ["", f"Razem alarmów: {total} na {len(summaries)} pól.",
              "Oceń każdy alarm w alarmy.csv: czy w tym terminie na polu był realny problem?",
              "To jest materiał do ustawienia progów przed sezonem."]
    with open(os.path.join(args.wyniki, "podsumowanie.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWyniki: {os.path.abspath(args.wyniki)}")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
