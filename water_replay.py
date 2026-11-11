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
samymi kolumnami. Wynik: katalog replay_wyniki/ z obserwacje.csv,
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
    stored = 0
    for opt in sorted(clean, key=lambda r: r["date"]):
        obs = ws.build_observation(field, opt, sources)
        if obs["obs_date"] in existing:
            continue
        if db.add_water_observation(**obs):
            stored += 1
    if stored:
        db.update_water_field_last_checked(field["id"])
    return {"scenes": len(optical), "clean": len(clean), "stored": stored,
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
    kcb_max, p, _ = wb.crop_params(field.get("crop_type"))
    taw = float(field.get("soil_taw_mm") or wb.DEFAULT_TAW_MM)
    ndvi_points = [(date.fromisoformat(o["obs_date"]), o["ndvi_mean"])
                   for o in observations if o.get("ndvi_mean") is not None]
    start = None
    if field.get("sowing_date"):
        try:
            start = date.fromisoformat(field["sowing_date"])
        except ValueError:
            start = None
    days = wb.run_balance(weather_series, ndvi_points, events, taw, p, kcb_max, start_date=start)
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
            "risk_flag": verdict.get("risk_flag"), "risk_type": verdict.get("risk_type"),
            "phenology_explained": verdict.get("phenology_explained"),
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
    return {"rows": rows, "alarms": alarms, "summary": summary}


# ---------------------------------------------------------------------------
# 3. Wczytywanie pól (GeoJSON / CSV) i CLI
# ---------------------------------------------------------------------------

FIELD_PROPS = ("region", "crop_type", "soil_taw_mm", "sowing_date")


def _to_wgs84(geom, crs_name: Optional[str]):
    if not crs_name or crs_name.upper() in ("EPSG:4326", "OGC:CRS84", "CRS84",
                                             "URN:OGC:DEF:CRS:OGC:1.3:CRS84"):
        return geom
    from pyproj import Transformer
    from shapely.ops import transform
    code = crs_name.split("::")[-1] if "::" in crs_name else crs_name
    if code.isdigit():
        code = f"EPSG:{code}"
    tr = Transformer.from_crs(code, "EPSG:4326", always_xy=True).transform
    return transform(tr, geom)


def load_fields(path: str) -> list:
    """Lista pól {parcel_id, irrigation_mode, geometry_wkt, region, ...}."""
    from shapely.geometry import shape
    from shapely import wkt as shapely_wkt
    out = []
    if path.lower().endswith((".geojson", ".json")):
        with open(path, encoding="utf-8") as f:
            gj = json.load(f)
        crs_name = ((gj.get("crs") or {}).get("properties") or {}).get("name")
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        for i, ft in enumerate(feats, 1):
            props = ft.get("properties") or {}
            geom = _to_wgs84(shape(ft["geometry"]), crs_name)
            out.append(_field_from_props(props, geom.wkt, i))
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                wkt = row.get("wkt") or row.get("WKT") or row.get("geometry_wkt")
                if not wkt:
                    raise ValueError(f"Wiersz {i}: brak kolumny 'wkt'")
                shapely_wkt.loads(wkt)
                out.append(_field_from_props(row, wkt, i))
    return out


def _field_from_props(props: dict, wkt: str, i: int) -> dict:
    pid = str(props.get("parcel_id") or props.get("id") or props.get("name") or f"pole-{i}")
    mode = (props.get("irrigation_mode") or "rainfed").strip().lower()
    if mode not in ("irrigation", "rainfed"):
        mode = "irrigation" if mode in ("irygacyjny", "nawadniane", "irrigated") else "rainfed"
    f = {"parcel_id": pid, "irrigation_mode": mode, "geometry_wkt": wkt}
    for k in FIELD_PROPS:
        v = props.get(k)
        if v not in (None, ""):
            f[k] = float(v) if k == "soil_taw_mm" else str(v)
    return f


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
