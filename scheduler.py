"""Scheduler monitoringu poplonów — odpytuje Sentinel Hub co 48h."""

import logging
import threading
from datetime import date, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

import core
import db

log = logging.getLogger("scheduler")
logging.basicConfig(
    filename="monitoring.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

_scheduler = None

# --- okna sezonowe (kiedy zaczyna się sezon dla danego typu poplonu) ---

def _season_start(typ, rok):
    if typ == "scierniskowy":
        return date(rok, 7, 1)
    return date(rok, 8, 1)  # ozimy


def _season_end(typ, rok):
    # Naprawa: dawniej osobna, zahardkodowana data (1.03) drygowała od
    # core.CFG["ozimy"]["t_end"] (wtedy 20.02) — teraz oba miejsca czytają tę
    # samą definicję, żeby scheduler w tle pobierał dane w tym samym oknie,
    # które pokazuje operatorowi wykres/API.
    return core.CFG[typ]["t_end"](rok)


# --- główna pętla ---

def check_all_monitored_parcels():
    log.info("START: sprawdzanie monitorowanych dzialek")
    parcels = db.get_monitored_parcels(status="active")
    log.info(f"Aktywnych dzialek: {len(parcels)}")

    for p in parcels:
        pid = p["id"]
        parcel_id = p["parcel_id"]
        typ = p["typ"]
        rok = p["rok"]
        try:
            _check_parcel(p)
            db.update_last_checked(pid)
            new_alarm_ids = detect_anomalies(pid, typ, rok)
            for alarm_id in new_alarm_ids:
                try:
                    import notifier
                    notifier.notify_alarm(alarm_id, pid)
                except Exception as e:
                    log.error(f"Blad powiadomienia alarm {alarm_id}: {e}")
        except Exception as e:
            log.error(f"Blad dla dzialki {parcel_id} (id={pid}): {e}")

    log.info("KONIEC: sprawdzanie monitorowanych dzialek")


def _check_parcel(p):
    pid = p["id"]
    parcel_id = p["parcel_id"]
    typ = p["typ"]
    rok = p["rok"]

    last_checked = p.get("last_checked_at")
    if last_checked:
        try:
            t_start = datetime.fromisoformat(last_checked).date() - timedelta(days=1)
        except ValueError:
            t_start = _season_start(typ, rok)
    else:
        t_start = _season_start(typ, rok)

    t_end = min(date.today(), _season_end(typ, rok))
    if t_start >= t_end:
        log.info(f"  {parcel_id}: zakres dat pusty ({t_start} >= {t_end}), pomijam")
        return

    log.info(f"  {parcel_id}: pobieranie NDVI {t_start} -> {t_end}")
    geom, _ = core.fetch_parcel_geometry(parcel_id)
    geom_buf, _, _ = core.inner_buffer(geom, 10.0)
    records = core.fetch_ndvi_series(geom_buf, t_start, t_end)

    added = 0
    for r in records:
        if r["valid_frac"] >= core.MIN_VALID_FRAC:
            obs_date = r["date"].isoformat() if hasattr(r["date"], "isoformat") else str(r["date"])
            if db.add_observation(pid, obs_date, r["ndvi_mean"], r["valid_frac"]):
                added += 1
    log.info(f"  {parcel_id}: dodano {added} nowych obserwacji")

    # Rozszerzony monitoring wielowskaźnikowy (S2): NDVI/EVI/SAVI/NDRE/NDWI
    try:
        from analytics import fetch_multi_series, save_multi_observation

        sowing_d = None
        if p.get("sowing_date"):
            try:
                sowing_d = date.fromisoformat(p["sowing_date"])
            except ValueError:
                pass
        term_d = None
        if p.get("termination_date"):
            try:
                term_d = date.fromisoformat(p["termination_date"])
            except ValueError:
                pass

        token = core.cdse_token()
        multi_records = fetch_multi_series(geom_buf, t_start, t_end, token=token)
        multi_added = 0
        for rec in multi_records:
            if rec.get("valid_frac", 0) >= core.MIN_VALID_FRAC:
                save_multi_observation(pid, rec, sowing_date=sowing_d, typ=typ,
                                       termination_date=term_d)
                multi_added += 1
        log.info(f"  {parcel_id}: zapisano {multi_added} obserwacji multi")
    except Exception as e:
        log.warning(f"  {parcel_id}: multi_observations pominięte ({e})")


def _add_alarm_with_indicators(parcel_db_id, alarm_type, alarm_date, ndvi_value, message):
    """Wrapper na db.add_alarm — dołącza snapshot EVI/SAVI/NDRE/NDWI/RVI z
    najbliższej obserwacji multi_observations (jeśli dostępna), żeby
    calibration.py mógł później sugerować korekty wag Confidence Score,
    nie tylko progów NDVI. Brak danych multi (np. tabela jeszcze nie
    zainicjalizowana) nie blokuje utworzenia alarmu — po prostu zostają None."""
    extra = {}
    try:
        from analytics import get_multi_indicators_near
        extra = get_multi_indicators_near(parcel_db_id, alarm_date, db_path=db.DB_PATH)
    except Exception as e:
        log.warning(f"  nie udało się dociągnąć wskaźników multi dla alarmu ({e})")

    return db.add_alarm(
        parcel_db_id, alarm_type, alarm_date, ndvi_value, message,
        evi_value=extra.get("evi"), savi_value=extra.get("savi"),
        ndre_value=extra.get("ndre"), ndwi_value=extra.get("ndwi"),
        rvi_value=extra.get("rvi"), reip_value=extra.get("reip"),
    )


def detect_anomalies(parcel_db_id, typ, rok):
    """Sprawdza anomalie i zwraca listę nowo dodanych alarm_id."""
    obs = db.get_observations(parcel_db_id)
    parcel = db.get_parcel_by_id(parcel_db_id)
    today = date.today()
    new_alarms = []

    if not obs:
        return []

    termination_date = None
    if parcel and parcel.get("termination_date"):
        try:
            termination_date = date.fromisoformat(parcel["termination_date"])
        except ValueError:
            pass

    # filtruj obserwacje do daty likwidacji (jesli podana)
    relevant_obs = [o for o in obs
                    if termination_date is None or o["obs_date"] <= termination_date.isoformat()]

    if not relevant_obs:
        return []

    last3 = relevant_obs[-3:]
    max_ndvi = max(o["ndvi_mean"] for o in relevant_obs)

    # alarm 1: brak wegetacji po terminie
    alarm_threshold_date = {
        "scierniskowy": date(rok, 9, 20),
        "ozimy": date(rok, 11, 1),
    }.get(typ)

    if alarm_threshold_date and today >= alarm_threshold_date:
        if len(last3) >= 2 and all(o["ndvi_mean"] < core.NDVI_BARE_MAX for o in last3):
            if not db.alarm_exists_recent(parcel_db_id, "no_vegetation", days=14):
                ndvi_val = last3[-1]["ndvi_mean"]
                aid = _add_alarm_with_indicators(
                    parcel_db_id, "no_vegetation", today.isoformat(), ndvi_val,
                    f"NDVI={ndvi_val:.2f} — brak wegetacji przez ostatnie {len(last3)} pomiary. "
                    f"Poplon powinien być widoczny po {alarm_threshold_date.isoformat()}.")
                new_alarms.append(aid)
                log.warning(f"  ALARM no_vegetation dla id={parcel_db_id}")

    # alarm 2: nagły spadek NDVI — interpretacja zależy od typu i okresu
    sowing_date_str = parcel.get("sowing_date") if parcel else None
    sowing_date_parsed = None
    if sowing_date_str:
        try:
            sowing_date_parsed = date.fromisoformat(sowing_date_str)
        except ValueError:
            pass

    SCIERNISKOWY_LATE_DROP_START = date(rok, 9, 20)   # po tej dacie spadek = prawdopodobna likwidacja
    OZIMY_PRESOWING_DROP_END = date(rok, 12, 1)        # przed tą datą spadek = podejrzane (zasiew)

    for i in range(1, len(relevant_obs)):
        prev = relevant_obs[i - 1]
        curr = relevant_obs[i]
        try:
            d_prev = date.fromisoformat(prev["obs_date"])
            d_curr = date.fromisoformat(curr["obs_date"])
        except ValueError:
            continue
        days_apart = (d_curr - d_prev).days
        drop = prev["ndvi_mean"] - curr["ndvi_mean"]

        if not (prev["ndvi_mean"] >= core.NDVI_COVER_MID and drop > 0.20 and days_apart <= 14):
            continue

        # pomiń jeśli to zadeklarowana likwidacja
        if termination_date and d_curr >= termination_date:
            continue

        if typ == "scierniskowy":
            # Spadek po 20 września dla ścierniskowego = prawdopodobna legalna likwidacja.
            # Nie alarmuj jeśli minęło 8 tygodni od siewu (lub siew nieznany).
            if d_curr >= SCIERNISKOWY_LATE_DROP_START:
                if sowing_date_parsed is None:
                    # brak daty siewu — nie wiemy czy 8 tyg. minęło, pomijamy
                    continue
                min_end = sowing_date_parsed + timedelta(weeks=8)
                if d_curr >= min_end:
                    # 8 tygodni minęło — to legalna likwidacja, nie alarm
                    log.info(f"  Spadek {d_curr} dla ścierniskowego po 8 tyg. od siewu "
                             f"({min_end}) — traktowane jako likwidacja, brak alarmu.")
                    continue
                # 8 tygodni NIE minęło — alarm, za wcześnie
                if not db.alarm_exists_recent(parcel_db_id, "sudden_drop", days=14):
                    aid = _add_alarm_with_indicators(
                        parcel_db_id, "sudden_drop", curr["obs_date"], curr["ndvi_mean"],
                        f"Gwałtowny spadek NDVI: {prev['ndvi_mean']:.2f} → {curr['ndvi_mean']:.2f} "
                        f"({prev['obs_date']} → {curr['obs_date']}) przed wymaganym minimum "
                        f"8 tygodni od siewu ({sowing_date_parsed} + 8 tyg. = {min_end}). "
                        f"Możliwa przedwczesna likwidacja.")
                    new_alarms.append(aid)
                    log.warning(f"  ALARM sudden_drop (za wczesna likwidacja) id={parcel_db_id}")
                    break
            else:
                # Spadek przed 20 września — nieoczekiwany w sezonie
                if not db.alarm_exists_recent(parcel_db_id, "sudden_drop", days=14):
                    aid = _add_alarm_with_indicators(
                        parcel_db_id, "sudden_drop", curr["obs_date"], curr["ndvi_mean"],
                        f"Gwałtowny spadek NDVI: {prev['ndvi_mean']:.2f} → {curr['ndvi_mean']:.2f} "
                        f"w ciągu {days_apart} dni ({prev['obs_date']} → {curr['obs_date']}). "
                        f"Nieoczekiwany przed sezonem oceny.")
                    new_alarms.append(aid)
                    log.warning(f"  ALARM sudden_drop id={parcel_db_id}")
                    break

        elif typ == "ozimy":
            # Dla ozimego: nagły spadek PRZED początkiem zimy (przed 1 grudnia)
            # może oznaczać zaoranie pod uprawę główną — to niezgodne z wymogami.
            if d_curr < OZIMY_PRESOWING_DROP_END:
                if not db.alarm_exists_recent(parcel_db_id, "sudden_drop", days=14):
                    aid = _add_alarm_with_indicators(
                        parcel_db_id, "sudden_drop", curr["obs_date"], curr["ndvi_mean"],
                        f"Gwałtowny spadek NDVI: {prev['ndvi_mean']:.2f} → {curr['ndvi_mean']:.2f} "
                        f"({prev['obs_date']} → {curr['obs_date']}) przed zimą. "
                        f"Możliwe zaoranie poplonu i zasiew uprawy głównej — "
                        f"niezgodne z wymogami FLD_KSR_004f (poplon musi pozostać do 15.02).")
                    new_alarms.append(aid)
                    log.warning(f"  ALARM sudden_drop (ozimy przed zima) id={parcel_db_id}")
                    break
            # Spadek po 1 grudnia dla ozimego: naturalny (mróz, śnieg) — nie alarmuj

    # alarm 3: NDVI nigdy nie osiagnelo wymaganego poziomu
    below_threshold_date = {
        "scierniskowy": date(rok, 10, 5),
        "ozimy": date(rok, 11, 20),
    }.get(typ)

    if below_threshold_date and today >= below_threshold_date:
        if max_ndvi < core.NDVI_COVER_MID:
            if not db.alarm_exists_recent(parcel_db_id, "below_threshold", days=14):
                aid = _add_alarm_with_indicators(
                    parcel_db_id, "below_threshold", today.isoformat(), max_ndvi,
                    f"Maksymalne NDVI={max_ndvi:.2f} ponizej progu {core.NDVI_COVER_MID:.2f} — "
                    f"poplon nigdy nie osiagnal wymaganego poziomu wegetacji.")
                new_alarms.append(aid)
                log.warning(f"  ALARM below_threshold dla id={parcel_db_id}")

    return new_alarms


def start_scheduler():
    global _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(check_all_monitored_parcels, "interval", hours=48,
                       id="monitor_job", replace_existing=True)
    _scheduler.start()
    log.info("Scheduler uruchomiony (co 48h)")


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Scheduler zatrzymany")
