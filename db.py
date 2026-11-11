"""Warstwa bazy danych SQLite dla monitoringu poplonów."""

import sqlite3
import threading
from datetime import datetime

DB_PATH = "monitoring.db"
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS monitored_parcels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT NOT NULL,
            typ TEXT NOT NULL,
            rok INTEGER NOT NULL,
            sowing_date TEXT,
            termination_date TEXT,
            termination_method TEXT,
            farmer_phone TEXT,
            farmer_name TEXT,
            notes TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            last_checked_at TEXT
        );
        CREATE TABLE IF NOT EXISTS ndvi_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_db_id INTEGER NOT NULL,
            obs_date TEXT NOT NULL,
            ndvi_mean REAL NOT NULL,
            valid_frac REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(parcel_db_id, obs_date),
            FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
        );
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_db_id INTEGER NOT NULL,
            alarm_type TEXT NOT NULL,
            alarm_date TEXT NOT NULL,
            ndvi_value REAL,
            evi_value REAL,
            savi_value REAL,
            ndre_value REAL,
            ndwi_value REAL,
            rvi_value REAL,
            reip_value REAL,
            message TEXT,
            notified_operator INTEGER DEFAULT 0,
            notified_farmer INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            photo_summary_notified INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(parcel_db_id) REFERENCES monitored_parcels(id)
        );
        CREATE TABLE IF NOT EXISTS farmer_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alarm_id INTEGER NOT NULL,
            parcel_db_id INTEGER NOT NULL,
            photo_path TEXT NOT NULL,
            farmer_comment TEXT,
            ai_analysis TEXT,
            ai_coverage_pct REAL,
            ai_verdict TEXT,
            ndvi_at_alarm REAL,
            evi_at_alarm REAL,
            savi_at_alarm REAL,
            ndre_at_alarm REAL,
            ndwi_at_alarm REAL,
            rvi_at_alarm REAL,
            reip_at_alarm REAL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(alarm_id) REFERENCES alarms(id)
        );
        CREATE TABLE IF NOT EXISTS threshold_calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            changed_at TEXT DEFAULT (datetime('now')),
            old_cover_high REAL,
            new_cover_high REAL,
            old_cover_mid REAL,
            new_cover_mid REAL,
            samples_count INTEGER,
            approved_by TEXT DEFAULT 'operator'
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS water_monitored_fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT NOT NULL,
            irrigation_mode TEXT NOT NULL DEFAULT 'rainfed',
            geometry_wkt TEXT,
            region TEXT,
            farmer_phone TEXT,
            farmer_name TEXT,
            notes TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            last_checked_at TEXT
        );
        CREATE TABLE IF NOT EXISTS irrigation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            event_date TEXT NOT NULL,
            method TEXT NOT NULL,
            amount_or_duration REAL,
            amount_unit TEXT,
            source TEXT DEFAULT 'manual',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(water_field_id) REFERENCES water_monitored_fields(id)
        );
        CREATE TABLE IF NOT EXISTS water_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            obs_date TEXT NOT NULL,
            ndwi_mean REAL,
            ndvi_mean REAL,
            vv_db REAL,
            vh_db REAL,
            rvi REAL,
            valid_frac REAL,
            precip_mm REAL,
            dry_spell INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(water_field_id, obs_date),
            FOREIGN KEY(water_field_id) REFERENCES water_monitored_fields(id)
        );
        CREATE TABLE IF NOT EXISTS water_alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            obs_date TEXT NOT NULL,
            risk_type TEXT NOT NULL,
            message TEXT,
            calibration_status TEXT,
            ndwi_value REAL,
            vv_value REAL,
            components_json TEXT,
            triage_status TEXT DEFAULT 'pending',
            triage_note TEXT,
            triage_by TEXT,
            triaged_at TEXT,
            notified_farmer INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(water_field_id) REFERENCES water_monitored_fields(id)
        );
        CREATE TABLE IF NOT EXISTS irrigation_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            planned_date TEXT NOT NULL,
            method TEXT,
            notes TEXT,
            status TEXT DEFAULT 'planned',
            irrigation_event_id INTEGER,
            completed_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(water_field_id) REFERENCES water_monitored_fields(id),
            FOREIGN KEY(irrigation_event_id) REFERENCES irrigation_events(id)
        );
        """)
        # Migracja dla baz założonych przed dodaniem tej kolumny —
        # CREATE TABLE IF NOT EXISTS nie dotknie już istniejącej tabeli.
        try:
            c.execute("ALTER TABLE alarms ADD COLUMN photo_summary_notified INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # kolumna już istnieje

        # Migracja dla baz water_monitored_fields założonych przed decyzją,
        # że geometria pola jest wymagana WPROST (nie doczytywana z ULDK) —
        # wczesna wersja tej tabeli (przed rozpoznaniem, że ULDK nie działa
        # dla pól poza Polską, np. Uzbekistan) nie miała tej kolumny.
        try:
            c.execute("ALTER TABLE water_monitored_fields ADD COLUMN geometry_wkt TEXT")
        except sqlite3.OperationalError:
            pass  # kolumna już istnieje

        # Migracje modułu wodnego (listopad 2026): uprawa (klucz kohorty),
        # parametry bilansu wodnego FAO-56 oraz nowe kolumny obserwacji
        # (bilans P-ET0, równomierność NDMI w polu, orbita S1).
        for table, col, typ in (
            ("water_monitored_fields", "crop_type", "TEXT"),
            ("water_monitored_fields", "sowing_date", "TEXT"),
            ("water_monitored_fields", "soil_taw_mm", "REAL"),
            ("water_observations", "et0_mm", "REAL"),
            ("water_observations", "cwb_14d_mm", "REAL"),
            ("water_observations", "cwb_30d_mm", "REAL"),
            ("water_observations", "ndwi_p10", "REAL"),
            ("water_observations", "ndwi_p50", "REAL"),
            ("water_observations", "ndwi_low_frac", "REAL"),
            ("water_observations", "s1_rel_orbit", "INTEGER"),
            ("water_observations", "s1_date", "TEXT"),
            ("irrigation_events", "amount_mm", "REAL"),
        ):
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # kolumna już istnieje

        # Migracje dla baz założonych przed dodaniem wielowskaźnikowego
        # snapshotu przy alarmie (potrzebnego do sugestii wag w calibration.py).
        for table, cols in (
            ("alarms", ("evi_value", "savi_value", "ndre_value", "ndwi_value",
                        "rvi_value", "reip_value")),
            ("farmer_photos", ("evi_at_alarm", "savi_at_alarm", "ndre_at_alarm",
                               "ndwi_at_alarm", "rvi_at_alarm", "reip_at_alarm")),
        ):
            for col in cols:
                try:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
                except sqlite3.OperationalError:
                    pass  # kolumna już istnieje


# --- monitored_parcels ---

_MONITORED_PARCEL_OPTIONAL_COLUMNS = {
    "sowing_date", "termination_date", "termination_method",
    "farmer_phone", "farmer_name", "notes", "status",
}


def add_monitored_parcel(parcel_id, typ, rok, **kwargs):
    # Whitelist nazw kolumn — nie ufamy kluczom kwargs, żeby wykluczyć
    # wstrzyknięcie dowolnych identyfikatorów SQL przez przyszłych wołających.
    unknown = set(kwargs.keys()) - _MONITORED_PARCEL_OPTIONAL_COLUMNS
    if unknown:
        raise ValueError(f"Nieznane pola dla monitored_parcels: {sorted(unknown)}")
    fields = ["parcel_id", "typ", "rok"] + list(kwargs.keys())
    values = [parcel_id, typ, rok] + list(kwargs.values())
    placeholders = ",".join(["?"] * len(values))
    cols = ",".join(fields)
    with _lock, _conn() as c:
        cur = c.execute(f"INSERT INTO monitored_parcels ({cols}) VALUES ({placeholders})", values)
        return cur.lastrowid


def get_monitored_parcels(status=None):
    with _lock, _conn() as c:
        if status:
            rows = c.execute("SELECT * FROM monitored_parcels WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM monitored_parcels ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_monitored_parcels_by_farmer_phone(farmer_phone, status=None):
    """Wszystkie monitorowane działki jednego rolnika (klucz: farmer_phone —
    bardziej jednoznaczny niż farmer_name, który bywa różnie wpisywany).
    Używane do zbiorczego raportu PDF (patrz report_pdf.py)."""
    with _lock, _conn() as c:
        if status:
            rows = c.execute(
                "SELECT * FROM monitored_parcels WHERE farmer_phone=? AND status=? "
                "ORDER BY created_at DESC", (farmer_phone, status)).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM monitored_parcels WHERE farmer_phone=? "
                "ORDER BY created_at DESC", (farmer_phone,)).fetchall()
        return [dict(r) for r in rows]


def get_parcel_by_id(parcel_db_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM monitored_parcels WHERE id=?", (parcel_db_id,)).fetchone()
        return dict(row) if row else None


def update_parcel_status(parcel_db_id, status):
    with _lock, _conn() as c:
        c.execute("UPDATE monitored_parcels SET status=? WHERE id=?", (status, parcel_db_id))


def update_last_checked(parcel_db_id):
    with _lock, _conn() as c:
        c.execute("UPDATE monitored_parcels SET last_checked_at=datetime('now') WHERE id=?", (parcel_db_id,))


# --- ndvi_observations ---

def add_observation(parcel_db_id, obs_date, ndvi_mean, valid_frac):
    try:
        with _lock, _conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO ndvi_observations (parcel_db_id,obs_date,ndvi_mean,valid_frac) VALUES (?,?,?,?)",
                (parcel_db_id, obs_date, ndvi_mean, valid_frac))
            return True
    except Exception:
        return False


def get_observations(parcel_db_id):
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM ndvi_observations WHERE parcel_db_id=? ORDER BY obs_date ASC",
            (parcel_db_id,)).fetchall()
        return [dict(r) for r in rows]


# --- alarms ---

def add_alarm(parcel_db_id, alarm_type, alarm_date, ndvi_value, message,
              evi_value=None, savi_value=None, ndre_value=None,
              ndwi_value=None, rvi_value=None, reip_value=None):
    """evi_value/savi_value/ndre_value/ndwi_value/rvi_value/reip_value są
    opcjonalne (backward-compat) — jeśli podane, tworzą snapshot wszystkich
    wskaźników w momencie alarmu, używany później przez calibration.py do
    sugerowania korekt wag Confidence Score (nie tylko progów NDVI).
    reip_value: potrzebne, żeby calibration.py mogło ocenić separację tego
    samego zblendowanego sygnału red-edge (NDRE+REIP), który faktycznie jest
    ważony w compute_confidence_score — patrz analytics.blend_red_edge_norm."""
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO alarms (parcel_db_id,alarm_type,alarm_date,ndvi_value,message,"
            "evi_value,savi_value,ndre_value,ndwi_value,rvi_value,reip_value) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (parcel_db_id, alarm_type, alarm_date, ndvi_value, message,
             evi_value, savi_value, ndre_value, ndwi_value, rvi_value, reip_value))
        return cur.lastrowid


def get_alarms(parcel_db_id=None, resolved=None):
    q = "SELECT * FROM alarms WHERE 1=1"
    params = []
    if parcel_db_id is not None:
        q += " AND parcel_db_id=?"
        params.append(parcel_db_id)
    if resolved is not None:
        q += " AND resolved=?"
        params.append(int(resolved))
    q += " ORDER BY created_at DESC"
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(q, params).fetchall()]


def get_alarm_by_id(alarm_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM alarms WHERE id=?", (alarm_id,)).fetchone()
        return dict(row) if row else None


def resolve_alarm(alarm_id):
    with _lock, _conn() as c:
        c.execute("UPDATE alarms SET resolved=1 WHERE id=?", (alarm_id,))


def claim_photo_notification(alarm_id) -> bool:
    """Atomowo 'zajmuje' prawo do wysłania zbiorczego podsumowania analizy
    zdjęć dla danego alarmu. Zwraca True tylko wątkowi, który jako pierwszy
    to zrobił — chroni przed podwójną wysyłką maila, gdy rolnik prześle dwie
    partie zdjęć dla tego samego alarmu blisko siebie w czasie (dwa
    równoległe wątki `_analyze_photos`)."""
    with _lock, _conn() as c:
        cur = c.execute(
            "UPDATE alarms SET photo_summary_notified=1 "
            "WHERE id=? AND photo_summary_notified=0",
            (alarm_id,))
        return cur.rowcount > 0


def release_photo_notification_claim(alarm_id):
    """Cofa 'zajęcie' z claim_photo_notification, gdy wysyłka się nie
    powiodła (albo warunki wstępne nie zostały spełnione) — pozwala
    spróbować ponownie przy kolejnej partii zdjęć."""
    with _lock, _conn() as c:
        c.execute("UPDATE alarms SET photo_summary_notified=0 WHERE id=?", (alarm_id,))


def alarm_exists_recent(parcel_db_id, alarm_type, days=14):
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) FROM alarms WHERE parcel_db_id=? AND alarm_type=? "
            "AND alarm_date >= date('now',?) AND resolved=0",
            (parcel_db_id, alarm_type, f"-{days} days")).fetchone()
        return row[0] > 0


def mark_alarm_notified(alarm_id, operator=False, farmer=False):
    with _lock, _conn() as c:
        if operator:
            c.execute("UPDATE alarms SET notified_operator=1 WHERE id=?", (alarm_id,))
        if farmer:
            c.execute("UPDATE alarms SET notified_farmer=1 WHERE id=?", (alarm_id,))


# --- farmer_photos ---

def add_farmer_photo(alarm_id, parcel_db_id, photo_path, farmer_comment=None, ndvi_at_alarm=None,
                      evi_at_alarm=None, savi_at_alarm=None, ndre_at_alarm=None,
                      ndwi_at_alarm=None, rvi_at_alarm=None, reip_at_alarm=None):
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO farmer_photos (alarm_id,parcel_db_id,photo_path,farmer_comment,ndvi_at_alarm,"
            "evi_at_alarm,savi_at_alarm,ndre_at_alarm,ndwi_at_alarm,rvi_at_alarm,reip_at_alarm) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (alarm_id, parcel_db_id, photo_path, farmer_comment, ndvi_at_alarm,
             evi_at_alarm, savi_at_alarm, ndre_at_alarm, ndwi_at_alarm, rvi_at_alarm, reip_at_alarm))
        return cur.lastrowid


def update_photo_analysis(photo_id, ai_analysis, ai_coverage_pct, ai_verdict):
    with _lock, _conn() as c:
        c.execute(
            "UPDATE farmer_photos SET ai_analysis=?,ai_coverage_pct=?,ai_verdict=? WHERE id=?",
            (ai_analysis, ai_coverage_pct, ai_verdict, photo_id))


def get_photos_for_calibration(min_samples=5):
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM farmer_photos WHERE ai_verdict IS NOT NULL "
            "AND ai_coverage_pct IS NOT NULL AND ndvi_at_alarm IS NOT NULL").fetchall()
        return [dict(r) for r in rows]


def get_photos_by_alarm(alarm_id):
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM farmer_photos WHERE alarm_id=?", (alarm_id,)).fetchall()]


# --- threshold_calibration_log ---

def log_threshold_change(old_high, new_high, old_mid, new_mid, samples_count):
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO threshold_calibration_log "
            "(old_cover_high,new_cover_high,old_cover_mid,new_cover_mid,samples_count) "
            "VALUES (?,?,?,?,?)",
            (old_high, new_high, old_mid, new_mid, samples_count))


def get_last_calibration():
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT * FROM threshold_calibration_log ORDER BY changed_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None


# --- settings (key/value, np. kalibrowalne progi NDVI) ---------------------

def get_setting(key, default=None):
    with _lock, _conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key, value):
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO settings (key,value,updated_at) VALUES (?,?,datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, str(value)))


# ---------------------------------------------------------------------------
# water_monitored_fields / irrigation_events — moduł teledetekcji i wody,
# Krok 1 planu (log nawodnienia). Osobna tabela i lista pól od
# monitored_parcels — cykl życia pola wodnego nie jest przywiązany do
# sezonu poplonu (brak sowing_date/termination_date/typ). irrigation_mode
# to atrybut KONKRETNEGO pola ('irrigation'|'rainfed'), nie globalna flaga
# aplikacji — jedna instancja obsługuje jednocześnie pola nawadniane
# (Uzbekistan) i uprawy deszczowe (Europa/Lower Silesia 360).
# ---------------------------------------------------------------------------

_WATER_IRRIGATION_MODES = {"irrigation", "rainfed"}

# Whitelist opcjonalnych kolumn — analogicznie do
# _MONITORED_PARCEL_OPTIONAL_COLUMNS: nie ufamy kluczom kwargs, żeby
# wykluczyć wstrzyknięcie dowolnych identyfikatorów SQL przez przyszłych
# wołających.
_WATER_FIELD_OPTIONAL_COLUMNS = {
    "region", "farmer_phone", "farmer_name", "notes", "status",
    "crop_type", "sowing_date", "soil_taw_mm",
}

# Kolumny, które wolno zmienić po utworzeniu pola (update_water_field).
_WATER_FIELD_EDITABLE_COLUMNS = {
    "region", "farmer_phone", "farmer_name", "notes",
    "crop_type", "sowing_date", "soil_taw_mm",
}

_IRRIGATION_METHODS = {"zalewowe", "kroplowe", "deszczowanie"}


def add_water_field(parcel_id, irrigation_mode, geometry_wkt, **kwargs):
    if irrigation_mode not in _WATER_IRRIGATION_MODES:
        raise ValueError(f"Nieznany irrigation_mode: {irrigation_mode!r}")
    try:
        from shapely import wkt as _shapely_wkt
        _geom = _shapely_wkt.loads(geometry_wkt)
        if _geom.is_empty:
            raise ValueError("geometria jest pusta")
    except Exception as e:
        raise ValueError(f"Nieprawidlowa geometria WKT dla water_monitored_fields: {e}")
    unknown = set(kwargs.keys()) - _WATER_FIELD_OPTIONAL_COLUMNS
    if unknown:
        raise ValueError(f"Nieznane pola dla water_monitored_fields: {sorted(unknown)}")
    fields = ["parcel_id", "irrigation_mode", "geometry_wkt"] + list(kwargs.keys())
    values = [parcel_id, irrigation_mode, geometry_wkt] + list(kwargs.values())
    placeholders = ",".join(["?"] * len(values))
    cols = ",".join(fields)
    with _lock, _conn() as c:
        cur = c.execute(f"INSERT INTO water_monitored_fields ({cols}) VALUES ({placeholders})", values)
        return cur.lastrowid


def get_water_fields(status=None):
    with _lock, _conn() as c:
        if status:
            rows = c.execute(
                "SELECT * FROM water_monitored_fields WHERE status=? ORDER BY created_at DESC",
                (status,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM water_monitored_fields ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_water_field_by_id(water_field_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM water_monitored_fields WHERE id=?", (water_field_id,)).fetchone()
        return dict(row) if row else None


def update_water_field_status(water_field_id, status):
    with _lock, _conn() as c:
        c.execute("UPDATE water_monitored_fields SET status=? WHERE id=?", (status, water_field_id))


def update_water_field(water_field_id, **kwargs):
    """Zmienia edytowalne atrybuty pola wodnego (whitelist kolumn).
    Zwraca liczbę zmienionych wierszy (0 = brak pola)."""
    unknown = set(kwargs.keys()) - _WATER_FIELD_EDITABLE_COLUMNS
    if unknown:
        raise ValueError(f"Nieznane/nieedytowalne pola water_monitored_fields: {sorted(unknown)}")
    if not kwargs:
        return 0
    sets = ",".join(f"{k}=?" for k in kwargs)
    with _lock, _conn() as c:
        cur = c.execute(f"UPDATE water_monitored_fields SET {sets} WHERE id=?",
                        list(kwargs.values()) + [water_field_id])
        return cur.rowcount


def update_water_field_last_checked(water_field_id):
    with _lock, _conn() as c:
        c.execute("UPDATE water_monitored_fields SET last_checked_at=datetime('now') WHERE id=?",
                  (water_field_id,))


# Jednostki dawki nawodnienia przeliczalne wprost na mm słupa wody
# (bilans wodny, water_balance.py). Czas trwania (h, min) nie jest
# przeliczalny bez wydatku instalacji — wtedy amount_mm zostaje puste.
_MM_UNITS = {"mm", "l/m2", "l/m²", "dm3/m2"}
_M3HA_UNITS = {"m3/ha", "m³/ha", "m3ha", "m3"}


def irrigation_amount_mm(amount, unit) -> "float | None":
    """Dawka w mm z (wartość, jednostka). None, gdy nie da się przeliczyć."""
    if amount is None or unit is None:
        return None
    u = str(unit).strip().lower().replace(" ", "")
    if u in _MM_UNITS:
        return float(amount)
    if u in _M3HA_UNITS:
        return float(amount) / 10.0   # 1 mm = 10 m³/ha
    return None


def add_irrigation_event(water_field_id, event_date, method, amount_or_duration=None,
                          amount_unit=None, source="manual", notes=None, amount_mm=None):
    if method not in _IRRIGATION_METHODS:
        raise ValueError(f"Nieznana metoda nawodnienia: {method!r}")
    if amount_mm is None:
        amount_mm = irrigation_amount_mm(amount_or_duration, amount_unit)
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO irrigation_events (water_field_id,event_date,method,"
            "amount_or_duration,amount_unit,source,notes,amount_mm) VALUES (?,?,?,?,?,?,?,?)",
            (water_field_id, event_date, method, amount_or_duration, amount_unit, source, notes,
             amount_mm))
        return cur.lastrowid


def get_irrigation_events(water_field_id):
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM irrigation_events WHERE water_field_id=? ORDER BY event_date ASC",
            (water_field_id,)).fetchall()
        return [dict(r) for r in rows]


def get_irrigation_events_for_calibration(min_samples=5):
    """Surowe dane wejściowe pod przyszłą rekalibrację progów anomalii
    wodnej (etap 3 planu, analogicznie do get_photos_for_calibration).
    Sygnatura z min_samples zarezerwowana już teraz (próg stosuje wołający,
    tak jak dziś robi to calibration.compute_suggested_thresholds), żeby
    uniknąć migracji API w etapie 3."""
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM irrigation_events ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# water_observations — historia obserwacji per pole wodne (NDWI/NDVI z S2,
# VV/VH/RVI z S1, opady i flaga posuchy z dnia obserwacji). Potrzebne, żeby
# water_stress.py mogło liczyć anomalię względem WŁASNEJ HISTORII pola bez
# każdorazowego przeliczania całego okna od nowa (analogicznie do
# ndvi_observations / sentinel1.save_s1_observations).
# ---------------------------------------------------------------------------

_WATER_OBS_EXTRA_COLUMNS = {
    "et0_mm", "cwb_14d_mm", "cwb_30d_mm",
    "ndwi_p10", "ndwi_p50", "ndwi_low_frac",
    "s1_rel_orbit", "s1_date",
}


def add_water_observation(water_field_id, obs_date, ndwi_mean=None, ndvi_mean=None,
                           vv_db=None, vh_db=None, rvi=None, valid_frac=None,
                           precip_mm=None, dry_spell=False, **extra):
    unknown = set(extra.keys()) - _WATER_OBS_EXTRA_COLUMNS
    if unknown:
        raise ValueError(f"Nieznane pola dla water_observations: {sorted(unknown)}")
    cols = ["water_field_id", "obs_date", "ndwi_mean", "ndvi_mean", "vv_db", "vh_db",
            "rvi", "valid_frac", "precip_mm", "dry_spell"] + list(extra.keys())
    vals = [water_field_id, obs_date, ndwi_mean, ndvi_mean, vv_db, vh_db, rvi,
            valid_frac, precip_mm, int(bool(dry_spell))] + list(extra.values())
    try:
        with _lock, _conn() as c:
            c.execute(
                f"INSERT OR IGNORE INTO water_observations ({','.join(cols)}) "
                f"VALUES ({','.join(['?'] * len(vals))})", vals)
            return True
    except Exception:
        return False


def get_water_observations(water_field_id, limit=None):
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM water_observations WHERE water_field_id=? ORDER BY obs_date ASC",
            (water_field_id,)).fetchall()
        rows = [dict(r) for r in rows]
        return rows[-limit:] if limit else rows


# ---------------------------------------------------------------------------
# water_alarms — kolejka triage doradcy dla silnika anomalii stresu wodnego
# (water_stress.py, Krok 2). Wzorem `alarms` (moduł poplonów): alarm to
# NIEPOTWIERDZONA flaga ryzyka z silnika, którą doradca musi
# potwierdzić/odrzucić/oznaczyć jako wymagającą wizyty (sekcja 4 planu:
# "Alert trafia najpierw do kolejki doradcy, nie bezpośrednio do rolnika").
# `notified_farmer` zarezerwowane pod Krok 4 (panel rolnika) — świadomie
# NIE jest jeszcze nigdzie ustawiane w tym kroku.
# ---------------------------------------------------------------------------

_WATER_TRIAGE_STATUSES = {"pending", "confirmed", "rejected", "needs_visit"}


def add_water_alarm(water_field_id, obs_date, risk_type, message, calibration_status,
                     ndwi_value=None, vv_value=None, components_json=None):
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO water_alarms (water_field_id,obs_date,risk_type,message,"
            "calibration_status,ndwi_value,vv_value,components_json) VALUES (?,?,?,?,?,?,?,?)",
            (water_field_id, obs_date, risk_type, message, calibration_status,
             ndwi_value, vv_value, components_json))
        return cur.lastrowid


def get_water_alarms(water_field_id=None, triage_status=None):
    q = "SELECT * FROM water_alarms WHERE 1=1"
    params = []
    if water_field_id is not None:
        q += " AND water_field_id=?"
        params.append(water_field_id)
    if triage_status is not None:
        q += " AND triage_status=?"
        params.append(triage_status)
    q += " ORDER BY created_at DESC"
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(q, params).fetchall()]


def get_water_alarm_by_id(alarm_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM water_alarms WHERE id=?", (alarm_id,)).fetchone()
        return dict(row) if row else None


def water_alarm_exists_recent(water_field_id, risk_type, days=7):
    """Analogicznie do alarm_exists_recent w module poplonów — nie twórz
    kolejnego alarmu tego samego typu dla tego samego pola, jeśli jeden już
    czeka/został niedawno rozstrzygnięty (chroni kolejkę triage przed
    zalaniem tym samym sygnałem co 48h przy kolejnych przebiegach schedulera)."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT 1 FROM water_alarms WHERE water_field_id=? AND risk_type=? "
            "AND created_at >= datetime('now', ?) LIMIT 1",
            (water_field_id, risk_type, f"-{int(days)} days")).fetchone()
        return row is not None


def set_water_alarm_triage(alarm_id, triage_status, triage_note=None, triage_by=None):
    if triage_status not in _WATER_TRIAGE_STATUSES:
        raise ValueError(f"Nieznany triage_status: {triage_status!r}")
    with _lock, _conn() as c:
        c.execute(
            "UPDATE water_alarms SET triage_status=?, triage_note=?, triage_by=?, "
            "triaged_at=datetime('now') WHERE id=?",
            (triage_status, triage_note, triage_by, alarm_id))


# ---------------------------------------------------------------------------
# irrigation_plans — planowanie nawadniania (sekcja 5.2 planu, "faza 2").
# Celowo PROSTY kalendarz/harmonogram, nie system rekomendacji dawek: pole
# method jest tu opcjonalne (w przeciwienstwie do irrigation_events, gdzie
# jest wymagane) - na etapie planowania rolnik czesto jeszcze nie wie, jaka
# metoda uzyje. "wykonano" (complete_irrigation_plan w warstwie API) moze
# opcjonalnie od razu zalogowac realne zdarzenie w irrigation_events, zeby
# rolnik nie musial wpisywac tych samych danych dwa razy - ale to osobna
# tabela, bo plan i rzeczywiste zdarzenie to rozne rzeczy (plan moze zostac
# pominiety/przesuniety, patrz status 'skipped').
# ---------------------------------------------------------------------------

_IRRIGATION_PLAN_STATUSES = {"planned", "done", "skipped"}


def add_irrigation_plan(water_field_id, planned_date, method=None, notes=None):
    if method is not None and method not in _IRRIGATION_METHODS:
        raise ValueError(f"Nieznana metoda nawodnienia: {method!r}")
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO irrigation_plans (water_field_id,planned_date,method,notes) "
            "VALUES (?,?,?,?)",
            (water_field_id, planned_date, method, notes))
        return cur.lastrowid


def get_irrigation_plans(water_field_id=None, status=None):
    q = "SELECT * FROM irrigation_plans WHERE 1=1"
    params = []
    if water_field_id is not None:
        q += " AND water_field_id=?"
        params.append(water_field_id)
    if status is not None:
        q += " AND status=?"
        params.append(status)
    q += " ORDER BY planned_date ASC"
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(q, params).fetchall()]


def get_irrigation_plan_by_id(plan_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM irrigation_plans WHERE id=?", (plan_id,)).fetchone()
        return dict(row) if row else None


def set_irrigation_plan_status(plan_id, status, irrigation_event_id=None, notes=None):
    if status not in _IRRIGATION_PLAN_STATUSES:
        raise ValueError(f"Nieznany status planu nawodnienia: {status!r}")
    with _lock, _conn() as c:
        if notes is not None:
            if status in ("done", "skipped"):
                c.execute(
                    "UPDATE irrigation_plans SET status=?, irrigation_event_id=?, "
                    "notes=?, completed_at=datetime('now') WHERE id=?",
                    (status, irrigation_event_id, notes, plan_id))
            else:
                c.execute(
                    "UPDATE irrigation_plans SET status=?, irrigation_event_id=?, "
                    "notes=?, completed_at=NULL WHERE id=?",
                    (status, irrigation_event_id, notes, plan_id))
        else:
            if status in ("done", "skipped"):
                c.execute(
                    "UPDATE irrigation_plans SET status=?, irrigation_event_id=?, "
                    "completed_at=datetime('now') WHERE id=?",
                    (status, irrigation_event_id, plan_id))
            else:
                c.execute(
                    "UPDATE irrigation_plans SET status=?, irrigation_event_id=?, "
                    "completed_at=NULL WHERE id=?",
                    (status, irrigation_event_id, plan_id))


def delete_irrigation_plan(plan_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM irrigation_plans WHERE id=?", (plan_id,))
