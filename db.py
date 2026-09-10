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
        """)
        # Migracja dla baz założonych przed dodaniem tej kolumny —
        # CREATE TABLE IF NOT EXISTS nie dotknie już istniejącej tabeli.
        try:
            c.execute("ALTER TABLE alarms ADD COLUMN photo_summary_notified INTEGER DEFAULT 0")
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
