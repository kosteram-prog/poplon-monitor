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
        CREATE TABLE IF NOT EXISTS weather_cache (
            key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            fetched_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS yield_maps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            parcel_id TEXT,
            crop_type TEXT,
            season_year INTEGER,
            date_from TEXT,
            date_to TEXT,
            reported_yield_t_ha REAL,
            geometry_wkt TEXT,
            status TEXT DEFAULT 'pending',
            error TEXT,
            stats_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS replay_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            obs_date TEXT NOT NULL,
            risk_type TEXT NOT NULL DEFAULT '',
            rating TEXT NOT NULL,
            metrics_json TEXT,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(water_field_id, obs_date, risk_type),
            FOREIGN KEY(water_field_id) REFERENCES water_monitored_fields(id)
        );
        CREATE TABLE IF NOT EXISTS rain_gauge_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            reading_date TEXT NOT NULL,
            precip_mm REAL NOT NULL,
            notes TEXT,
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
            ("water_observations", "cwb_30d_pct", "REAL"),
            ("water_observations", "vpd_max_kpa", "REAL"),
            ("water_observations", "tmax_c", "REAL"),
            ("water_observations", "sm_0_7", "REAL"),
            ("water_observations", "sm_7_28", "REAL"),
            ("water_observations", "sm_28_100", "REAL"),
            ("irrigation_events", "amount_mm", "REAL"),
        ):
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # kolumna już istnieje

        # Panel rolnika i SMS (listopad 2026): rolnik = numer telefonu
        # (klucz grupowania pól, jak w raporcie zbiorczym), dostęp do panelu
        # przez losowy token w linku (bez kont i haseł), skrzynka nadawcza
        # SMS (każda wiadomość — także niewysłana, gdy brak bramki — z
        # treścią do skopiowania) i odpowiedź rolnika na alert.
        c.executescript("""
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL UNIQUE,
            name TEXT,
            lang TEXT DEFAULT 'pl',
            token TEXT UNIQUE,
            sms_opt_in INTEGER DEFAULT 1,
            auto_suggestions INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            last_seen_at TEXT
        );
        CREATE TABLE IF NOT EXISTS sms_outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            phone TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            provider TEXT,
            provider_response TEXT,
            water_field_id INTEGER,
            water_alarm_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """)
        for table, col, typ in (
            ("water_alarms", "farmer_ack", "TEXT"),
            ("water_alarms", "farmer_ack_note", "TEXT"),
            ("water_alarms", "farmer_ack_at", "TEXT"),
            ("water_alarms", "farmer_notified_at", "TEXT"),
            ("rain_gauge_readings", "source", "TEXT"),
        ):
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass

        # OPTRAM (optram.py): parametry trapezu per pole wodne i seria
        # wilgotności względnej W (0–1) z Sentinel-2.
        c.executescript("""
        CREATE TABLE IF NOT EXISTS optram_params (
            water_field_id INTEGER PRIMARY KEY,
            i_d REAL, s_d REAL, i_w REAL, s_w REAL, r2_d REAL, r2_w REAL,
            ndvi_lo REAL, ndvi_hi REAL, n_px INTEGER, n_scenes INTEGER,
            date_from TEXT, date_to TEXT, region_wkt TEXT, quality TEXT,
            warnings_json TEXT, bins_json TEXT, scatter_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS optram_series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            obs_date TEXT NOT NULL,
            w REAL, ndvi REAL, valid_frac REAL,
            UNIQUE(water_field_id, obs_date)
        );
        """)

        # Termika Landsat (thermal.py): metryki per scena nad polem wodnym.
        c.executescript("""
        CREATE TABLE IF NOT EXISTS thermal_scenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            water_field_id INTEGER NOT NULL,
            scene_date TEXT NOT NULL,
            t_field REAL, t_p10 REAL, t_p90 REAL, ndvi REAL,
            t_cold REAL, t_hot REAL, tsi REAL, n_px INTEGER, valid_frac REAL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(water_field_id, scene_date)
        );
        """)

        # Próbki biomasy poplonu z pola (biomass_calibration.py) — do
        # kalibracji modelu biomasy/N z CIred-edge.
        c.executescript("""
        CREATE TABLE IF NOT EXISTS biomass_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parcel_id TEXT NOT NULL,
            sample_date TEXT NOT NULL,
            dm_t_ha REAL NOT NULL,
            n_kg_ha REAL,
            mix TEXT,
            fresh_g REAL, frame_m2 REAL, n_frames INTEGER, dm_pct REAL, n_pct REAL,
            geometry_wkt TEXT,
            notes TEXT,
            scene_date TEXT, delta_days INTEGER, valid_frac REAL,
            cire REAL, ndre REAL, ndvi REAL, evi REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """)

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
    "et0_mm", "cwb_14d_mm", "cwb_30d_mm", "cwb_30d_pct",
    "vpd_max_kpa", "tmax_c", "sm_0_7", "sm_7_28", "sm_28_100",
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


# ---------------------------------------------------------------------------
# weather_cache — pamięć podręczna odpowiedzi Open-Meteo (weather.om_get)
# ---------------------------------------------------------------------------

def get_weather_cache(key):
    """(payload, fetched_at_epoch) albo None."""
    with _lock, _conn() as c:
        row = c.execute("SELECT payload, fetched_at FROM weather_cache WHERE key=?", (key,)).fetchone()
        return (row[0], row[1]) if row else None


def set_weather_cache(key, payload):
    import time as _time
    with _lock, _conn() as c:
        c.execute("INSERT INTO weather_cache (key,payload,fetched_at) VALUES (?,?,?) "
                  "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, fetched_at=excluded.fetched_at",
                  (key, payload, _time.time()))


# ---------------------------------------------------------------------------
# rain_gauge_readings — odczyty deszczomierza rolnika. W bilansie wodnym
# zastępują opad z modelu (siatka 9-25 km nie widzi lokalnych burz).
# ---------------------------------------------------------------------------

def add_rain_reading(water_field_id, reading_date, precip_mm, notes=None):
    with _lock, _conn() as c:
        cur = c.execute("INSERT INTO rain_gauge_readings (water_field_id,reading_date,precip_mm,notes) "
                        "VALUES (?,?,?,?)", (water_field_id, reading_date, float(precip_mm), notes))
        return cur.lastrowid


def get_rain_readings(water_field_id):
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM rain_gauge_readings WHERE water_field_id=? "
                         "ORDER BY reading_date ASC, id ASC", (water_field_id,)).fetchall()
        return [dict(r) for r in rows]


def delete_rain_reading(reading_id):
    with _lock, _conn() as c:
        return c.execute("DELETE FROM rain_gauge_readings WHERE id=?", (reading_id,)).rowcount


# ---------------------------------------------------------------------------
# replay_ratings — oceny doradcy dla wyników powtórki sezonu (water_replay):
# 'confirmed' (alarm trafny), 'rejected' (fałszywy), 'missed' (problem był,
# alarmu nie było), 'unsure'. W odróżnieniu od kolejki triage obejmują OBA
# rodzaje błędów, więc kalibracja może próg zarówno zaostrzyć, jak i
# poluzować (water_replay.suggest_threshold_from_ratings).
# ---------------------------------------------------------------------------

_REPLAY_RATINGS = {"confirmed", "rejected", "missed", "unsure"}


def upsert_replay_rating(water_field_id, obs_date, rating, risk_type="", metrics_json=None, note=None):
    if rating not in _REPLAY_RATINGS:
        raise ValueError(f"Nieznana ocena: {rating!r}")
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO replay_ratings (water_field_id,obs_date,risk_type,rating,metrics_json,note) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(water_field_id,obs_date,risk_type) DO UPDATE SET "
            "rating=excluded.rating, metrics_json=excluded.metrics_json, note=excluded.note, "
            "created_at=datetime('now')",
            (water_field_id, obs_date, risk_type or "", rating, metrics_json, note))
        row = c.execute("SELECT * FROM replay_ratings WHERE water_field_id=? AND obs_date=? AND risk_type=?",
                        (water_field_id, obs_date, risk_type or "")).fetchone()
        return dict(row)


def get_replay_ratings(water_field_id=None):
    with _lock, _conn() as c:
        if water_field_id is None:
            rows = c.execute("SELECT * FROM replay_ratings ORDER BY obs_date").fetchall()
        else:
            rows = c.execute("SELECT * FROM replay_ratings WHERE water_field_id=? ORDER BY obs_date",
                             (water_field_id,)).fetchall()
        return [dict(r) for r in rows]


def delete_replay_rating(rating_id):
    with _lock, _conn() as c:
        return c.execute("DELETE FROM replay_ratings WHERE id=?", (rating_id,)).rowcount


# ---------------------------------------------------------------------------
# yield_maps — mapy stref plonu MVI (yield_mvi.py); pliki w uploads/yield_maps/<id>/
# ---------------------------------------------------------------------------

def add_yield_map(**kw):
    cols = ["label", "parcel_id", "crop_type", "season_year", "date_from", "date_to",
            "reported_yield_t_ha", "geometry_wkt"]
    vals = [kw.get(c) for c in cols]
    with _lock, _conn() as c:
        cur = c.execute(f"INSERT INTO yield_maps ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        return cur.lastrowid


def update_yield_map(map_id, status, stats_json=None, error=None):
    with _lock, _conn() as c:
        c.execute("UPDATE yield_maps SET status=?, stats_json=?, error=? WHERE id=?",
                  (status, stats_json, error, map_id))


def get_yield_maps():
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM yield_maps ORDER BY id DESC").fetchall()]


def get_yield_map(map_id):
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM yield_maps WHERE id=?", (map_id,)).fetchone()
        return dict(r) if r else None


def delete_yield_map(map_id):
    with _lock, _conn() as c:
        return c.execute("DELETE FROM yield_maps WHERE id=?", (map_id,)).rowcount


# ---------------------------------------------------------------------------
# Panel rolnika i SMS (farmer_panel.py). Rolnik identyfikowany numerem
# telefonu w postaci znormalizowanej (same cyfry z kodem kraju, bez "+").
# ---------------------------------------------------------------------------

_FARMER_EDITABLE = {"name", "lang", "sms_opt_in", "auto_suggestions"}
FARMER_LANGS = {"pl", "en", "ru", "uz"}


def get_farmers():
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM farmers ORDER BY name, phone").fetchall()]


def get_farmer_by_phone(phone):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM farmers WHERE phone=?", (phone,)).fetchone()
        return dict(row) if row else None


def get_farmer_by_id(farmer_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM farmers WHERE id=?", (farmer_id,)).fetchone()
        return dict(row) if row else None


def get_farmer_by_token(token):
    if not token:
        return None
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM farmers WHERE token=?", (token,)).fetchone()
        return dict(row) if row else None


def add_farmer(phone, name=None, lang="pl", token=None):
    if lang not in FARMER_LANGS:
        raise ValueError(f"Nieznany język: {lang!r}")
    with _lock, _conn() as c:
        cur = c.execute("INSERT INTO farmers (phone,name,lang,token) VALUES (?,?,?,?)",
                        (phone, name, lang, token))
        return cur.lastrowid


def update_farmer(farmer_id, **kwargs):
    unknown = set(kwargs) - _FARMER_EDITABLE - {"token"}
    if unknown:
        raise ValueError(f"Nieznane pola rolnika: {sorted(unknown)}")
    if "lang" in kwargs and kwargs["lang"] not in FARMER_LANGS:
        raise ValueError(f"Nieznany język: {kwargs['lang']!r}")
    if not kwargs:
        return
    cols = ",".join(f"{k}=?" for k in kwargs)
    with _lock, _conn() as c:
        c.execute(f"UPDATE farmers SET {cols} WHERE id=?", (*kwargs.values(), farmer_id))


def touch_farmer_seen(farmer_id):
    with _lock, _conn() as c:
        c.execute("UPDATE farmers SET last_seen_at=datetime('now') WHERE id=?", (farmer_id,))


def add_sms(phone, kind, text, status, farmer_id=None, provider=None, provider_response=None,
            water_field_id=None, water_alarm_id=None):
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO sms_outbox (farmer_id,phone,kind,text,status,provider,provider_response,"
            "water_field_id,water_alarm_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (farmer_id, phone, kind, text, status, provider, provider_response,
             water_field_id, water_alarm_id))
        return cur.lastrowid


def get_sms_outbox(limit=50, farmer_id=None):
    q = "SELECT * FROM sms_outbox"
    params = []
    if farmer_id is not None:
        q += " WHERE farmer_id=?"
        params.append(farmer_id)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(q, params).fetchall()]


def count_sms_sent_since(phone, since_sql_modifier="-1 day", kind=None, water_field_id=None):
    """Liczba SMS-ów faktycznie wysłanych (status 'sent') na numer od
    datetime('now', since_sql_modifier) — do limitu kosztów i do
    nieponawiania tej samej sugestii nawadniania."""
    q = ("SELECT COUNT(*) FROM sms_outbox WHERE phone=? AND status='sent' "
         "AND created_at >= datetime('now', ?)")
    params = [phone, since_sql_modifier]
    if kind:
        q += " AND kind=?"
        params.append(kind)
    if water_field_id is not None:
        q += " AND water_field_id=?"
        params.append(water_field_id)
    with _lock, _conn() as c:
        return c.execute(q, params).fetchone()[0]


def last_sms(kind, water_field_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM sms_outbox WHERE kind=? AND water_field_id=? AND status='sent' "
                        "ORDER BY id DESC LIMIT 1", (kind, water_field_id)).fetchone()
        return dict(row) if row else None


_FARMER_ACKS = {"seen", "will_irrigate", "irrigated", "disagree"}


def set_water_alarm_farmer_ack(alarm_id, ack, note=None):
    if ack not in _FARMER_ACKS:
        raise ValueError(f"Nieznana odpowiedź rolnika: {ack!r}")
    with _lock, _conn() as c:
        c.execute("UPDATE water_alarms SET farmer_ack=?, farmer_ack_note=?, "
                  "farmer_ack_at=datetime('now') WHERE id=?", (ack, note, alarm_id))


def mark_water_alarm_farmer_notified(alarm_id):
    with _lock, _conn() as c:
        c.execute("UPDATE water_alarms SET notified_farmer=1, farmer_notified_at=datetime('now') "
                  "WHERE id=?", (alarm_id,))


def set_rain_reading_source(reading_id, source):
    with _lock, _conn() as c:
        c.execute("UPDATE rain_gauge_readings SET source=? WHERE id=?", (source, reading_id))


# ---------------------------------------------------------------------------
# OPTRAM (optram.py)
# ---------------------------------------------------------------------------

_OPTRAM_COLS = ("i_d", "s_d", "i_w", "s_w", "r2_d", "r2_w", "ndvi_lo", "ndvi_hi", "n_px", "n_scenes",
                "date_from", "date_to", "region_wkt", "quality", "warnings_json", "bins_json",
                "scatter_json")


def save_optram_params(water_field_id, row: dict):
    vals = [row.get(k) for k in _OPTRAM_COLS]
    cols = ",".join(_OPTRAM_COLS)
    with _lock, _conn() as c:
        c.execute("DELETE FROM optram_params WHERE water_field_id=?", (water_field_id,))
        c.execute(f"INSERT INTO optram_params (water_field_id,{cols}) VALUES (?{',?' * len(vals)})",
                  (water_field_id, *vals))


def get_optram_params(water_field_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM optram_params WHERE water_field_id=?", (water_field_id,)).fetchone()
        return dict(row) if row else None


def delete_optram(water_field_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM optram_params WHERE water_field_id=?", (water_field_id,))
        c.execute("DELETE FROM optram_series WHERE water_field_id=?", (water_field_id,))


def clear_optram_series(water_field_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM optram_series WHERE water_field_id=?", (water_field_id,))


def upsert_optram_series(water_field_id, rows):
    with _lock, _conn() as c:
        for r in rows:
            c.execute("INSERT INTO optram_series (water_field_id,obs_date,w,ndvi,valid_frac) VALUES (?,?,?,?,?) "
                      "ON CONFLICT(water_field_id,obs_date) DO UPDATE SET w=excluded.w, ndvi=excluded.ndvi, "
                      "valid_frac=excluded.valid_frac",
                      (water_field_id, r["obs_date"], r["w"], r["ndvi"], r.get("valid_frac")))


def get_optram_series(water_field_id):
    with _lock, _conn() as c:
        rows = c.execute("SELECT obs_date,w,ndvi,valid_frac FROM optram_series WHERE water_field_id=? "
                         "ORDER BY obs_date ASC", (water_field_id,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Termika Landsat (thermal.py)
# ---------------------------------------------------------------------------

_THERMAL_COLS = ("t_field", "t_p10", "t_p90", "ndvi", "t_cold", "t_hot", "tsi", "n_px", "valid_frac")


def upsert_thermal(water_field_id, rows):
    cols = ",".join(_THERMAL_COLS)
    upd = ",".join(f"{k}=excluded.{k}" for k in _THERMAL_COLS)
    with _lock, _conn() as c:
        for r in rows:
            c.execute(f"INSERT INTO thermal_scenes (water_field_id,scene_date,{cols}) "
                      f"VALUES (?,?{',?' * len(_THERMAL_COLS)}) "
                      f"ON CONFLICT(water_field_id,scene_date) DO UPDATE SET {upd}",
                      (water_field_id, r["scene_date"], *[r.get(k) for k in _THERMAL_COLS]))


def get_thermal_series(water_field_id):
    with _lock, _conn() as c:
        rows = c.execute(f"SELECT scene_date,{','.join(_THERMAL_COLS)} FROM thermal_scenes "
                         "WHERE water_field_id=? ORDER BY scene_date ASC", (water_field_id,)).fetchall()
        return [dict(r) for r in rows]


def delete_thermal(water_field_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM thermal_scenes WHERE water_field_id=?", (water_field_id,))


# ---------------------------------------------------------------------------
# Próbki biomasy poplonu (biomass_calibration.py)
# ---------------------------------------------------------------------------

_BIOMASS_SAMPLE_COLS = ("parcel_id", "sample_date", "dm_t_ha", "n_kg_ha", "mix", "fresh_g", "frame_m2",
                        "n_frames", "dm_pct", "n_pct", "geometry_wkt", "notes")
_BIOMASS_MATCH_COLS = ("scene_date", "delta_days", "valid_frac", "cire", "ndre", "ndvi", "evi")


def add_biomass_sample(**kw):
    unknown = set(kw) - set(_BIOMASS_SAMPLE_COLS)
    if unknown:
        raise ValueError(f"Nieznane pola próbki: {sorted(unknown)}")
    cols = [k for k in _BIOMASS_SAMPLE_COLS if k in kw]
    with _lock, _conn() as c:
        cur = c.execute(f"INSERT INTO biomass_samples ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [kw[k] for k in cols])
        return cur.lastrowid


def update_biomass_sample_match(sample_id, match: dict):
    vals = [match.get(k) for k in _BIOMASS_MATCH_COLS]
    with _lock, _conn() as c:
        c.execute(f"UPDATE biomass_samples SET {','.join(k + '=?' for k in _BIOMASS_MATCH_COLS)} WHERE id=?",
                  (*vals, sample_id))


def get_biomass_samples():
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM biomass_samples ORDER BY sample_date DESC, id DESC").fetchall()]


def get_biomass_sample(sample_id):
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM biomass_samples WHERE id=?", (sample_id,)).fetchone()
        return dict(row) if row else None


def delete_biomass_sample(sample_id):
    with _lock, _conn() as c:
        c.execute("DELETE FROM biomass_samples WHERE id=?", (sample_id,))
