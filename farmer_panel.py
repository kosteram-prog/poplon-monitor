"""Panel rolnika i SMS dla modułu wodnego.

Zasada z planu modułu wodnego: alert z silnika trafia najpierw do kolejki
doradcy (water_alarms, triage), a do rolnika dopiero POTWIERDZONY. Ten moduł
jest tym ostatnim krokiem:

* rolnik = numer telefonu (tak jak w raporcie zbiorczym poplonów); pola
  przypisuje się mu przez farmer_phone pola wodnego, rekord w tabeli farmers
  powstaje automatycznie przy pierwszym odczycie listy rolników;
* panel rolnika to jedna strona /f/<token> bez logowania: losowy token w
  linku z SMS-a, do odwołania przez doradcę (nowy link unieważnia stary);
* SMS idzie: (a) gdy doradca potwierdzi alert i zaznaczy „SMS do rolnika”,
  (b) gdy doradca ręcznie wyśle sugestię nawodnienia albo link do panelu,
  (c) automatycznie raz dziennie z sugestią nawodnienia — tylko dla
  rolników, którym doradca świadomie to włączył (auto_suggestions);
* każda wiadomość ląduje w skrzynce sms_outbox — także niewysłana (brak
  bramki SMS, rolnik wypisany, limit dzienny), żeby doradca widział treść
  i mógł ją przekazać inną drogą.

Bramki: SMSAPI.pl (SMSAPI_TOKEN, jak dotychczasowe SMS-y poplonów) albo
Twilio (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM) — Twilio
przydaje się dla numerów spoza Polski (Uzbekistan). Wybór: SMS_PROVIDER
(smsapi | twilio), domyślnie ta, dla której są dane dostępowe.
"""
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import db
from auth import require_operator_key

log = logging.getLogger("farmer_panel")

router = APIRouter(tags=["farmer"])

DEFAULT_COUNTRY_CODE = "48"
MAX_SMS_PER_FARMER_DAY = 4          # limit kosztów i zabezpieczenie przed pętlą
AUTO_SUGGEST_DAYS = 2               # auto-SMS, gdy stres trwa albo przyjdzie w ciągu tylu dni
SUGGESTION_REPEAT_DAYS = 3          # nie ponawiaj sugestii dla pola częściej niż co tyle dni
RECENT_IRRIGATION_DAYS = 2          # świeże nawodnienie wstrzymuje auto-sugestię
ALERT_WINDOW_DAYS = 30              # alerty pokazywane w panelu rolnika
FARMER_LOG_MAX_AGE_DAYS = 30        # jak daleko wstecz rolnik może wpisać nawodnienie/opad
BALANCE_CACHE_TTL_S = 3600

BALANCE_RISK_TYPE = "water_balance_deficit"
UNIFORMITY_RISK_TYPES = ("irrigation_uniformity_anomaly", "uneven_moisture_risk")


# ---------------------------------------------------------------------------
# Numery telefonów
# ---------------------------------------------------------------------------

def normalize_phone(raw) -> Optional[str]:
    """Numer w postaci samych cyfr z kodem kraju ('48600100200',
    '998901234567'). 9 cyfr bez kodu = numer krajowy w kraju
    SMS_DEFAULT_COUNTRY (domyślnie Polska). None, gdy to nie wygląda na numer."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if not plus and digits.startswith("00"):
        digits = digits[2:]
        plus = True
    if not plus and len(digits) == 9:
        digits = os.environ.get("SMS_DEFAULT_COUNTRY", DEFAULT_COUNTRY_CODE) + digits
    if not 10 <= len(digits) <= 15:
        return None
    return digits


def display_phone(phone: str) -> str:
    return f"+{phone}" if phone else ""


def default_lang_for_phone(phone: str) -> str:
    if phone.startswith("48"):
        return "pl"
    if phone.startswith("998"):
        return "uz"
    if phone.startswith(("7", "992", "996", "993")):
        return "ru"
    return "en"


def new_token() -> str:
    return secrets.token_urlsafe(16)


# ---------------------------------------------------------------------------
# Rolnicy <-> pola
# ---------------------------------------------------------------------------

def sync_farmers_from_fields() -> int:
    """Zakłada rekordy rolników dla numerów z pól wodnych, których jeszcze
    nie ma w tabeli farmers. Zwraca liczbę nowych."""
    added = 0
    for f in db.get_water_fields():
        phone = normalize_phone(f.get("farmer_phone"))
        if not phone or db.get_farmer_by_phone(phone):
            continue
        try:
            db.add_farmer(phone, name=f.get("farmer_name"), lang=default_lang_for_phone(phone),
                          token=new_token())
            added += 1
        except Exception as e:  # wyścig dwóch żądań — rekord już jest
            log.info(f"sync_farmers_from_fields: {phone}: {e}")
    return added


def fields_for_farmer(farmer: dict, status: Optional[str] = "active") -> list:
    return [f for f in db.get_water_fields(status=status)
            if normalize_phone(f.get("farmer_phone")) == farmer["phone"]]


def farmer_for_field(field: dict) -> Optional[dict]:
    phone = normalize_phone(field.get("farmer_phone"))
    if not phone:
        return None
    farmer = db.get_farmer_by_phone(phone)
    if farmer is None:
        sync_farmers_from_fields()
        farmer = db.get_farmer_by_phone(phone)
    return farmer


def ensure_token(farmer: dict) -> dict:
    if not farmer.get("token"):
        db.update_farmer(farmer["id"], token=new_token())
        farmer = db.get_farmer_by_id(farmer["id"])
    return farmer


def panel_link(farmer: dict) -> str:
    base = os.environ.get("APP_BASE_URL", "").rstrip("/")
    return f"{base}/f/{farmer['token']}"


def field_label(field: dict, max_len: int = 18) -> str:
    pid = str(field.get("parcel_id") or field.get("id"))
    return pid if len(pid) <= max_len else "…" + pid[-(max_len - 1):]


# ---------------------------------------------------------------------------
# Teksty SMS (krótkie, bez polskich znaków dla pl/en/uz — 160 znaków GSM na
# segment; ru cyrylicą = 70 znaków na segment, więc tam teksty najkrótsze)
# ---------------------------------------------------------------------------

SMS_TEXTS = {
    "pl": {
        "balance": "PopMonitor, pole {f}: brakuje wody w glebie. Doradca zaleca nawodnienie ok. {d} mm ({m} m3/ha) do {when}. {link}",
        "balance_nodose": "PopMonitor, pole {f}: brakuje wody w glebie, doradca zaleca nawodnienie. {link}",
        "stress": "PopMonitor, pole {f}: satelita pokazuje stres wodny roslin, doradca to potwierdzil. Sprawdz pole: {link}",
        "performance": "PopMonitor, pole {f}: rosliny przesychaja mimo nawadniania. Sprawdz instalacje: {link}",
        "uniformity": "PopMonitor, pole {f}: czesc pola jest wyraznie suchsza. Sprawdz rownomiernosc nawadniania: {link}",
        "generic": "PopMonitor, pole {f}: doradca potwierdzil problem z woda. Szczegoly: {link}",
        "suggestion": "PopMonitor, pole {f}: wg modelu zapas wody skonczy sie {when}. Sugerowane nawodnienie ok. {d} mm ({m} m3/ha). {link}",
        "link": "PopMonitor: Twoje pola - zapas wody w glebie i zapis nawodnien: {link}",
        "today": "dzis", "tomorrow": "jutro",
    },
    "en": {
        "balance": "PopMonitor, field {f}: soil is running dry. Your advisor recommends irrigating approx. {d} mm ({m} m3/ha) by {when}. {link}",
        "balance_nodose": "PopMonitor, field {f}: soil is running dry, your advisor recommends irrigating. {link}",
        "stress": "PopMonitor, field {f}: satellite shows crop water stress, confirmed by your advisor. Check the field: {link}",
        "performance": "PopMonitor, field {f}: crop is drying despite irrigation. Check the system: {link}",
        "uniformity": "PopMonitor, field {f}: part of the field is clearly drier. Check irrigation uniformity: {link}",
        "generic": "PopMonitor, field {f}: your advisor confirmed a water problem. Details: {link}",
        "suggestion": "PopMonitor, field {f}: model says soil water runs out {when}. Suggested irrigation approx. {d} mm ({m} m3/ha). {link}",
        "link": "PopMonitor: your fields - soil water and irrigation log: {link}",
        "today": "today", "tomorrow": "tomorrow",
    },
    "ru": {
        "balance": "Поле {f}: в почве мало воды. Агроном советует полив ~{d} мм ({m} м3/га) до {when}. {link}",
        "balance_nodose": "Поле {f}: в почве мало воды, агроном советует полив. {link}",
        "stress": "Поле {f}: спутник видит водный стресс, агроном подтвердил. {link}",
        "performance": "Поле {f}: растения сохнут несмотря на полив. Проверьте систему: {link}",
        "uniformity": "Поле {f}: часть поля заметно суше. Проверьте равномерность полива: {link}",
        "generic": "Поле {f}: агроном подтвердил проблему с водой. {link}",
        "suggestion": "Поле {f}: запас воды кончится {when}. Полив ~{d} мм ({m} м3/га). {link}",
        "link": "PopMonitor: ваши поля - запас воды и журнал полива: {link}",
        "today": "сегодня", "tomorrow": "завтра",
    },
    "uz": {
        "balance": "PopMonitor, {f} dala: tuproqda suv kam. Agronom {when} gacha taxminan {d} mm ({m} m3/ga) sug'orishni tavsiya qiladi. {link}",
        "balance_nodose": "PopMonitor, {f} dala: tuproqda suv kam, agronom sug'orishni tavsiya qiladi. {link}",
        "stress": "PopMonitor, {f} dala: sun'iy yo'ldosh suv tanqisligini ko'rsatmoqda, agronom tasdiqladi. {link}",
        "performance": "PopMonitor, {f} dala: sug'orishga qaramay o'simliklar qurimoqda. Tizimni tekshiring: {link}",
        "uniformity": "PopMonitor, {f} dala: dalaning bir qismi ancha quruq. Sug'orish tekisligini tekshiring: {link}",
        "generic": "PopMonitor, {f} dala: agronom suv muammosini tasdiqladi. {link}",
        "suggestion": "PopMonitor, {f} dala: hisobga ko'ra suv zaxirasi {when} tugaydi. Taxminan {d} mm ({m} m3/ga) sug'oring. {link}",
        "link": "PopMonitor: dalalaringiz - tuproqdagi suv va sug'orish jurnali: {link}",
        "today": "bugun", "tomorrow": "ertaga",
    },
}

GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
           "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà")
GSM7_EXT = set("^{}\\[~]|€")


def sms_segments(text: str) -> int:
    """Liczba części SMS (GSM-7: 160/153 znaki, inaczej UCS-2: 70/67)."""
    if all(ch in GSM7 or ch in GSM7_EXT for ch in text):
        n = sum(2 if ch in GSM7_EXT else 1 for ch in text)
        return 1 if n <= 160 else -(-n // 153)
    n = len(text)
    return 1 if n <= 70 else -(-n // 67)


def _lang(farmer: Optional[dict]) -> str:
    lang = (farmer or {}).get("lang") or "pl"
    return lang if lang in SMS_TEXTS else "pl"


def _when(d_iso: Optional[str], lang: str, today: Optional[date] = None) -> str:
    today = today or date.today()
    tx = SMS_TEXTS[lang]
    if not d_iso:
        return tx["today"]
    try:
        d = date.fromisoformat(str(d_iso)[:10])
    except ValueError:
        return str(d_iso)
    if d <= today:
        return tx["today"]
    if d == today + timedelta(days=1):
        return tx["tomorrow"]
    return d.strftime("%d.%m")


def _fmt_num(v) -> str:
    return f"{float(v):.0f}"


def alarm_kind(risk_type: str) -> str:
    if risk_type == BALANCE_RISK_TYPE:
        return "balance"
    if risk_type == "water_stress_risk":
        return "stress"
    if risk_type == "irrigation_performance_anomaly":
        return "performance"
    if risk_type in UNIFORMITY_RISK_TYPES:
        return "uniformity"
    return "generic"


def _balance_from_alarm(alarm: dict) -> dict:
    try:
        comp = json.loads(alarm.get("components_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return (comp or {}).get("water_balance") or {}


def alert_sms_text(alarm: dict, field: dict, farmer: dict, today: Optional[date] = None) -> str:
    lang = _lang(farmer)
    tx = SMS_TEXTS[lang]
    kind = alarm_kind(alarm.get("risk_type", ""))
    link = panel_link(farmer)
    f = field_label(field)
    if kind == "balance":
        bal = _balance_from_alarm(alarm)
        if bal.get("suggested_dose_mm") is not None:
            return tx["balance"].format(
                f=f, d=_fmt_num(bal["suggested_dose_mm"]),
                m=_fmt_num(bal.get("suggested_dose_m3_ha") or bal["suggested_dose_mm"] * 10),
                when=_when(bal.get("suggested_irrigation_date"), lang, today), link=link)
        return tx["balance_nodose"].format(f=f, link=link)
    return tx[kind].format(f=f, link=link)


def suggestion_sms_text(summary: dict, field: dict, farmer: dict, today: Optional[date] = None) -> str:
    lang = _lang(farmer)
    return SMS_TEXTS[lang]["suggestion"].format(
        f=field_label(field), d=_fmt_num(summary["suggested_dose_mm"]),
        m=_fmt_num(summary.get("suggested_dose_m3_ha") or summary["suggested_dose_mm"] * 10),
        when=_when(summary.get("suggested_irrigation_date"), lang, today), link=panel_link(farmer))


def link_sms_text(farmer: dict) -> str:
    return SMS_TEXTS[_lang(farmer)]["link"].format(link=panel_link(farmer))


# ---------------------------------------------------------------------------
# Wysyłka
# ---------------------------------------------------------------------------

def sms_provider() -> Optional[str]:
    choice = os.environ.get("SMS_PROVIDER", "").strip().lower()
    has_smsapi = bool(os.environ.get("SMSAPI_TOKEN"))
    has_twilio = all(os.environ.get(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))
    if choice == "twilio":
        return "twilio" if has_twilio else None
    if choice == "smsapi":
        return "smsapi" if has_smsapi else None
    if has_smsapi:
        return "smsapi"
    if has_twilio:
        return "twilio"
    return None


def _send_smsapi(phone: str, text: str) -> tuple:
    data = {"to": phone, "message": text, "format": "json", "encoding": "utf-8"}
    sender = os.environ.get("SMSAPI_SENDER", "PopMonitor")
    if sender:
        data["from"] = sender
    r = requests.post("https://api.smsapi.pl/sms.do",
                      headers={"Authorization": f"Bearer {os.environ['SMSAPI_TOKEN']}"},
                      data=data, timeout=15)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:300]}
    ok = r.status_code == 200 and int(body.get("count", 0) or 0) > 0
    return ok, json.dumps(body, ensure_ascii=False)[:1000]


def _send_twilio(phone: str, text: str) -> tuple:
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                      auth=(sid, os.environ["TWILIO_AUTH_TOKEN"]),
                      data={"To": f"+{phone}", "From": os.environ["TWILIO_FROM"], "Body": text},
                      timeout=15)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:300]}
    ok = r.status_code in (200, 201) and body.get("status") not in ("failed", "undelivered")
    return ok, json.dumps({k: body.get(k) for k in ("sid", "status", "error_code", "message", "raw")
                           if body.get(k) is not None}, ensure_ascii=False)[:1000]


def send_sms(farmer: dict, text: str, kind: str, water_field_id=None, water_alarm_id=None) -> dict:
    """Wysyła (albo tylko odkłada do skrzynki) SMS do rolnika. Nigdy nie
    rzuca — wynik w polu status: sent | failed | not_configured |
    opted_out | rate_limited."""
    phone = farmer["phone"]
    provider = sms_provider()
    status, response = None, None
    if not farmer.get("sms_opt_in", 1):
        status = "opted_out"
    elif db.count_sms_sent_since(phone, "-1 day") >= int(
            os.environ.get("SMS_MAX_PER_FARMER_DAY", MAX_SMS_PER_FARMER_DAY)):
        status = "rate_limited"
    elif provider is None:
        status = "not_configured"
    else:
        try:
            ok, response = (_send_twilio if provider == "twilio" else _send_smsapi)(phone, text)
            status = "sent" if ok else "failed"
        except Exception as e:
            status, response = "failed", str(e)[:500]
    sms_id = db.add_sms(phone, kind, text, status, farmer_id=farmer["id"], provider=provider,
                        provider_response=response, water_field_id=water_field_id,
                        water_alarm_id=water_alarm_id)
    if status == "sent":
        log.info(f"SMS {kind} -> {display_phone(phone)} (id {sms_id})")
    else:
        log.warning(f"SMS {kind} -> {display_phone(phone)} nie wysłany: {status} {response or ''}")
    return {"id": sms_id, "status": status, "text": text, "phone": display_phone(phone),
            "segments": sms_segments(text), "provider": provider}


def notify_confirmed_alarm(alarm_id: int) -> dict:
    """SMS do rolnika po potwierdzeniu alertu przez doradcę."""
    alarm = db.get_water_alarm_by_id(alarm_id)
    if not alarm:
        raise LookupError("Nie ma takiego alertu.")
    if alarm.get("triage_status") != "confirmed":
        raise ValueError("SMS do rolnika tylko dla alertu potwierdzonego przez doradcę.")
    field = db.get_water_field_by_id(alarm["water_field_id"])
    farmer = farmer_for_field(field or {})
    if not farmer:
        return {"status": "no_phone", "text": None}
    farmer = ensure_token(farmer)
    res = send_sms(farmer, alert_sms_text(alarm, field, farmer), "alert",
                   water_field_id=field["id"], water_alarm_id=alarm_id)
    if res["status"] == "sent":
        db.mark_water_alarm_farmer_notified(alarm_id)
    return res


def send_panel_link(farmer_id: int) -> dict:
    farmer = db.get_farmer_by_id(farmer_id)
    if not farmer:
        raise LookupError("Nie ma takiego rolnika.")
    farmer = ensure_token(farmer)
    return send_sms(farmer, link_sms_text(farmer), "panel_link")


# ---------------------------------------------------------------------------
# Bilans pól rolnika (z krótką pamięcią podręczną — panel otwierany z
# telefonu nie powinien za każdym razem odpytywać pogody)
# ---------------------------------------------------------------------------

_balance_cache: dict = {}
_balance_lock = threading.Lock()


def _balance_key(field: dict) -> tuple:
    ev = db.get_irrigation_events(field["id"])
    rain = db.get_rain_readings(field["id"])
    checks = db.get_soil_checks(field["id"])
    return (field["id"], date.today().isoformat(), len(ev), len(rain), len(checks),
            field.get("crop_type"), field.get("soil_taw_mm"), field.get("sowing_date"),
            tuple(sorted((k, v) for k, v in (db.get_field_soil(field["id"]) or {}).items()
                         if k in ("texture_class", "root_limit_cm", "sg_fetched_at", "salinity_class",
                                  "ec_value", "ec_method", "gw_depth_m"))))


def field_balance_summary(field: dict, use_cache: bool = True) -> dict:
    import water_balance
    key = _balance_key(field)
    now = time.time()
    if use_cache:
        with _balance_lock:
            hit = _balance_cache.get(key)
            if hit and now - hit[0] < BALANCE_CACHE_TTL_S:
                return hit[1]
    summary = water_balance.compute_field_balance(field, quick_soil=True)["summary"]
    with _balance_lock:
        _balance_cache[key] = (now, summary)
    return summary


def clear_balance_cache():
    with _balance_lock:
        _balance_cache.clear()


def suggestion_due(summary: dict) -> bool:
    if summary.get("suggested_dose_mm") is None:
        return False
    if summary.get("status") == "stress":
        return True
    dts = summary.get("days_to_stress")
    return dts is not None and dts <= AUTO_SUGGEST_DAYS


def send_irrigation_suggestion(water_field_id: int, summary: Optional[dict] = None,
                               kind: str = "suggestion") -> dict:
    field = db.get_water_field_by_id(water_field_id)
    if not field:
        raise LookupError("Nie ma takiego pola.")
    if field.get("irrigation_mode") != "irrigation":
        raise ValueError("Sugestie nawodnienia tylko dla pól nawadnianych.")
    farmer = farmer_for_field(field)
    if not farmer:
        return {"status": "no_phone", "text": None}
    farmer = ensure_token(farmer)
    summary = summary or field_balance_summary(field)
    if summary.get("suggested_dose_mm") is None:
        raise ValueError("Model bilansu nie przewiduje stresu w horyzoncie prognozy — nie ma czego sugerować.")
    return send_sms(farmer, suggestion_sms_text(summary, field, farmer), kind,
                    water_field_id=water_field_id)


def auto_suggestions_job(today: Optional[date] = None) -> list:
    """Codzienny przebieg: automatyczne SMS-y z sugestią nawodnienia dla
    rolników z włączonym auto_suggestions. Zwraca listę wyników."""
    today = today or date.today()
    out = []
    sync_farmers_from_fields()
    for farmer in db.get_farmers():
        if not farmer.get("auto_suggestions") or not farmer.get("sms_opt_in", 1):
            continue
        for field in fields_for_farmer(farmer):
            if field.get("irrigation_mode") != "irrigation":
                continue
            try:
                if db.count_sms_sent_since(farmer["phone"], f"-{SUGGESTION_REPEAT_DAYS} days",
                                           kind="auto_suggestion", water_field_id=field["id"]):
                    continue
                recent = [e for e in db.get_irrigation_events(field["id"])
                          if e.get("event_date") and
                          date.fromisoformat(e["event_date"][:10]) >= today - timedelta(days=RECENT_IRRIGATION_DAYS)]
                if recent:
                    continue
                import water_overview
                trust = water_overview.balance_trust(field, today)
                if not trust["trusted"]:
                    log.info(f"auto_suggestions_job: pole {field['id']} pominięte ({trust['reason']})")
                    continue
                summary = field_balance_summary(field, use_cache=False)
                if not suggestion_due(summary):
                    continue
                res = send_irrigation_suggestion(field["id"], summary, kind="auto_suggestion")
                out.append({"water_field_id": field["id"], **res})
            except Exception as e:
                log.error(f"auto_suggestions_job: pole {field.get('id')}: {e}")
    return out


# ---------------------------------------------------------------------------
# Widok rolnika (dane do strony /f/<token>)
# ---------------------------------------------------------------------------

def _public_alarm(a: dict) -> dict:
    bal = _balance_from_alarm(a) if a.get("risk_type") == BALANCE_RISK_TYPE else {}
    return {
        "id": a["id"], "obs_date": a.get("obs_date"), "kind": alarm_kind(a.get("risk_type", "")),
        "advisor_note": a.get("triage_note"), "confirmed_at": a.get("triaged_at"),
        "farmer_ack": a.get("farmer_ack"), "farmer_ack_note": a.get("farmer_ack_note"),
        "dose_mm": bal.get("suggested_dose_mm"), "dose_m3_ha": bal.get("suggested_dose_m3_ha"),
        "by_date": bal.get("suggested_irrigation_date"),
    }


def _public_balance(s: dict) -> dict:
    keys = ("status", "as_of", "depletion_mm", "depletion_frac", "raw_mm", "taw_mm",
            "days_to_stress", "suggested_irrigation_date", "suggested_dose_mm",
            "suggested_dose_m3_ha", "initial_state_reliable", "forecast", "cwb_30d_class",
            "cwb_30d_pct", "soil_moisture", "anchored_from", "season_start")
    return {k: s.get(k) for k in keys if k in s}


def farmer_view(farmer: dict, with_balance: bool = True) -> dict:
    cutoff = (date.today() - timedelta(days=ALERT_WINDOW_DAYS)).isoformat()
    fields_out = []
    for f in fields_for_farmer(farmer):
        alarms = [a for a in db.get_water_alarms(water_field_id=f["id"], triage_status="confirmed")
                  if (a.get("obs_date") or "") >= cutoff]
        obs = db.get_water_observations(f["id"], limit=1)
        last = obs[-1] if obs else None
        item = {
            "id": f["id"], "label": f.get("parcel_id"), "crop_type": f.get("crop_type"),
            "irrigation_mode": f.get("irrigation_mode"),
            "alerts": [_public_alarm(a) for a in alarms],
            "last_obs": None if not last else {
                "date": last.get("obs_date"), "ndvi": last.get("ndvi_mean"), "ndmi": last.get("ndwi_mean")},
            "irrigations": [{"date": e["event_date"], "method": e.get("method"), "amount_mm": e.get("amount_mm"),
                             "source": e.get("source")}
                            for e in db.get_irrigation_events(f["id"])][-5:][::-1],
            "rain": [{"date": r["reading_date"], "mm": r["precip_mm"]}
                     for r in db.get_rain_readings(f["id"])][-5:][::-1],
            "soil_checks": [{"date": c["check_date"], "level": c.get("level"), "fill_pct": c["fill_pct"]}
                            for c in db.get_soil_checks(f["id"])][-3:][::-1],
            "salt": {k: (db.get_field_soil(f["id"]) or {}).get(k) for k in
                     ("farmer_salt_crust", "farmer_leaching", "farmer_leaching_m3", "farmer_report_at")},
            "balance": None, "balance_error": None, "balance_note": None,
        }
        if with_balance and f.get("geometry_wkt"):
            try:
                item["balance"] = _public_balance(field_balance_summary(f))
                import water_overview
                item["balance_note"] = water_overview.balance_trust(f)["reason"]
            except Exception as e:
                log.warning(f"farmer_view: bilans pola {f['id']}: {e}")
                item["balance_error"] = "unavailable"
        fields_out.append(item)
    import weather as weather_mod
    return {"farmer": {"name": farmer.get("name"), "lang": _lang(farmer)},
            "fields": fields_out, "attribution": weather_mod.OPEN_METEO_ATTRIBUTION,
            "generated_at": datetime.now().isoformat(timespec="minutes")}


# ---------------------------------------------------------------------------
# API — strona rolnika (dostęp tokenem z linku, bez klucza operatora)
# ---------------------------------------------------------------------------

def _farmer_or_404(token: str) -> dict:
    farmer = db.get_farmer_by_token(token)
    if not farmer:
        raise HTTPException(404, "Link jest nieaktualny — poproś doradcę o nowy.")
    return farmer


def _farmer_field_or_404(farmer: dict, water_field_id: int) -> dict:
    for f in fields_for_farmer(farmer):
        if f["id"] == water_field_id:
            return f
    raise HTTPException(404, "Nie ma takiego pola.")


def _check_recent_date(v: str) -> str:
    try:
        d = date.fromisoformat(v)
    except ValueError:
        raise ValueError("Data w formacie RRRR-MM-DD")
    if d > date.today():
        raise ValueError("Data nie może być z przyszłości")
    if d < date.today() - timedelta(days=FARMER_LOG_MAX_AGE_DAYS):
        raise ValueError(f"Można wpisać najwyżej {FARMER_LOG_MAX_AGE_DAYS} dni wstecz")
    return v


class FarmerIrrigationRequest(BaseModel):
    event_date: str
    method: str = Field(..., pattern="^(zalewowe|kroplowe|deszczowanie)$")
    amount_mm: Optional[float] = Field(None, gt=0, le=300)
    amount_m3_ha: Optional[float] = Field(None, gt=0, le=3000)
    notes: Optional[str] = Field(None, max_length=500)


class FarmerRainRequest(BaseModel):
    reading_date: str
    precip_mm: float = Field(..., ge=0, le=300)


class FarmerSaltRequest(BaseModel):
    salt_crust: Optional[str] = Field(None, pattern="^(none|spots|much|unknown)$")
    leaching: Optional[str] = Field(None, pattern="^(yes|no|unknown)$")
    leaching_m3_ha: Optional[float] = Field(None, gt=0, le=20000)


class FarmerSoilCheckRequest(BaseModel):
    check_date: str
    level: str = Field(..., pattern="^(wet|moist|medium|dry|very_dry)$")


class FarmerAckRequest(BaseModel):
    ack: str = Field(..., pattern="^(seen|will_irrigate|irrigated|disagree)$")
    note: Optional[str] = Field(None, max_length=500)


_FARMER_HTML_PATH = Path(__file__).with_name("static") / "farmer.html"


@router.get("/f/{token}", response_class=HTMLResponse, include_in_schema=False)
def farmer_page(token: str):
    # Strona ładuje się zawsze (ładny komunikat o nieaktualnym linku robi JS);
    # noindex, żeby linki z SMS-ów nie trafiały do wyszukiwarek.
    html = _FARMER_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache", "X-Robots-Tag": "noindex",
                                       "Referrer-Policy": "no-referrer"})


@router.get("/api/f/{token}")
def farmer_api_view(token: str):
    farmer = _farmer_or_404(token)
    db.touch_farmer_seen(farmer["id"])
    return farmer_view(farmer)


@router.post("/api/f/{token}/field/{water_field_id}/irrigation")
def farmer_api_irrigation(token: str, water_field_id: int, req: FarmerIrrigationRequest):
    farmer = _farmer_or_404(token)
    _farmer_field_or_404(farmer, water_field_id)
    try:
        _check_recent_date(req.event_date)
    except ValueError as e:
        raise HTTPException(422, str(e))
    amount_mm = req.amount_mm if req.amount_mm is not None else (
        req.amount_m3_ha / 10.0 if req.amount_m3_ha is not None else None)
    event_id = db.add_irrigation_event(water_field_id, req.event_date, req.method,
                                       amount_mm=amount_mm, source="farmer_panel", notes=req.notes)
    return {"id": event_id, "amount_mm": amount_mm}


@router.post("/api/f/{token}/field/{water_field_id}/rain")
def farmer_api_rain(token: str, water_field_id: int, req: FarmerRainRequest):
    farmer = _farmer_or_404(token)
    _farmer_field_or_404(farmer, water_field_id)
    try:
        _check_recent_date(req.reading_date)
    except ValueError as e:
        raise HTTPException(422, str(e))
    rid = db.add_rain_reading(water_field_id, req.reading_date, req.precip_mm, notes="panel rolnika")
    db.set_rain_reading_source(rid, "farmer_panel")
    return {"id": rid}


@router.post("/api/f/{token}/field/{water_field_id}/soil_check")
def farmer_api_soil_check(token: str, water_field_id: int, req: FarmerSoilCheckRequest):
    """Ocena wilgotności „gleba w dłoni” od rolnika — punkt kontrolny bilansu."""
    import water_balance
    farmer = _farmer_or_404(token)
    _farmer_field_or_404(farmer, water_field_id)
    try:
        _check_recent_date(req.check_date)
    except ValueError as e:
        raise HTTPException(422, str(e))
    fill = water_balance.SOIL_FEEL_LEVELS[req.level]
    cid = db.add_soil_check(water_field_id, req.check_date, fill, method="feel", level=req.level,
                            source="farmer_panel")
    return {"id": cid, "fill_pct": fill}


@router.post("/api/f/{token}/field/{water_field_id}/salt")
def farmer_api_salt(token: str, water_field_id: int, req: FarmerSaltRequest):
    """Rolnik: czy widać naloty soli i czy pole było płukane zimą. Trafia do
    doradcy jako wskazówka (soil_water.field_salinity -> hints) — nie zmienia
    bilansu, dopóki doradca nie wpisze klasy albo pomiaru EC."""
    from datetime import datetime
    farmer = _farmer_or_404(token)
    _farmer_field_or_404(farmer, water_field_id)
    ch = req.model_dump(exclude_unset=True)
    db.save_field_soil(water_field_id, farmer_salt_crust=ch.get("salt_crust"), farmer_leaching=ch.get("leaching"),
                       farmer_leaching_m3=ch.get("leaching_m3_ha") if ch.get("leaching") == "yes" else None,
                       farmer_report_at=datetime.now().isoformat(timespec="seconds"))
    return {"status": "saved"}


@router.post("/api/f/{token}/alarm/{alarm_id}/ack")
def farmer_api_ack(token: str, alarm_id: int, req: FarmerAckRequest):
    farmer = _farmer_or_404(token)
    alarm = db.get_water_alarm_by_id(alarm_id)
    if not alarm or alarm.get("triage_status") != "confirmed":
        raise HTTPException(404, "Nie ma takiego alertu.")
    _farmer_field_or_404(farmer, alarm["water_field_id"])
    db.set_water_alarm_farmer_ack(alarm_id, req.ack, req.note)
    return {"id": alarm_id, "farmer_ack": req.ack}


# ---------------------------------------------------------------------------
# API — operator (rolnicy, linki, skrzynka SMS)
# ---------------------------------------------------------------------------

class FarmerUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    lang: Optional[str] = Field(None, pattern="^(pl|en|ru|uz)$")
    sms_opt_in: Optional[bool] = None
    auto_suggestions: Optional[bool] = None


def _farmer_out(farmer: dict) -> dict:
    fields = fields_for_farmer(farmer)
    ids = {f["id"] for f in fields}
    acks = [a for a in db.get_water_alarms(triage_status="confirmed")
            if a["water_field_id"] in ids and a.get("farmer_ack")]
    acks.sort(key=lambda a: a.get("farmer_ack_at") or "", reverse=True)
    return {
        **farmer, "phone_display": display_phone(farmer["phone"]),
        "link": panel_link(farmer) if farmer.get("token") else None,
        "fields": [{"id": f["id"], "parcel_id": f["parcel_id"], "irrigation_mode": f["irrigation_mode"]}
                   for f in fields],
        "last_ack": None if not acks else {
            "alarm_id": acks[0]["id"], "ack": acks[0]["farmer_ack"], "note": acks[0].get("farmer_ack_note"),
            "at": acks[0].get("farmer_ack_at"), "parcel_id": next(
                (f["parcel_id"] for f in fields if f["id"] == acks[0]["water_field_id"]), None)},
    }


@router.get("/api/farmers")
def farmers_list():
    sync_farmers_from_fields()
    return [_farmer_out(f) for f in db.get_farmers()]


@router.get("/api/farmers/sms_status")
def farmers_sms_status():
    return {"provider": sms_provider(), "app_base_url": os.environ.get("APP_BASE_URL") or None,
            "max_per_day": int(os.environ.get("SMS_MAX_PER_FARMER_DAY", MAX_SMS_PER_FARMER_DAY)),
            "default_country": os.environ.get("SMS_DEFAULT_COUNTRY", DEFAULT_COUNTRY_CODE)}


@router.get("/api/farmers/outbox")
def farmers_outbox(limit: int = 30, farmer_id: Optional[int] = None):
    rows = db.get_sms_outbox(limit=max(1, min(limit, 200)), farmer_id=farmer_id)
    for r in rows:
        r["phone_display"] = display_phone(r["phone"])
        r["segments"] = sms_segments(r["text"])
    return rows


@router.patch("/api/farmers/{farmer_id}")
def farmers_update(farmer_id: int, req: FarmerUpdateRequest, _auth=Depends(require_operator_key)):
    if not db.get_farmer_by_id(farmer_id):
        raise HTTPException(404, "Nie ma takiego rolnika.")
    upd = {k: (int(v) if isinstance(v, bool) else v)
           for k, v in req.model_dump(exclude_unset=True).items() if v is not None}
    db.update_farmer(farmer_id, **upd)
    return _farmer_out(db.get_farmer_by_id(farmer_id))


@router.post("/api/farmers/{farmer_id}/new_link")
def farmers_new_link(farmer_id: int, _auth=Depends(require_operator_key)):
    if not db.get_farmer_by_id(farmer_id):
        raise HTTPException(404, "Nie ma takiego rolnika.")
    db.update_farmer(farmer_id, token=new_token())
    return _farmer_out(db.get_farmer_by_id(farmer_id))


@router.post("/api/farmers/{farmer_id}/send_link")
def farmers_send_link(farmer_id: int, _auth=Depends(require_operator_key)):
    try:
        return send_panel_link(farmer_id)
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.post("/api/farmers/alarm/{alarm_id}/notify")
def farmers_notify_alarm(alarm_id: int, _auth=Depends(require_operator_key)):
    try:
        return notify_confirmed_alarm(alarm_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.post("/api/farmers/field/{water_field_id}/suggest")
def farmers_send_suggestion(water_field_id: int, _auth=Depends(require_operator_key)):
    try:
        return send_irrigation_suggestion(water_field_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(502, f"Bilans wodny niedostępny: {e}")
