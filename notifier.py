"""Powiadomienia: email do operatora i SMS do rolnika."""

import html
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

import db

log = logging.getLogger("notifier")

ALARM_DESCRIPTIONS = {
    "no_vegetation": "Brak wegetacji — NDVI na poziomie gołej gleby przez ostatnie pomiary. "
                     "Poplon może nie być wysiany lub wschody nie powiodły się.",
    "sudden_drop":   "Nagły spadek NDVI — możliwa przedwczesna likwidacja poplonu, "
                     "zniszczenie przez przymrozki lub chorobę.",
    "below_threshold": "NDVI nigdy nie osiągnęło wymaganego poziomu — poplon prawdopodobnie "
                       "nie wykształcił wystarczającej okrywy roślinnej.",
}


def send_operator_email(alarm_id, parcel_id, alarm_type, alarm_date, ndvi_value, message):
    operator_email = os.environ.get("OPERATOR_EMAIL")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    app_url = os.environ.get("APP_BASE_URL", "")

    if not operator_email or not smtp_user:
        log.warning("Brak OPERATOR_EMAIL lub SMTP_USER — email nie wysłany")
        return False

    try:
        desc = ALARM_DESCRIPTIONS.get(alarm_type, alarm_type)
        # parcel_id/alarm_type/message pochodzą pośrednio od danych
        # wejściowych operatora/rolnika — escapujemy przed wstawieniem do
        # HTML maila, żeby wykluczyć wstrzyknięcie znaczników.
        safe_parcel_id = html.escape(str(parcel_id))
        safe_alarm_type = html.escape(str(alarm_type))
        safe_alarm_date = html.escape(str(alarm_date))
        safe_message = html.escape(str(message))
        safe_desc = html.escape(str(desc))
        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;max-width:600px">
        <h2 style="color:#B23B2A">⚠ Alarm NDVI — {safe_parcel_id}</h2>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="padding:6px;font-weight:bold">Działka</td><td>{safe_parcel_id}</td></tr>
          <tr><td style="padding:6px;font-weight:bold">Typ alarmu</td><td>{safe_alarm_type}</td></tr>
          <tr><td style="padding:6px;font-weight:bold">Data</td><td>{safe_alarm_date}</td></tr>
          <tr><td style="padding:6px;font-weight:bold">NDVI</td><td>{ndvi_value:.3f}</td></tr>
        </table>
        <p style="margin-top:16px"><b>Opis:</b> {safe_desc}</p>
        <p><b>Szczegóły:</b> {safe_message}</p>
        {f'<p><a href="{app_url}/#monitor">Przejdź do panelu monitoringu →</a></p>' if app_url else ''}
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ALARM NDVI] {parcel_id} — {alarm_type} ({alarm_date})"
        msg["From"] = smtp_user
        msg["To"] = operator_email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            s.starttls()
            s.login(smtp_user, smtp_password)
            s.sendmail(smtp_user, operator_email, msg.as_string())

        log.info(f"Email wysłany do operatora: alarm {alarm_id}")
        return True
    except Exception as e:
        log.error(f"Błąd wysyłania email: {e}")
        return False


PHOTO_VERDICT_LABELS = {
    "vegetation_present": "Zdjęcia sugerują OBECNOŚĆ poplonu",
    "bare_soil":          "Zdjęcia sugerują BRAK poplonu (goła gleba)",
    "mixed":               "Zdjęcia dają niejednoznaczny wynik",
}


def send_photo_analysis_email(alarm_id, parcel_id, alarm_type, alarm_date, summary):
    """Zamyka pętlę weryfikacji zdjęciem: gdy AI skończy analizować wszystkie
    zdjęcia zgłoszone przez rolnika dla danego alarmu, operator dostaje
    zbiorcze podsumowanie od razu — zamiast dowiadywać się o wyniku dopiero
    przy okazji przeglądu kalibracji progów. Nie rozwiązujemy alarmu
    automatycznie: to narzędzie do weryfikacji dopłat, więc ostateczna
    decyzja zawsze należy do operatora."""
    operator_email = os.environ.get("OPERATOR_EMAIL")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    app_url = os.environ.get("APP_BASE_URL", "")

    if not operator_email or not smtp_user:
        log.warning("Brak OPERATOR_EMAIL lub SMTP_USER — email o analizie zdjęć nie wysłany")
        return False

    try:
        verdict = summary.get("verdict")
        label = PHOTO_VERDICT_LABELS.get(verdict, "Analiza AI niejednoznaczna lub nieudana")
        coverage = summary.get("avg_coverage_pct")
        coverage_txt = f"{coverage:.0f}%" if coverage is not None else "brak danych"

        safe_parcel_id = html.escape(str(parcel_id))
        safe_alarm_type = html.escape(str(alarm_type))
        safe_alarm_date = html.escape(str(alarm_date))
        safe_label = html.escape(label)

        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;max-width:600px">
        <h2 style="color:#1E5631">📷 Zdjęcia od rolnika przeanalizowane — {safe_parcel_id}</h2>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="padding:6px;font-weight:bold">Działka</td><td>{safe_parcel_id}</td></tr>
          <tr><td style="padding:6px;font-weight:bold">Powiązany alarm</td><td>{safe_alarm_type} ({safe_alarm_date})</td></tr>
          <tr><td style="padding:6px;font-weight:bold">Wynik AI</td><td>{safe_label}</td></tr>
          <tr><td style="padding:6px;font-weight:bold">Śr. pokrycie roślinnością</td><td>{coverage_txt}</td></tr>
          <tr><td style="padding:6px;font-weight:bold">Przeanalizowane zdjęcia</td><td>{summary.get('n_analyzed', 0)}/{summary.get('n_total', 0)}</td></tr>
        </table>
        <p style="margin-top:16px;color:#7C6A2A"><b>To ocena AI — wymaga potwierdzenia operatora przed rozwiązaniem alarmu.</b></p>
        {f'<p><a href="{app_url}/#monitor">Przejdź do panelu monitoringu, żeby zweryfikować i rozwiązać alarm →</a></p>' if app_url else ''}
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Zdjęcia] {parcel_id} — {label}"
        msg["From"] = smtp_user
        msg["To"] = operator_email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            s.starttls()
            s.login(smtp_user, smtp_password)
            s.sendmail(smtp_user, operator_email, msg.as_string())

        log.info(f"Email o analizie zdjęć wysłany do operatora: alarm {alarm_id}")
        return True
    except Exception as e:
        log.error(f"Błąd wysyłania email o analizie zdjęć: {e}")
        return False


def send_farmer_sms(alarm_id, parcel_id, farmer_phone, farmer_name=None):
    token = os.environ.get("SMSAPI_TOKEN")
    app_url = os.environ.get("APP_BASE_URL", "")

    if not token:
        log.warning("Brak SMSAPI_TOKEN — SMS nie wysłany")
        return False

    try:
        name_part = f" {farmer_name}" if farmer_name else ""
        photo_link = f"{app_url}/photos/{alarm_id}" if app_url else f"/photos/{alarm_id}"
        short_parcel = parcel_id[-15:] if len(parcel_id) > 15 else parcel_id

        text = (f"Dzien dobry{name_part}. System wykryl anomalie na dzialce "
                f"{short_parcel}. Prosimy o zdjecia pola: {photo_link}")

        if len(text) > 160:
            short_parcel = parcel_id[-10:]
            text = (f"Dzien dobry. Anomalia na dzialce {short_parcel}. "
                    f"Zdjecia: {photo_link}")

        r = requests.post(
            "https://ssl.smsapi.pl/sms.do",
            headers={"Authorization": f"Bearer {token}"},
            data={"to": farmer_phone, "message": text,
                  "format": "json", "from": "PopMonitor"},
            timeout=15,
        )
        data = r.json()
        if r.status_code == 200 and data.get("count", 0) > 0:
            log.info(f"SMS wysłany do rolnika {farmer_phone}: alarm {alarm_id}")
            return True
        else:
            log.error(f"Błąd SMSAPI: {data}")
            return False
    except Exception as e:
        log.error(f"Błąd wysyłania SMS: {e}")
        return False


def notify_alarm(alarm_id, parcel_db_id):
    parcel = db.get_parcel_by_id(parcel_db_id)
    if not parcel:
        log.error(f"notify_alarm: nie znaleziono działki id={parcel_db_id}")
        return

    alarms = db.get_alarms(parcel_db_id)
    alarm = next((a for a in alarms if a["id"] == alarm_id), None)
    if not alarm:
        log.error(f"notify_alarm: nie znaleziono alarmu id={alarm_id}")
        return

    email_ok = send_operator_email(
        alarm_id=alarm_id,
        parcel_id=parcel["parcel_id"],
        alarm_type=alarm["alarm_type"],
        alarm_date=alarm["alarm_date"],
        ndvi_value=alarm["ndvi_value"] or 0.0,
        message=alarm["message"] or "",
    )

    sms_ok = False
    if parcel.get("farmer_phone"):
        sms_ok = send_farmer_sms(
            alarm_id=alarm_id,
            parcel_id=parcel["parcel_id"],
            farmer_phone=parcel["farmer_phone"],
            farmer_name=parcel.get("farmer_name"),
        )

    db.mark_alarm_notified(alarm_id, operator=email_ok, farmer=sms_ok)
