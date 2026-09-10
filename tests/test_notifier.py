"""Testy notifier.py: escapowanie HTML w mailu do operatora (ochrona przed
wstrzyknięciem znaczników przez dane pochodzące pośrednio od użytkownika)."""
import email
from unittest.mock import MagicMock, patch

import notifier


def test_send_operator_email_escapuje_html(monkeypatch):
    monkeypatch.setenv("OPERATOR_EMAIL", "op@example.com")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")

    with patch("notifier.smtplib.SMTP") as mock_smtp:
        instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = instance

        ok = notifier.send_operator_email(
            alarm_id=1,
            parcel_id="<img src=x onerror=alert(1)>141201_1",
            alarm_type="no_vegetation",
            alarm_date="2026-09-01",
            ndvi_value=0.12,
            message="<script>alert(1)</script> szczegóły",
        )
        assert ok is True

        raw = instance.sendmail.call_args[0][2]
        parsed = email.message_from_string(raw)
        html_part = None
        for part in parsed.walk():
            if part.get_content_type() == "text/html":
                html_part = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8")
                break

        assert html_part is not None
        assert "<img src=x" not in html_part
        assert "<script>" not in html_part
        assert "&lt;img" in html_part
        assert "&lt;script" in html_part


def test_send_operator_email_brak_konfiguracji_zwraca_false(monkeypatch):
    monkeypatch.delenv("OPERATOR_EMAIL", raising=False)
    monkeypatch.delenv("SMTP_USER", raising=False)
    ok = notifier.send_operator_email(
        alarm_id=1, parcel_id="X", alarm_type="no_vegetation",
        alarm_date="2026-09-01", ndvi_value=0.1, message="test",
    )
    assert ok is False
