"""Testy zamknięcia pętli weryfikacji zdjęciem: po zakończeniu analizy AI
wszystkich zdjęć zgłoszonych dla alarmu, operator dostaje zbiorcze
podsumowanie mailem — bez automatycznego rozwiązywania alarmu (decyzja
zawsze należy do operatora, to narzędzie do weryfikacji dopłat)."""
from unittest.mock import MagicMock, patch

import photo_handler


def _photo(ai_verdict=None, ai_coverage_pct=None, ai_analysis=None):
    return {"ai_verdict": ai_verdict, "ai_coverage_pct": ai_coverage_pct, "ai_analysis": ai_analysis}


def test_aggregate_photo_verdict_wieksza_liczba_vegetation():
    photos = [
        _photo("vegetation_present", 70, "{}"),
        _photo("vegetation_present", 60, "{}"),
        _photo("bare_soil", 5, "{}"),
    ]
    summary = photo_handler._aggregate_photo_verdict(photos)
    assert summary["verdict"] == "vegetation_present"
    assert summary["n_analyzed"] == 3
    assert summary["n_total"] == 3
    assert round(summary["avg_coverage_pct"], 1) == round((70 + 60 + 5) / 3, 1)


def test_aggregate_photo_verdict_remis_daje_mixed():
    photos = [_photo("vegetation_present", 50, "{}"), _photo("bare_soil", 10, "{}")]
    summary = photo_handler._aggregate_photo_verdict(photos)
    assert summary["verdict"] == "mixed"


def test_aggregate_photo_verdict_brak_analizy():
    photos = [_photo(None, None, None), _photo(None, None, None)]
    summary = photo_handler._aggregate_photo_verdict(photos)
    assert summary["verdict"] is None
    assert summary["n_analyzed"] == 0


def test_notify_nie_wysyla_gdy_analiza_niekompletna(isolated_db):
    isolated_db.init_db()
    parcel_db_id = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = isolated_db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")
    isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake1.jpg")
    isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake2.jpg")
    # tylko jedno z dwóch zdjęć ma zapisaną analizę -> pętla NIE powinna się zamknąć

    photos = isolated_db.get_photos_by_alarm(alarm_id)
    isolated_db.update_photo_analysis(photos[0]["id"], ai_analysis="{}",
                                       ai_coverage_pct=50, ai_verdict="vegetation_present")

    with patch("notifier.send_photo_analysis_email") as mock_send:
        photo_handler._notify_photo_analysis_complete(alarm_id)
        mock_send.assert_not_called()


def test_notify_wysyla_zbiorcze_podsumowanie_gdy_wszystko_przeanalizowane(isolated_db):
    isolated_db.init_db()
    parcel_db_id = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = isolated_db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")
    isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake1.jpg")
    isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake2.jpg")

    photos = isolated_db.get_photos_by_alarm(alarm_id)
    for p in photos:
        isolated_db.update_photo_analysis(p["id"], ai_analysis="{}",
                                           ai_coverage_pct=65, ai_verdict="vegetation_present")

    with patch("notifier.send_photo_analysis_email") as mock_send:
        photo_handler._notify_photo_analysis_complete(alarm_id)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["alarm_id"] == alarm_id
        assert kwargs["parcel_id"] == "141201_1.0001.6509"
        assert kwargs["summary"]["verdict"] == "vegetation_present"
        assert kwargs["summary"]["n_analyzed"] == 2

    # alarm NIE zostaje automatycznie rozwiązany -- decyzja należy do operatora
    alarm = isolated_db.get_alarm_by_id(alarm_id)
    assert alarm["resolved"] == 0


def test_brakujacy_plik_nie_blokuje_petli_na_zawsze(isolated_db, monkeypatch):
    """Regresja: brakujący plik zdjęcia kiedyś zostawiał ai_analysis=NULL na
    zawsze, więc _notify_photo_analysis_complete nigdy nie uznawał alarmu za
    'w pełni przeanalizowany' — nawet dla kolejnych, poprawnych zdjęć."""
    isolated_db.init_db()
    parcel_db_id = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = isolated_db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")
    pid = isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/nieistniejacy/plik.jpg")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    # Nie tworzymy prawdziwego klienta Anthropic (środowisko testowe może nie
    # mieć działającej sieci/proxy) — dla tego testu nie ma to znaczenia, bo
    # brakujący plik jest odrzucany przed jakimkolwiek wywołaniem API.
    with patch("anthropic.Anthropic") as mock_client_cls, \
         patch("photo_handler._notify_photo_analysis_complete") as mock_notify:
        mock_client_cls.return_value = MagicMock()
        photo_handler._analyze_photos(alarm_id, [pid])
        mock_notify.assert_called_once_with(alarm_id)

    photos = isolated_db.get_photos_by_alarm(alarm_id)
    assert photos[0]["ai_analysis"] is not None
    assert photos[0]["ai_verdict"] == "uncertain"

    # kompletność jest teraz policzalna -- kolejne (poprawne) zdjęcie domyka pętlę
    with patch("notifier.send_photo_analysis_email") as mock_send:
        photo_handler._notify_photo_analysis_complete(alarm_id)
        mock_send.assert_called_once()


def test_claim_photo_notification_chroni_przed_podwojna_wysylka(isolated_db):
    """Regresja: dwie partie zdjęć dla tego samego alarmu, przetwarzane
    "równocześnie" (dwa wywołania), nie mogą obie wysłać maila."""
    isolated_db.init_db()
    parcel_db_id = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = isolated_db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")

    first = isolated_db.claim_photo_notification(alarm_id)
    second = isolated_db.claim_photo_notification(alarm_id)
    assert first is True
    assert second is False, "drugie 'zajęcie' nie powinno się udać dopóki pierwsze nie zostanie zwolnione"

    isolated_db.release_photo_notification_claim(alarm_id)
    third = isolated_db.claim_photo_notification(alarm_id)
    assert third is True, "po zwolnieniu kolejna próba powinna móc zająć ponownie"


def test_notify_zwalnia_claim_gdy_wysylka_sie_nie_udaje(isolated_db):
    """Jeśli send_photo_analysis_email zwróci False (np. SMTP nieskonfigurowany),
    claim musi zostać zwolniony, żeby kolejna partia zdjęć mogła spróbować."""
    isolated_db.init_db()
    parcel_db_id = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = isolated_db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")
    isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake1.jpg")
    photos = isolated_db.get_photos_by_alarm(alarm_id)
    isolated_db.update_photo_analysis(photos[0]["id"], ai_analysis="{}",
                                       ai_coverage_pct=10, ai_verdict="bare_soil")

    with patch("notifier.send_photo_analysis_email", return_value=False):
        photo_handler._notify_photo_analysis_complete(alarm_id)

    # claim powinien być zwolniony (nie "utknąć" na 1 na zawsze)
    assert isolated_db.claim_photo_notification(alarm_id) is True


def test_dwa_watki_zglaszajace_te_sama_alarme_wysylaja_tylko_jeden_mail(isolated_db):
    """Prawdziwy test współbieżności: dwa wątki jednocześnie wywołują
    _notify_photo_analysis_complete dla tego samego alarmu — mail ma
    wyjść dokładnie raz."""
    import threading

    isolated_db.init_db()
    parcel_db_id = isolated_db.add_monitored_parcel("141201_1.0001.6509", "scierniskowy", 2026)
    alarm_id = isolated_db.add_alarm(parcel_db_id, "no_vegetation", "2026-09-01", 0.1, "test")
    isolated_db.add_farmer_photo(alarm_id, parcel_db_id, "/tmp/fake1.jpg")
    photos = isolated_db.get_photos_by_alarm(alarm_id)
    isolated_db.update_photo_analysis(photos[0]["id"], ai_analysis="{}",
                                       ai_coverage_pct=70, ai_verdict="vegetation_present")

    with patch("notifier.send_photo_analysis_email", return_value=True) as mock_send:
        threads = [threading.Thread(target=photo_handler._notify_photo_analysis_complete,
                                     args=(alarm_id,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert mock_send.call_count == 1, f"oczekiwano dokładnie 1 wysyłki, było {mock_send.call_count}"


def test_send_photo_analysis_email_escapuje_i_wymaga_konfiguracji(monkeypatch):
    import notifier

    monkeypatch.delenv("OPERATOR_EMAIL", raising=False)
    assert notifier.send_photo_analysis_email(
        1, "X", "no_vegetation", "2026-09-01", {"verdict": "bare_soil", "avg_coverage_pct": 5,
                                                  "n_analyzed": 1, "n_total": 1},
    ) is False

    monkeypatch.setenv("OPERATOR_EMAIL", "op@example.com")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    with patch("notifier.smtplib.SMTP") as mock_smtp:
        instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = instance
        ok = notifier.send_photo_analysis_email(
            1, "<script>alert(1)</script>", "no_vegetation", "2026-09-01",
            {"verdict": "vegetation_present", "avg_coverage_pct": 72.3, "n_analyzed": 2, "n_total": 2},
        )
        assert ok is True
        import email as email_mod
        raw = instance.sendmail.call_args[0][2]
        parsed = email_mod.message_from_string(raw)
        html_part = next(p for p in parsed.walk() if p.get_content_type() == "text/html")
        body = html_part.get_payload(decode=True).decode(html_part.get_content_charset() or "utf-8")
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script" in body
