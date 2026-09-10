"""Lekka autoryzacja dla mutujących/kosztownych endpointów.

Ten serwis jest wewnętrznym narzędziem operatorskim bez systemu kont
użytkowników. Zamiast budować pełny system logowania, chronimy endpointy,
które zmieniają stan lub generują koszty (kalibracja progów, dodawanie/
usuwanie działek z monitoringu, wysyłka SMS, wymuszone sprawdzenie) prostym
współdzielonym sekretem przekazywanym w nagłówku `X-Operator-Key`.

Jeśli zmienna środowiskowa OPERATOR_API_KEY nie jest ustawiona, dependency
nie blokuje ruchu (żeby nie psuć lokalnego/deweloperskiego uruchomienia bez
dodatkowej konfiguracji) — ale ostrzega o tym raz w logach przy starcie.
"""
import logging
import os

from fastapi import Header, HTTPException

log = logging.getLogger("auth")

_warned = False


def require_operator_key(x_operator_key: str | None = Header(default=None)):
    global _warned
    expected = os.environ.get("OPERATOR_API_KEY")
    if not expected:
        if not _warned:
            log.warning(
                "OPERATOR_API_KEY nie jest ustawiony — endpointy operatorskie "
                "są otwarte bez autoryzacji. Ustaw OPERATOR_API_KEY w .env, "
                "żeby to wyłączyć."
            )
            _warned = True
        return True
    if x_operator_key != expected:
        raise HTTPException(status_code=401, detail="Nieprawidłowy lub brakujący klucz operatora.")
    return True
