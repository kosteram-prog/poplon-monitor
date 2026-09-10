"""Formularz zdjęć od rolników + analiza przez Claude API."""

import base64
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import db

log = logging.getLogger("photo_handler")
router = APIRouter()

UPLOAD_DIR = Path("uploads/photos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FORM_HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Prześlij zdjęcia pola</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, Arial, sans-serif; background: #F2F4EF;
  min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 16px; }}
.card {{ background: #fff; border-radius: 12px; padding: 24px; max-width: 480px;
  width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
h1 {{ font-size: 20px; color: #1C2620; margin-bottom: 6px; }}
.sub {{ font-size: 13px; color: #5C6B60; margin-bottom: 20px; }}
.parcel-id {{ font-family: monospace; font-size: 13px; background: #F3F5F0;
  padding: 6px 10px; border-radius: 6px; margin-bottom: 20px; color: #1C2620; }}
label {{ display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }}
textarea {{ width: 100%; border: 1px solid #DCE1D6; border-radius: 8px;
  padding: 10px; font-size: 14px; min-height: 80px; margin-bottom: 16px; resize: vertical; }}
.file-input {{ width: 100%; border: 2px dashed #DCE1D6; border-radius: 8px;
  padding: 20px; text-align: center; margin-bottom: 20px; cursor: pointer;
  font-size: 14px; color: #5C6B60; }}
input[type=file] {{ display: none; }}
button {{ width: 100%; padding: 14px; background: #1E5631; color: #fff;
  border: 0; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; }}
button:active {{ background: #2F7A46; }}
.error {{ background: #FBECEA; color: #7C2A1D; padding: 12px; border-radius: 8px;
  margin-bottom: 16px; font-size: 13px; display: none; }}
</style>
</head>
<body>
<div class="card">
  <h1>📷 Zdjęcia pola</h1>
  <p class="sub">Proszę wykonać 1–3 zdjęcia pola i przesłać je poniżej.</p>
  <div class="parcel-id">Działka: {parcel_id}</div>
  <form method="post" enctype="multipart/form-data">
    <label for="comment">Komentarz (opcjonalnie):</label>
    <textarea id="comment" name="comment" placeholder="Np. poplon wygląda dobrze, widoczna zielona okrywa..."></textarea>
    <label>Zdjęcia (max 3):</label>
    <div class="file-input" onclick="document.getElementById('photos').click()">
      <span id="fileLabel">📁 Dotknij aby wybrać zdjęcia</span>
    </div>
    <input type="file" id="photos" name="photos" accept="image/*" multiple
      onchange="updateLabel(this)">
    <div class="error" id="err"></div>
    <button type="submit">Wyślij zdjęcia</button>
  </form>
</div>
<script>
function updateLabel(input) {{
  const n = input.files.length;
  if (n > 3) {{ input.value=''; document.getElementById('err').style.display='block';
    document.getElementById('err').textContent='Maksymalnie 3 zdjęcia.'; return; }}
  document.getElementById('fileLabel').textContent = n === 0 ? '📁 Dotknij aby wybrać zdjęcia'
    : `✅ Wybrano ${{n}} ${{n===1?'zdjęcie':'zdjęcia'}}`;
  document.getElementById('err').style.display='none';
}}
</script>
</body></html>"""

THANKS_HTML = """<!DOCTYPE html>
<html lang="pl">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dziękujemy</title>
<style>
body {{ font-family:-apple-system,Arial,sans-serif; background:#F2F4EF;
  min-height:100vh; display:flex; align-items:center; justify-content:center; padding:16px; }}
.card {{ background:#fff; border-radius:12px; padding:32px 24px; max-width:400px;
  width:100%; text-align:center; box-shadow:0 2px 12px rgba(0,0,0,.08); }}
.icon {{ font-size:48px; margin-bottom:16px; }}
h1 {{ font-size:22px; color:#1E5631; margin-bottom:10px; }}
p {{ color:#5C6B60; font-size:14px; }}
</style></head>
<body><div class="card">
<div class="icon">✅</div>
<h1>Dziękujemy!</h1>
<p>Zdjęcia zostały przesłane i zostaną przeanalizowane przez nasz system.<br><br>
Nie musisz robić nic więcej.</p>
</div></body></html>"""

ERROR_HTML = """<!DOCTYPE html>
<html lang="pl"><head><meta charset="UTF-8"><title>Błąd</title>
<style>body{{font-family:Arial;background:#F2F4EF;display:flex;align-items:center;
justify-content:center;min-height:100vh}}
.card{{background:#fff;border-radius:12px;padding:32px;max-width:400px;text-align:center}}</style>
</head><body><div class="card"><h1>⚠️ {message}</h1></div></body></html>"""


@router.get("/photos/{alarm_id}", response_class=HTMLResponse)
def photo_form(alarm_id: int):
    alarms = db.get_alarms()
    alarm = next((a for a in alarms if a["id"] == alarm_id), None)
    if not alarm:
        return HTMLResponse(ERROR_HTML.format(message="Link wygasł lub jest nieprawidłowy."), status_code=404)
    parcel = db.get_parcel_by_id(alarm["parcel_db_id"])
    parcel_id = parcel["parcel_id"] if parcel else str(alarm["parcel_db_id"])
    return HTMLResponse(FORM_HTML.format(parcel_id=parcel_id))


@router.post("/photos/{alarm_id}", response_class=HTMLResponse)
async def photo_upload(alarm_id: int, comment: str = Form(""),
                       photos: list[UploadFile] = File(...)):
    alarms = db.get_alarms()
    alarm = next((a for a in alarms if a["id"] == alarm_id), None)
    if not alarm:
        return HTMLResponse(ERROR_HTML.format(message="Nieznany alarm."), status_code=404)

    parcel_db_id = alarm["parcel_db_id"]
    photos = photos[:3]  # max 3

    album_dir = UPLOAD_DIR / str(alarm_id)
    album_dir.mkdir(parents=True, exist_ok=True)

    ndvi_at_alarm = alarm.get("ndvi_value")
    # Snapshot pozostałych wskaźników z momentu alarmu (jeśli dostępny —
    # patrz scheduler._add_alarm_with_indicators) — używany przez
    # calibration.py do sugerowania korekt wag Confidence Score.
    evi_at_alarm = alarm.get("evi_value")
    savi_at_alarm = alarm.get("savi_value")
    ndre_at_alarm = alarm.get("ndre_value")
    ndwi_at_alarm = alarm.get("ndwi_value")
    rvi_at_alarm = alarm.get("rvi_value")
    reip_at_alarm = alarm.get("reip_value")
    photo_ids = []
    for photo in photos:
        if not photo.filename:
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in photo.filename if c.isalnum() or c in "._-")
        path = album_dir / f"{ts}_{safe_name}"
        content = await photo.read()
        path.write_bytes(content)
        pid = db.add_farmer_photo(alarm_id, parcel_db_id, str(path),
                                  farmer_comment=comment or None,
                                  ndvi_at_alarm=ndvi_at_alarm,
                                  evi_at_alarm=evi_at_alarm, savi_at_alarm=savi_at_alarm,
                                  ndre_at_alarm=ndre_at_alarm, ndwi_at_alarm=ndwi_at_alarm,
                                  rvi_at_alarm=rvi_at_alarm, reip_at_alarm=reip_at_alarm)
        photo_ids.append(pid)

    # analiza AI w tle (nie blokuje odpowiedzi)
    if photo_ids:
        t = threading.Thread(target=_analyze_photos, args=(alarm_id, photo_ids), daemon=True)
        t.start()

    return HTMLResponse(THANKS_HTML)


def _aggregate_photo_verdict(photos: list) -> dict:
    """Łączy wyniki AI ze wszystkich zdjęć zgłoszonych dla jednego alarmu w
    jedno podsumowanie (większościowo: vegetation_present vs bare_soil)."""
    analyzed = [p for p in photos if p.get("ai_verdict") and p["ai_verdict"] != "uncertain"]
    coverage_vals = [p["ai_coverage_pct"] for p in analyzed if p.get("ai_coverage_pct") is not None]
    avg_coverage = sum(coverage_vals) / len(coverage_vals) if coverage_vals else None
    veg = sum(1 for p in analyzed if p["ai_verdict"] == "vegetation_present")
    bare = sum(1 for p in analyzed if p["ai_verdict"] == "bare_soil")
    if veg > bare:
        verdict = "vegetation_present"
    elif bare > veg:
        verdict = "bare_soil"
    elif analyzed:
        verdict = "mixed"
    else:
        verdict = None
    return {
        "verdict": verdict,
        "avg_coverage_pct": avg_coverage,
        "n_analyzed": len(analyzed),
        "n_total": len(photos),
    }


def _notify_photo_analysis_complete(alarm_id: int):
    """Zamyka pętlę weryfikacji zdjęciem: gdy wszystkie zdjęcia zgłoszone dla
    danego alarmu mają już próbę analizy AI (sukces lub błąd), operator
    dostaje zbiorcze podsumowanie mailem od razu — zamiast dowiadywać się o
    wyniku dopiero przy okazji przeglądu kalibracji progów. Nie rozwiązuje
    alarmu automatycznie (patrz notifier.send_photo_analysis_email).

    Każde wgranie zdjęć odpala tę funkcję z osobnego wątku. Bez
    `claim_photo_notification` dwie partie zdjęć wysłane blisko siebie w
    czasie mogłyby obie "zobaczyć" komplet i obie wysłać mail — claim jest
    atomowym UPDATE-em w SQLite, więc tylko jeden wątek wygrywa."""
    claimed = False
    try:
        all_photos = db.get_photos_by_alarm(alarm_id)
        if not all_photos or not all(p.get("ai_analysis") is not None for p in all_photos):
            return  # jeszcze trwa analiza innych zdjęć z tego alarmu

        if not db.claim_photo_notification(alarm_id):
            return  # inny wątek już wysyła (albo wysłał) podsumowanie dla tego alarmu
        claimed = True

        alarm = db.get_alarm_by_id(alarm_id)
        if not alarm:
            return
        parcel = db.get_parcel_by_id(alarm["parcel_db_id"])
        if not parcel:
            return

        summary = _aggregate_photo_verdict(all_photos)
        from notifier import send_photo_analysis_email
        ok = send_photo_analysis_email(
            alarm_id=alarm_id,
            parcel_id=parcel["parcel_id"],
            alarm_type=alarm["alarm_type"],
            alarm_date=alarm["alarm_date"],
            summary=summary,
        )
        if not ok:
            # wysyłka się nie powiodła (np. SMTP down) — odblokuj, żeby
            # kolejna partia zdjęć/ponowna próba mogła spróbować jeszcze raz
            db.release_photo_notification_claim(alarm_id)
            claimed = False
    except Exception as e:
        log.error(f"Nie udało się wysłać podsumowania analizy zdjęć (alarm {alarm_id}): {e}")
        if claimed:
            db.release_photo_notification_claim(alarm_id)


def _analyze_photos(alarm_id: int, photo_ids: list):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("Brak ANTHROPIC_API_KEY — analiza AI pominięta")
        return

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        log.error("Biblioteka anthropic nie jest zainstalowana")
        return

    photos_in_db = db.get_photos_by_alarm(alarm_id)
    photos_map = {p["id"]: p for p in photos_in_db}

    for pid in photo_ids:
        p = photos_map.get(pid)
        if not p:
            continue
        path = Path(p["photo_path"])
        if not path.exists():
            log.warning(f"Plik zdjęcia nie istnieje: {path}")
            # Musimy zapisać próbę analizy (nawet nieudaną), bo inaczej
            # ai_analysis zostaje NULL na zawsze i _notify_photo_analysis_complete
            # nigdy nie uzna tego alarmu za "w pełni przeanalizowany" — operator
            # nie dostałby podsumowania nawet dla kolejnych, poprawnych zdjęć.
            db.update_photo_analysis(pid, ai_analysis='{"error":"file_missing"}',
                                     ai_coverage_pct=None, ai_verdict="uncertain")
            continue
        try:
            img_data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
            suffix = path.suffix.lower()
            media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                          "png": "image/png", "heic": "image/heic",
                          "webp": "image/webp"}.get(suffix.lstrip("."), "image/jpeg")

            msg = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=512,
                timeout=60.0,
                system="Jesteś ekspertem od oceny stanu upraw rolnych na zdjęciach. Odpowiadaj TYLKO w formacie JSON.",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                         "media_type": media_type, "data": img_data}},
                        {"type": "text", "text": (
                            "Oceń to zdjęcie pola rolnego. Zwróć JSON z polami:\n"
                            "{\n"
                            '  "vegetation_present": true/false,\n'
                            '  "coverage_pct": 0-100,\n'
                            '  "verdict": "vegetation_present"|"bare_soil"|"uncertain",\n'
                            '  "confidence": "high"|"medium"|"low",\n'
                            '  "notes": "krótki opis po polsku (max 100 znaków)"\n'
                            "}\n"
                            "vegetation_present=true jeśli widzisz rośliny pokrywające min. 20% powierzchni.\n"
                            "coverage_pct to szacowany procent pokrycia roślinnością."
                        )}
                    ]
                }]
            )
            raw = msg.content[0].text.strip()
            # usun ewentualne backticki
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            db.update_photo_analysis(
                pid,
                ai_analysis=json.dumps(data, ensure_ascii=False),
                ai_coverage_pct=float(data.get("coverage_pct", 0)),
                ai_verdict=data.get("verdict", "uncertain"),
            )
            log.info(f"Analiza AI zdjecia {pid}: verdict={data.get('verdict')}, "
                     f"coverage={data.get('coverage_pct')}%")
        except json.JSONDecodeError as e:
            log.error(f"Błąd parsowania JSON z Claude dla zdjecia {pid}: {e}")
            db.update_photo_analysis(pid, ai_analysis='{"error":"json_parse"}',
                                     ai_coverage_pct=None, ai_verdict="uncertain")
        except Exception as e:
            log.error(f"Błąd analizy AI zdjecia {pid}: {e}")
            db.update_photo_analysis(pid, ai_analysis=f'{{"error":"{type(e).__name__}"}}',
                                     ai_coverage_pct=None, ai_verdict="uncertain")

    # Zamknij pętlę: powiadom operatora zbiorczym wynikiem, jeśli to były
    # ostatnie brakujące zdjęcia dla tego alarmu.
    _notify_photo_analysis_complete(alarm_id)
