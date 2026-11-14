"""Panel rolnika i SMS (farmer_panel.py)."""
import json
from datetime import date, timedelta

import pytest

import farmer_panel as fp

WKT = "POLYGON((69.0 41.0, 69.001 41.0, 69.001 41.001, 69.0 41.001, 69.0 41.0))"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("SMS_PROVIDER", "SMSAPI_TOKEN", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM",
              "SMS_DEFAULT_COUNTRY", "SMS_MAX_PER_FARMER_DAY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APP_BASE_URL", "https://pm.example.org")
    fp.clear_balance_cache()
    yield
    fp.clear_balance_cache()


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append((url, kw))
        if "twilio" in url:
            return FakeResp(201, {"sid": "SM1", "status": "queued"})
        return FakeResp(200, {"count": 1, "list": [{"id": "x"}]})
    monkeypatch.setattr(fp.requests, "post", fake_post)
    return calls


def _field(db, phone="+998 90 123 45 67", mode="irrigation", name="Akmal", pid="UZ-Bekobod-12"):
    return db.add_water_field(pid, mode, WKT, farmer_phone=phone, farmer_name=name)


def _summary(**kw):
    s = {"status": "approaching", "as_of": date.today().isoformat(), "depletion_mm": 40.0,
         "depletion_frac": 0.4, "raw_mm": 50.0, "taw_mm": 100.0, "days_to_stress": 1,
         "suggested_irrigation_date": (date.today() + timedelta(days=1)).isoformat(),
         "suggested_dose_mm": 52.0, "suggested_dose_m3_ha": 520.0, "initial_state_reliable": True,
         "forecast": {"days": 7, "precip_mm": 0.0, "et0_mm": 45.0, "tmax_c": 36.0, "dry_hot": True}}
    s.update(kw)
    return s


# --- numery -----------------------------------------------------------------

@pytest.mark.parametrize("raw,exp", [
    ("600 100 200", "48600100200"), ("+48 600-100-200", "48600100200"),
    ("0048600100200", "48600100200"), ("+998 90 123 45 67", "998901234567"),
    ("12345", None), ("", None), (None, None)])
def test_normalize_phone(raw, exp):
    assert fp.normalize_phone(raw) == exp


def test_normalize_phone_kraj_domyslny(monkeypatch):
    monkeypatch.setenv("SMS_DEFAULT_COUNTRY", "998")
    assert fp.normalize_phone("901234567") == "998901234567"


def test_sync_tworzy_rolnika_z_jezykiem_i_tokenem(isolated_db):
    isolated_db.init_db()
    _field(isolated_db)
    _field(isolated_db, pid="UZ-2")                       # ten sam rolnik, drugie pole
    _field(isolated_db, phone="600100200", name="Jan", pid="PL-1")
    assert fp.sync_farmers_from_fields() == 2
    assert fp.sync_farmers_from_fields() == 0
    uz = isolated_db.get_farmer_by_phone("998901234567")
    assert uz["lang"] == "uz" and len(uz["token"]) >= 20 and uz["name"] == "Akmal"
    assert len(fp.fields_for_farmer(uz)) == 2
    assert isolated_db.get_farmer_by_phone("48600100200")["lang"] == "pl"


# --- teksty -----------------------------------------------------------------

def test_teksty_sms_wszystkie_jezyki_maja_klucze_i_link(isolated_db):
    isolated_db.init_db()
    fid = _field(isolated_db)
    fp.sync_farmers_from_fields()
    farmer = isolated_db.get_farmer_by_phone("998901234567")
    field = isolated_db.get_water_field_by_id(fid)
    keys = set(fp.SMS_TEXTS["pl"])
    for lang, tx in fp.SMS_TEXTS.items():
        assert set(tx) == keys, lang
        farmer["lang"] = lang
        txt = fp.suggestion_sms_text(_summary(), field, farmer)
        assert farmer["token"] in txt and "52" in txt and "520" in txt
        assert fp.sms_segments(txt) <= 3
        assert fp.sms_segments(fp.link_sms_text(farmer)) <= 2


def test_teksty_pl_en_uz_sa_gsm7(isolated_db):
    for lang in ("pl", "en", "uz"):
        for k, v in fp.SMS_TEXTS[lang].items():
            assert all(ch in fp.GSM7 or ch in fp.GSM7_EXT for ch in v), (lang, k)


def test_sms_segments():
    assert fp.sms_segments("a" * 160) == 1
    assert fp.sms_segments("a" * 161) == 2
    assert fp.sms_segments("ж" * 70) == 1
    assert fp.sms_segments("ж" * 71) == 2


def test_alert_text_wg_typu(isolated_db):
    isolated_db.init_db()
    fid = _field(isolated_db)
    fp.sync_farmers_from_fields()
    farmer = dict(isolated_db.get_farmer_by_phone("998901234567"), lang="pl")
    field = isolated_db.get_water_field_by_id(fid)
    bal = {"water_balance": _summary()}
    t = fp.alert_sms_text({"risk_type": "water_balance_deficit", "components_json": json.dumps(bal)},
                          field, farmer)
    assert "52 mm" in t and "jutro" in t
    t2 = fp.alert_sms_text({"risk_type": "irrigation_uniformity_anomaly"}, field, farmer)
    assert "rownomiernosc" in t2


# --- wysyłka ----------------------------------------------------------------

def _farmer(db, **kw):
    fid = _field(db)
    fp.sync_farmers_from_fields()
    f = db.get_farmer_by_phone("998901234567")
    if kw:
        db.update_farmer(f["id"], **kw)
        f = db.get_farmer_by_id(f["id"])
    return f, fid


def test_bez_bramki_trafia_do_skrzynki(isolated_db, sent):
    isolated_db.init_db()
    farmer, _ = _farmer(isolated_db)
    res = fp.send_sms(farmer, "test", "panel_link")
    assert res["status"] == "not_configured" and not sent
    assert isolated_db.get_sms_outbox()[0]["text"] == "test"


def test_smsapi_i_limit_dzienny(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    monkeypatch.setenv("SMS_MAX_PER_FARMER_DAY", "2")
    farmer, _ = _farmer(isolated_db)
    assert [fp.send_sms(farmer, "x", "k")["status"] for _ in range(3)] == ["sent", "sent", "rate_limited"]
    assert len(sent) == 2 and sent[0][1]["data"]["to"] == "998901234567"
    assert sent[0][1]["data"]["encoding"] == "utf-8"


def test_twilio(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    for k, v in (("TWILIO_ACCOUNT_SID", "AC1"), ("TWILIO_AUTH_TOKEN", "t"), ("TWILIO_FROM", "+100")):
        monkeypatch.setenv(k, v)
    farmer, _ = _farmer(isolated_db)
    assert fp.send_sms(farmer, "x", "k")["status"] == "sent"
    assert "AC1" in sent[0][0] and sent[0][1]["data"]["To"] == "+998901234567"


def test_wypisany_nie_dostaje(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    farmer, _ = _farmer(isolated_db, sms_opt_in=0)
    assert fp.send_sms(farmer, "x", "k")["status"] == "opted_out" and not sent


def test_blad_bramki_nie_rzuca(isolated_db, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")

    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(fp.requests, "post", boom)
    farmer, _ = _farmer(isolated_db)
    assert fp.send_sms(farmer, "x", "k")["status"] == "failed"


# --- potwierdzony alert -> SMS ---------------------------------------------

def test_sms_tylko_dla_potwierdzonego(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    _, fid = _farmer(isolated_db)
    aid = isolated_db.add_water_alarm(fid, date.today().isoformat(), "water_stress_risk", "m", "nk")
    with pytest.raises(ValueError):
        fp.notify_confirmed_alarm(aid)
    isolated_db.set_water_alarm_triage(aid, "confirmed")
    res = fp.notify_confirmed_alarm(aid)
    assert res["status"] == "sent"
    a = isolated_db.get_water_alarm_by_id(aid)
    assert a["notified_farmer"] == 1 and a["farmer_notified_at"]


def test_triage_api_z_sms(app_client, isolated_db, sent, monkeypatch):
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    _, fid = _farmer(isolated_db)
    aid = isolated_db.add_water_alarm(fid, date.today().isoformat(), "water_stress_risk", "m", "nk")
    r = app_client.post(f"/api/water/alarm/{aid}/triage",
                        json={"triage_status": "confirmed", "notify_farmer": True})
    assert r.status_code == 200 and r.json()["sms"]["status"] == "sent"
    aid2 = isolated_db.add_water_alarm(fid, date.today().isoformat(), "uneven_moisture_risk", "m", "nk")
    r = app_client.post(f"/api/water/alarm/{aid2}/triage", json={"triage_status": "rejected",
                                                                 "notify_farmer": True})
    assert "sms" not in r.json() and len(sent) == 1


def test_pole_bez_telefonu(isolated_db):
    isolated_db.init_db()
    fid = isolated_db.add_water_field("X", "irrigation", WKT)
    aid = isolated_db.add_water_alarm(fid, "2026-06-01", "water_stress_risk", "m", "nk")
    isolated_db.set_water_alarm_triage(aid, "confirmed")
    assert fp.notify_confirmed_alarm(aid)["status"] == "no_phone"


# --- automatyczne sugestie ---------------------------------------------------

def test_auto_sugestie(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    farmer, fid = _farmer(isolated_db)
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    assert fp.auto_suggestions_job() == []                # opcja wyłączona
    isolated_db.update_farmer(farmer["id"], auto_suggestions=1)
    assert fp.auto_suggestions_job() == []                # brak dziennika nawodnień — bilans niewiarygodny
    isolated_db.add_irrigation_event(fid, (date.today() - timedelta(days=20)).isoformat(), "kroplowe",
                                     amount_mm=40)
    res = fp.auto_suggestions_job()
    assert [r["status"] for r in res] == ["sent"]
    assert fp.auto_suggestions_job() == []                # nie ponawia w ciągu 3 dni


def test_auto_sugestie_wstrzymane_po_nawodnieniu_i_bez_potrzeby(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    farmer, fid = _farmer(isolated_db, auto_suggestions=1)
    monkeypatch.setattr(fp, "field_balance_summary",
                        lambda f, use_cache=True: _summary(days_to_stress=6, status="ok"))
    assert fp.auto_suggestions_job() == []
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    isolated_db.add_irrigation_event(fid, date.today().isoformat(), "kroplowe", amount_mm=30)
    assert fp.auto_suggestions_job() == []


def test_auto_sugestie_pomija_pola_deszczowe(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    _field(isolated_db, mode="rainfed")
    fp.sync_farmers_from_fields()
    f = isolated_db.get_farmer_by_phone("998901234567")
    isolated_db.update_farmer(f["id"], auto_suggestions=1)
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    assert fp.auto_suggestions_job() == []


# --- strona rolnika (API po tokenie) ----------------------------------------

def test_panel_rolnika_api(app_client, isolated_db, monkeypatch):
    farmer, fid = _farmer(isolated_db)
    other = isolated_db.add_water_field("OBCE", "irrigation", WKT, farmer_phone="600100200")
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    aid = isolated_db.add_water_alarm(fid, date.today().isoformat(), "water_balance_deficit", "m", "nk",
                                      components_json=json.dumps({"water_balance": _summary()}))
    isolated_db.add_water_alarm(fid, date.today().isoformat(), "water_stress_risk", "m", "nk")  # pending
    isolated_db.set_water_alarm_triage(aid, "confirmed", triage_note="Podlej w nocy")
    tok = farmer["token"]

    assert app_client.get("/f/zly-token").status_code == 200       # strona się ładuje
    assert app_client.get("/api/f/zly-token").status_code == 404
    data = app_client.get(f"/api/f/{tok}").json()
    assert [f["id"] for f in data["fields"]] == [fid]
    f0 = data["fields"][0]
    assert len(f0["alerts"]) == 1 and f0["alerts"][0]["advisor_note"] == "Podlej w nocy"
    assert f0["alerts"][0]["dose_mm"] == 52.0 and f0["balance"]["suggested_dose_mm"] == 52.0
    assert "components_json" not in json.dumps(data) and "message" not in f0["alerts"][0]

    r = app_client.post(f"/api/f/{tok}/field/{fid}/irrigation",
                        json={"event_date": date.today().isoformat(), "method": "kroplowe", "amount_m3_ha": 300})
    assert r.status_code == 200 and r.json()["amount_mm"] == 30
    ev = isolated_db.get_irrigation_events(fid)[-1]
    assert ev["source"] == "farmer_panel" and ev["amount_mm"] == 30
    assert app_client.post(f"/api/f/{tok}/field/{other}/irrigation",
                           json={"event_date": date.today().isoformat(), "method": "kroplowe"}).status_code == 404
    old = (date.today() - timedelta(days=60)).isoformat()
    assert app_client.post(f"/api/f/{tok}/field/{fid}/rain",
                           json={"reading_date": old, "precip_mm": 5}).status_code == 422
    assert app_client.post(f"/api/f/{tok}/field/{fid}/rain",
                           json={"reading_date": date.today().isoformat(), "precip_mm": 5}).status_code == 200
    r = app_client.post(f"/api/f/{tok}/alarm/{aid}/ack", json={"ack": "will_irrigate", "note": "jutro"})
    assert r.status_code == 200
    assert isolated_db.get_water_alarm_by_id(aid)["farmer_ack"] == "will_irrigate"
    assert isolated_db.get_farmer_by_id(farmer["id"])["last_seen_at"]


def test_nowy_link_uniewaznia_stary(app_client, isolated_db):
    farmer, _ = _farmer(isolated_db)
    old = farmer["token"]
    r = app_client.post(f"/api/farmers/{farmer['id']}/new_link")
    assert r.status_code == 200 and r.json()["token"] != old
    assert app_client.get(f"/api/f/{old}").status_code == 404


def test_operator_lista_i_edycja(app_client, isolated_db):
    _field(isolated_db)
    lst = app_client.get("/api/farmers").json()
    assert len(lst) == 1 and lst[0]["link"].startswith("https://pm.example.org/f/")
    fid = lst[0]["id"]
    r = app_client.patch(f"/api/farmers/{fid}", json={"lang": "ru", "auto_suggestions": True})
    assert r.json()["lang"] == "ru" and r.json()["auto_suggestions"] == 1
    assert app_client.patch(f"/api/farmers/{fid}", json={"lang": "de"}).status_code == 422
    r = app_client.post(f"/api/farmers/{fid}/send_link")
    assert r.json()["status"] == "not_configured"
    assert app_client.get("/api/farmers/outbox").json()[0]["kind"] == "panel_link"
    assert app_client.get("/api/farmers/sms_status").json()["provider"] is None


def test_reczna_sugestia(app_client, isolated_db, monkeypatch):
    _, fid = _farmer(isolated_db)
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    r = app_client.post(f"/api/farmers/field/{fid}/suggest")
    assert r.status_code == 200 and "52" in r.json()["text"]
    monkeypatch.setattr(fp, "field_balance_summary",
                        lambda f, use_cache=True: _summary(suggested_dose_mm=None))
    assert app_client.post(f"/api/farmers/field/{fid}/suggest").status_code == 409



def test_auto_sugestie_wstrzymane_gdy_satelita_widzi_wode(isolated_db, sent, monkeypatch):
    isolated_db.init_db()
    monkeypatch.setenv("SMSAPI_TOKEN", "tok")
    farmer, fid = _farmer(isolated_db, auto_suggestions=1)
    isolated_db.add_irrigation_event(fid, (date.today() - timedelta(days=20)).isoformat(), "kroplowe",
                                     amount_mm=40)
    isolated_db.upsert_thermal(fid, [{"scene_date": (date.today() - timedelta(days=3)).isoformat(),
                                      "t_field": 30, "t_cold": 29.5, "t_hot": 50, "tsi": 0.05, "ndvi": 0.7}])
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    assert fp.auto_suggestions_job() == [] and not sent


def test_panel_rolnika_informuje_o_braku_dziennika(app_client, isolated_db, monkeypatch):
    farmer, fid = _farmer(isolated_db)
    monkeypatch.setattr(fp, "field_balance_summary", lambda f, use_cache=True: _summary())
    data = app_client.get(f"/api/f/{farmer['token']}").json()
    assert data["fields"][0]["balance_note"] == "no_irrigation_log"
