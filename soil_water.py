"""Pojemność wodna gleby (TAW) dla bilansu wodnego — bez testu laboratoryjnego.

TAW [mm] = woda dostępna na metr gleby (AWC, mm/m) × efektywna głębokość
korzeni (Zr, m). Źródła, od najlepszego:
  1. `soil_taw_mm` wpisane przez doradcę (np. z badania gleby) — wprost.
  2. Gatunek gleby określony w polu (próba wałeczkowa, 2 minuty, bez
     laboratorium) — AWC z tabeli gatunków.
  3. SoilGrids 2.0 (ISRIC, siatka 250 m): piasek, ił, węgiel organiczny
     i szkielet w 6 warstwach do 2 m, pobierane raz na pole.
  4. Domyślnie 100 mm (stary tryb, gdy nic innego nie ma).

AWC liczymy z uziarnienia funkcją pedotransferu Saxton & Rawls (2006, Soil
Sci. Soc. Am. J. 70:1569): θ(33 kPa) − θ(1500 kPa), pomniejszone o szkielet
(cfvo). Gotowych warstw wilgotności SoilGrids (wv0033/wv1500) celowo NIE
używamy — w sprawdzeniu dawały piaskom ~220 mm/m, a lessom ~90 mm/m,
odwrotnie niż w rzeczywistości.

Głębokość korzeni — FAO-56 tab. 22, dolna część przedziału (uprawy
nawadniane i zwykłe warunki), ograniczona głębokością gleby, jeśli doradca
ją poda (żwir, skała, warstwa nieprzepuszczalna). Uproszczenie: Zr stałe
w sezonie (FAO-56 przewiduje wzrost korzeni od siewu).
"""
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

DEFAULT_TAW_MM = 100.0
DEFAULT_ZR_M = 1.0
DEFAULT_OM_PCT = 1.5
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_DEPTHS = (("0-5cm", 0, 5), ("5-15cm", 5, 15), ("15-30cm", 15, 30),
                    ("30-60cm", 30, 60), ("60-100cm", 60, 100), ("100-200cm", 100, 200))
SOILGRIDS_RETRY_AFTER = timedelta(hours=6)    # po błędzie nie pytamy przy każdym bilansie
SOILGRIDS_ATTRIBUTION = "SoilGrids 2.0 (ISRIC, CC-BY 4.0)"

# Efektywna głębokość korzeni [m] — FAO-56 tab. 22 (dolna część przedziału).
ROOT_DEPTH_M = {
    "pszenica": 1.2, "wheat": 1.2, "bugdoy": 1.2,
    "jęczmień": 1.1, "jeczmien": 1.1, "barley": 1.1,
    "kukurydza": 1.2, "maize": 1.2, "corn": 1.2,
    "bawełna": 1.2, "bawelna": 1.2, "cotton": 1.2, "paxta": 1.2,
    "burak": 1.0, "burak cukrowy": 1.0, "sugar beet": 1.0,
    "ziemniak": 0.5, "ziemniaki": 0.5, "potato": 0.5,
    "rzepak": 1.2, "rapeseed": 1.2,
    "soja": 0.9, "soybean": 0.9,
    "ryż": 0.6, "ryz": 0.6, "rice": 0.6,
    "lucerna": 1.5, "alfalfa": 1.5,
    "warzywa": 0.5, "vegetables": 0.5,
}

# Gatunki gleby (klasy USDA, zbieżne z PTG 2008) — środek klasy (piasek %, ił %)
# do funkcji pedotransferu. Kolejność = od najlżejszej.
TEXTURE_CLASSES = {
    "sand": (92, 3), "loamy_sand": (82, 6), "sandy_loam": (65, 10),
    "loam": (40, 20), "silt_loam": (20, 15), "silt": (7, 6),
    "sandy_clay_loam": (60, 27), "clay_loam": (32, 34), "silty_clay_loam": (10, 34),
    "sandy_clay": (52, 42), "silty_clay": (7, 47), "clay": (20, 60),
}


def saxton_rawls(sand_pct: float, clay_pct: float, om_pct: float = DEFAULT_OM_PCT) -> tuple:
    """(θ_FC, θ_WP) [m³/m³] — Saxton & Rawls 2006, równania 1–2."""
    s, c, om = sand_pct / 100.0, clay_pct / 100.0, min(max(om_pct, 0.0), 8.0)
    t15 = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
    wp = t15 + (0.14 * t15 - 0.02)
    t33 = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
    fc = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    return fc, wp


def awc_mm_per_m(sand_pct: float, clay_pct: float, om_pct: float = DEFAULT_OM_PCT,
                 coarse_pct: float = 0.0) -> float:
    fc, wp = saxton_rawls(sand_pct, clay_pct, om_pct)
    return max(0.0, (fc - wp) * 1000.0 * (1.0 - min(max(coarse_pct, 0.0), 90.0) / 100.0))


def usda_texture(sand: float, clay: float) -> str:
    """Klasa uziarnienia USDA z udziału piasku i iłu [%]."""
    silt = 100.0 - sand - clay
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy_sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):
        return "sandy_loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt_loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy_clay_loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay_loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty_clay_loam"
    if clay >= 35 and sand > 45:
        return "sandy_clay"
    if clay >= 40 and silt >= 40:
        return "silty_clay"
    return "clay"


def root_depth_m(crop_type: Optional[str]) -> tuple:
    """(Zr [m], czy znana uprawa)."""
    key = str(crop_type).strip().lower() if crop_type else ""
    if key in ROOT_DEPTH_M:
        return ROOT_DEPTH_M[key], True
    return DEFAULT_ZR_M, False


# ---------------------------------------------------------------------------
# SoilGrids
# ---------------------------------------------------------------------------

def fetch_soilgrids(lat: float, lon: float, timeout: int = 40) -> list:
    """Profil z SoilGrids: [{"top", "bottom", "sand", "clay", "om", "coarse"}]
    (procenty). Warstwy bez danych (woda, zabudowa) pomijane. Rzuca
    RuntimeError, gdy API nie odpowiada albo nie ma żadnych danych."""
    params = [("lon", round(float(lon), 5)), ("lat", round(float(lat), 5)), ("value", "mean")]
    params += [("property", p) for p in ("sand", "clay", "soc", "cfvo")]
    params += [("depth", d[0]) for d in SOILGRIDS_DEPTHS]
    last = None
    for attempt in range(2):
        try:
            r = requests.get(SOILGRIDS_URL, params=params, timeout=timeout)
            if r.status_code == 200:
                break
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 + 3 * attempt)
    else:
        raise RuntimeError(f"SoilGrids niedostępne: {last}")
    layers = {l["name"]: {d["label"]: d["values"].get("mean") for d in l["depths"]}
              for l in r.json().get("properties", {}).get("layers", [])}
    out = []
    for label, top, bottom in SOILGRIDS_DEPTHS:
        sand = (layers.get("sand") or {}).get(label)
        clay = (layers.get("clay") or {}).get(label)
        if sand is None or clay is None:
            continue
        soc = (layers.get("soc") or {}).get(label)          # dg/kg
        cfvo = (layers.get("cfvo") or {}).get(label)        # cm³/dm³
        out.append({"top": top, "bottom": bottom, "sand": sand / 10.0, "clay": clay / 10.0,
                    "om": round(soc / 100.0 * 1.724, 2) if soc is not None else DEFAULT_OM_PCT,
                    "coarse": cfvo / 10.0 if cfvo is not None else 0.0})
    if not out:
        raise RuntimeError("SoilGrids nie ma danych dla tego punktu (woda, zabudowa?).")
    return out


def profile_awc(profile: list, zr_m: float) -> Optional[dict]:
    """Woda dostępna w strefie 0–Zr z profilu warstwowego: {"taw_mm",
    "awc_mm_m", "sand", "clay", "om", "coarse", "depth_cm"} (średnie ważone
    miąższością w strefie korzeni). Poniżej ostatniej warstwy — jak ostatnia."""
    zr_cm = zr_m * 100.0
    taw = 0.0
    acc = {"sand": 0.0, "clay": 0.0, "om": 0.0, "coarse": 0.0}
    covered = 0.0
    layers = sorted(profile, key=lambda l: l["top"])
    if not layers:
        return None
    for i, l in enumerate(layers):
        bottom = l["bottom"] if i < len(layers) - 1 else max(l["bottom"], zr_cm)
        th = max(0.0, min(bottom, zr_cm) - l["top"])
        if th <= 0:
            continue
        taw += awc_mm_per_m(l["sand"], l["clay"], l["om"], l["coarse"]) * th / 100.0
        for k in acc:
            acc[k] += l[k] * th
        covered += th
    if covered <= 0:
        return None
    mean = {k: round(v / covered, 1) for k, v in acc.items()}
    return {"taw_mm": round(taw, 0), "awc_mm_m": round(taw / (covered / 100.0), 0),
            "depth_cm": round(covered), **mean}


# ---------------------------------------------------------------------------
# Rozstrzygnięcie dla pola
# ---------------------------------------------------------------------------

def _soilgrids_profile(field: dict, fetch: bool) -> Optional[list]:
    import db
    row = db.get_field_soil(field["id"]) or {}
    if row.get("sg_profile_json"):
        return json.loads(row["sg_profile_json"])
    if not fetch or not field.get("geometry_wkt"):
        return None
    err_at = row.get("sg_error_at")
    if err_at and datetime.fromisoformat(err_at) > datetime.now() - SOILGRIDS_RETRY_AFTER:
        return None
    from shapely import wkt as shapely_wkt
    c = shapely_wkt.loads(field["geometry_wkt"]).representative_point()
    try:
        prof = fetch_soilgrids(c.y, c.x)
    except Exception as e:
        log.warning("SoilGrids dla pola %s: %s", field.get("id"), e)
        db.save_field_soil(field["id"], sg_error=str(e)[:300], sg_error_at=datetime.now().isoformat(timespec="seconds"))
        return None
    db.save_field_soil(field["id"], sg_profile_json=json.dumps(prof), sg_fetched_at=datetime.now().isoformat(timespec="seconds"),
                       sg_error=None, sg_error_at=None)
    return prof


def field_taw(field: dict, fetch: bool = True) -> dict:
    """Pojemność wodna pola z jawnym źródłem. Klucze: taw_mm, source
    (manual | texture_field | soilgrids | default), awc_mm_m, zr_m,
    zr_limited, texture, texture_from, crop_known, root_limit_cm, sand, clay."""
    import db
    soil = db.get_field_soil(field["id"]) or {}
    zr, crop_known = root_depth_m(field.get("crop_type"))
    limit = soil.get("root_limit_cm")
    zr_limited = bool(limit and limit / 100.0 < zr)
    if zr_limited:
        zr = limit / 100.0
    base = {"zr_m": round(zr, 2), "zr_limited": zr_limited, "crop_known": crop_known,
            "root_limit_cm": limit, "texture": soil.get("texture_class"), "texture_from": None,
            "awc_mm_m": None, "sand": None, "clay": None}
    prof = _soilgrids_profile(field, fetch)
    sg = profile_awc(prof, zr) if prof else None
    if sg:
        base.update(sand=sg["sand"], clay=sg["clay"])

    if field.get("soil_taw_mm"):
        taw = float(field["soil_taw_mm"])
        return {**base, "taw_mm": taw, "source": "manual", "awc_mm_m": round(taw / zr) if zr else None,
                "texture_from": "field" if soil.get("texture_class") else None}
    if soil.get("texture_class") in TEXTURE_CLASSES:
        s, c = TEXTURE_CLASSES[soil["texture_class"]]
        om = sg["om"] if sg else DEFAULT_OM_PCT
        awc = awc_mm_per_m(s, c, om)
        return {**base, "taw_mm": round(awc * zr), "source": "texture_field", "awc_mm_m": round(awc),
                "texture_from": "field"}
    if sg:
        return {**base, "taw_mm": round(sg["awc_mm_m"] * zr), "source": "soilgrids",
                "awc_mm_m": sg["awc_mm_m"], "texture": usda_texture(sg["sand"], sg["clay"]),
                "texture_from": "soilgrids", "coarse_pct": sg["coarse"], "om_pct": sg["om"]}
    return {**base, "taw_mm": DEFAULT_TAW_MM, "source": "default"}


# ---------------------------------------------------------------------------
# Zasolenie i wody gruntowe
# ---------------------------------------------------------------------------
# Tolerancja upraw na zasolenie: (ECe_próg [dS/m], b [% spadku plonu na
# 1 dS/m powyżej progu], Ky) — FAO-56 tab. 23 (Maas–Hoffman) i tab. 24.
# Redukcja pobierania wody (FAO-56 r. 89/90): Ks_sól = 1 − b/(Ky·100)·(ECe − próg),
# mnożona przez Ks od niedoboru wody.
SALT_TOLERANCE = {
    "bawełna": (7.7, 5.2, 0.85), "bawelna": (7.7, 5.2, 0.85), "cotton": (7.7, 5.2, 0.85), "paxta": (7.7, 5.2, 0.85),
    "pszenica": (6.0, 7.1, 1.05), "wheat": (6.0, 7.1, 1.05), "bugdoy": (6.0, 7.1, 1.05),
    "jęczmień": (8.0, 5.0, 1.0), "jeczmien": (8.0, 5.0, 1.0), "barley": (8.0, 5.0, 1.0),
    "kukurydza": (1.7, 12.0, 1.25), "maize": (1.7, 12.0, 1.25), "corn": (1.7, 12.0, 1.25),
    "burak": (7.0, 5.9, 1.0), "burak cukrowy": (7.0, 5.9, 1.0), "sugar beet": (7.0, 5.9, 1.0),
    "ziemniak": (1.7, 12.0, 1.1), "ziemniaki": (1.7, 12.0, 1.1), "potato": (1.7, 12.0, 1.1),
    "soja": (5.0, 20.0, 0.85), "soybean": (5.0, 20.0, 0.85),
    "ryż": (3.0, 12.0, 1.0), "ryz": (3.0, 12.0, 1.0), "rice": (3.0, 12.0, 1.0),
    "lucerna": (2.0, 7.3, 1.1), "alfalfa": (2.0, 7.3, 1.1),
    "warzywa": (2.5, 9.9, 1.05), "vegetables": (2.5, 9.9, 1.05),
}
DEFAULT_SALT_TOLERANCE = (4.0, 10.0, 1.0)     # uprawa średnio wrażliwa — gdy nieznana

# Klasy zasolenia (jak w uzbeckiej praktyce: niezasolona, słabo, średnio,
# silnie, bardzo silnie) — środek przedziału ECe [dS/m] wg klasyfikacji FAO.
SALINITY_CLASSES = {"none": 1.0, "slight": 3.0, "moderate": 6.0, "strong": 12.0, "very_strong": 20.0}

# EC roztworu 1:5 -> ECe (Slavich & Petterson 1993, przybliżenie wg gatunku).
EC15_TO_ECE = {
    "sand": 15.0, "loamy_sand": 13.8, "sandy_loam": 9.5, "loam": 8.6, "silt_loam": 8.6, "silt": 8.6,
    "sandy_clay_loam": 8.6, "clay_loam": 8.6, "silty_clay_loam": 7.5, "sandy_clay": 7.5,
    "silty_clay": 6.0, "clay": 5.8,
}
ECW_TO_ECE = 1.5           # FAO-29: przy zwykłym wymywaniu ECe ≈ 1,5 × EC wody
GW_SHALLOW_M = 2.0         # płytsze wody gruntowe: podsiąk do korzeni i ryzyko zasolenia
EC_METHODS = ("ece", "ec_1_5", "ec_water")


def salinity_ece(soil: dict, texture: Optional[str]) -> tuple:
    """(ECe [dS/m] | None, źródło). Pomiar ma pierwszeństwo przed klasą."""
    v, m = soil.get("ec_value"), soil.get("ec_method")
    if v is not None and m in EC_METHODS:
        if m == "ece":
            return float(v), "ec_paste"
        if m == "ec_1_5":
            return float(v) * EC15_TO_ECE.get(texture or "", 8.6), "ec_1_5"
        return float(v) * ECW_TO_ECE, "ec_water"
    if soil.get("salinity_class") in SALINITY_CLASSES:
        return SALINITY_CLASSES[soil["salinity_class"]], "class"
    return None, "none"


def field_salinity(field: dict, soil: Optional[dict] = None, texture: Optional[str] = None) -> dict:
    """Wpływ zasolenia na pobieranie wody + wody gruntowe + zgłoszenia rolnika."""
    import db
    soil = soil if soil is not None else (db.get_field_soil(field["id"]) or {})
    key = str(field.get("crop_type") or "").strip().lower()
    thr, b, ky = SALT_TOLERANCE.get(key, DEFAULT_SALT_TOLERANCE)
    ece, src = salinity_ece(soil, texture or soil.get("texture_class"))
    ks = 1.0
    loss = 0.0
    if ece is not None and ece > thr:
        loss = min(100.0, b * (ece - thr))
        ks = max(0.0, 1.0 - b / (ky * 100.0) * (ece - thr))
    gw = soil.get("gw_depth_m")
    hints = []
    if soil.get("farmer_salt_crust") in ("spots", "much") and src in ("none", "class"):
        hints.append("farmer_crust")
    if soil.get("farmer_leaching") == "yes" and src == "none":
        hints.append("farmer_leaching")
    if gw is not None and gw < GW_SHALLOW_M:
        hints.append("gw_shallow")
    return {"ece": None if ece is None else round(ece, 1), "ece_source": src,
            "threshold": thr, "b": b, "ky": ky, "crop_known": key in SALT_TOLERANCE,
            "ks_salt": round(ks, 3), "yield_loss_pct": round(loss, 0),
            "salinity_class": soil.get("salinity_class"), "ec_value": soil.get("ec_value"),
            "ec_method": soil.get("ec_method"), "gw_depth_m": gw,
            "gw_shallow": gw is not None and gw < GW_SHALLOW_M,
            "farmer_salt_crust": soil.get("farmer_salt_crust"), "farmer_leaching": soil.get("farmer_leaching"),
            "farmer_leaching_m3": soil.get("farmer_leaching_m3"), "farmer_report_at": soil.get("farmer_report_at"),
            "hints": hints}
