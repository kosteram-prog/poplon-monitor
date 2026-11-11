"""report_pdf.py — generowanie raportów PDF z wyników /api/check_multi.

WAŻNE: ten moduł NIE wykonuje żadnych nowych zapytań do Copernicus/Sentinel
Hub/Open-Meteo. Buduje dokument WYŁĄCZNIE z już policzonej odpowiedzi
check_multi (przekazanej jako dict) — tej samej, którą widzi operator w UI.
Przeznaczenie: dokumentacja na potrzeby kontroli (np. ARiMR) — raport
pojedynczej działki albo zbiorczy raport wszystkich monitorowanych działek
jednego rolnika (jedna sekcja na działkę, PageBreak między nimi).

Zasada zgodna z resztą aplikacji: raport pokazuje TYLKO to, co już zostało
policzone i pokazane operatorowi — żadnych nowych progów, żadnej dodatkowej
interpretacji nieobecnej w danych źródłowych.
"""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

_STYLES = getSampleStyleSheet()
_STYLES.add(ParagraphStyle(
    name="H1poplon", parent=_STYLES["Heading1"], fontSize=16, spaceAfter=6))
_STYLES.add(ParagraphStyle(
    name="H2poplon", parent=_STYLES["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4))
_STYLES.add(ParagraphStyle(
    name="Caveat", parent=_STYLES["Normal"], fontSize=8, textColor=colors.grey,
    leading=11))
_STYLES.add(ParagraphStyle(
    name="Body", parent=_STYLES["Normal"], fontSize=9.5, leading=13))

_EVENT_TYPE_LABELS = {
    "tillage": "orka/uprawa gleby",
    "termination": "likwidacja poplonu",
}

_VERDICT_COLORS = {
    "green":  colors.HexColor("#3f7d3f"),
    "yellow": colors.HexColor("#b8860b"),
    "red":    colors.HexColor("#b02a2a"),
}


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _chart_image(multi_records: list, agro_events: list) -> "io.BytesIO | None":
    """Wykres NDVI/EVI/SAVI/NDRE w czasie z zaznaczonymi zdarzeniami
    agrotechnicznymi — PNG w pamięci, do osadzenia w PDF. Ten sam surowy
    zestaw danych co wykres w UI (multi_records), nie osobne przeliczenie."""
    recs = [r for r in multi_records if r.get("ndvi") is not None or r.get("evi") is not None]
    if not recs:
        return None
    try:
        dates = [datetime.fromisoformat(str(r["date"])) for r in recs]
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6.3, 2.6), dpi=130)
    series = [
        ("ndvi", "NDVI", "#3f7d3f"),
        ("evi",  "EVI",  "#7a4fb5"),
        ("savi", "SAVI", "#c98a2c"),
        ("ndre", "NDRE", "#2c7ab5"),
    ]
    for key, label, color in series:
        xs, ys = [], []
        for d, r in zip(dates, recs):
            v = r.get(key)
            if v is not None:
                xs.append(d)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2, label=label, color=color)

    for ev in agro_events or []:
        try:
            ed = datetime.fromisoformat(str(ev["date"]))
        except Exception:
            continue
        ax.axvline(ed, color="#999999", linestyle="--", linewidth=0.8)

    ax.set_ylim(-0.05, 1.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", ncol=4, frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def _parcel_section(data: dict) -> list:
    """Zwraca listę flowables reportlab dla jednej działki na bazie
    odpowiedzi check_multi (dict identyczny jak JSON zwracany przez API)."""
    story = []
    parcel_id = data.get("parcel_id", "—")
    info = data.get("parcel_info") or {}
    cs = data.get("confidence") or {}
    verdict = data.get("fused_verdict") or {}
    category = data.get("fused_category") or data.get("category")
    reasons = data.get("category_reasons") or []
    peak = data.get("peak_record")
    biomass = data.get("biomass")
    events = data.get("agro_events") or []
    wc = data.get("weather_context") or {}
    tc = data.get("terrain_context")
    gc = data.get("growing_conditions")

    story.append(Paragraph(f"Działka {parcel_id}", _STYLES["H1poplon"]))
    loc_bits = [info.get(k) for k in ("wojewodztwo", "powiat", "gmina") if info.get(k)]
    meta_line = f"TERYT: {info.get('teryt', '—')}"
    if loc_bits:
        meta_line += " • " + ", ".join(loc_bits)
    story.append(Paragraph(meta_line, _STYLES["Caveat"]))
    story.append(Paragraph(
        f"Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}", _STYLES["Caveat"]))
    story.append(Spacer(1, 4 * mm))

    verdict_color = _VERDICT_COLORS.get(category, colors.black)
    verdict_style = ParagraphStyle(
        name="VerdictLine", parent=_STYLES["Body"], textColor=verdict_color, fontSize=11)
    status_txt = verdict.get("status") or verdict.get("label") or "—"
    story.append(Paragraph(f"<b>Status: {status_txt}</b> (kategoria: {category or '—'})", verdict_style))
    if cs.get("score") is not None:
        story.append(Paragraph(
            f"Confidence Score: {cs['score']:.0f}/100 (ocena: {cs.get('grade', '—')})"
            + (f" • zakres orientacyjny: {cs['score_low']:.0f}–{cs['score_high']:.0f} "
               f"(przy tej jakości i kompletności danych, nie jest to przedział "
               f"statystyczny)" if cs.get("score_low") is not None else ""),
            _STYLES["Body"]))
    story.append(Spacer(1, 2 * mm))

    if reasons:
        story.append(Paragraph("Uzasadnienie:", _STYLES["H2poplon"]))
        for r in reasons:
            story.append(Paragraph(f"• {r}", _STYLES["Body"]))

    chart_buf = _chart_image(data.get("multi_records") or [], events)
    if chart_buf:
        story.append(Spacer(1, 3 * mm))
        story.append(Image(chart_buf, width=160 * mm, height=160 * mm * 2.6 / 6.3))

    if peak:
        story.append(Paragraph("Dzień reprezentatywny (peak):", _STYLES["H2poplon"]))
        rows = [["Data", "NDVI", "EVI2", "SAVI", "NDRE", "NDMI"],
                [peak.get("date", "—"), _fmt(peak.get("ndvi")), _fmt(peak.get("evi")),
                 _fmt(peak.get("savi")), _fmt(peak.get("ndre")), _fmt(peak.get("ndwi"))]]
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1E8")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)

    if biomass:
        story.append(Paragraph("Szacowana biomasa i sekwestracja CO₂ (szacunek roboczy, niekalibrowany):",
                                _STYLES["H2poplon"]))
        bio_range = (f"{biomass['biomass_low_t_ha']:.1f}–{biomass['biomass_high_t_ha']:.1f}"
                     if biomass.get("biomass_low_t_ha") is not None
                     else _fmt(biomass.get("biomass_t_ha"), 1))
        co2_range = (f"{biomass['carbon_co2eq_low_t_ha']:.1f}–{biomass['carbon_co2eq_high_t_ha']:.1f}"
                     if biomass.get("carbon_co2eq_low_t_ha") is not None
                     else _fmt(biomass.get("carbon_co2eq_t_ha"), 1))
        story.append(Paragraph(f"Biomasa: {bio_range} t s.m./ha • CO₂eq: {co2_range} t/ha", _STYLES["Body"]))

    if events:
        story.append(Paragraph("Wykryte zdarzenia agrotechniczne:", _STYLES["H2poplon"]))
        for ev in events:
            label = _EVENT_TYPE_LABELS.get(ev.get("type"), ev.get("type", "?"))
            story.append(Paragraph(
                f"• {ev.get('date', '—')} — {label} (pewność: {ev.get('confidence', '—')})",
                _STYLES["Body"]))

    if wc or gc or tc:
        story.append(Paragraph("Kontekst pogodowo-terenowy (informacyjny, nie wpływa na ocenę):",
                                _STYLES["H2poplon"]))
        bits = []
        if wc.get("frost_days"):
            bits.append(f"przymrozki: {wc['frost_days']} dni")
        if wc.get("dry_spell_days"):
            bits.append(f"posucha: {wc['dry_spell_days']} dni")
        if gc and gc.get("cumulative_gdd") is not None:
            bits.append(f"GDD: {gc['cumulative_gdd']:.0f} (baza {gc.get('gdd_base_temp_c')}°C)")
        if gc and gc.get("total_precip_mm") is not None:
            bits.append(f"opad: {gc['total_precip_mm']:.0f} mm")
        if tc and tc.get("slope_deg") is not None:
            bits.append(f"nachylenie: {tc['slope_deg']:.1f}° (erozja: {tc.get('erosion_risk', '—')})")
        story.append(Paragraph(" • ".join(bits) if bits else "brak danych", _STYLES["Body"]))

    story.append(Paragraph(
        "Raport wygenerowany automatycznie na bazie danych satelitarnych (Sentinel-1/2, "
        "Copernicus) i pogodowych (Open-Meteo). Wartości biomasy/CO₂ i warunków wzrostu mają "
        "charakter informacyjny/szacunkowy — nie stanowią pomiaru bezpośredniego.",
        _STYLES["Caveat"]))
    return story


def build_parcel_report(data: dict) -> bytes:
    """Raport PDF dla pojedynczej działki (jedna odpowiedź check_multi)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=18 * mm, rightMargin=18 * mm,
                             topMargin=16 * mm, bottomMargin=16 * mm)
    doc.build(_parcel_section(data))
    return buf.getvalue()


def build_farmer_report(parcels_data: list, farmer_name: "str | None" = None,
                         farmer_phone: "str | None" = None) -> bytes:
    """Zbiorczy raport PDF — po jednej sekcji na działkę (PageBreak między
    nimi), dla wszystkich monitorowanych działek jednego rolnika."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=18 * mm, rightMargin=18 * mm,
                             topMargin=16 * mm, bottomMargin=16 * mm)
    story = []
    header = farmer_name or farmer_phone or "—"
    story.append(Paragraph(f"Raport zbiorczy — {header}", _STYLES["H1poplon"]))
    story.append(Paragraph(f"Liczba działek: {len(parcels_data)} • "
                            f"Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            _STYLES["Caveat"]))
    story.append(Spacer(1, 4 * mm))
    for i, data in enumerate(parcels_data):
        if i > 0:
            story.append(PageBreak())
        story.extend(_parcel_section(data))
    doc.build(story)
    return buf.getvalue()
