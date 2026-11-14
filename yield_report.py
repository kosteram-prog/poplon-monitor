"""yield_report.py — raport PDF analizy wieloletniej stref plonu.

Jak report_pdf.py: dokument budowany WYŁĄCZNIE z już policzonych plików
analizy (uploads/yield_analyses/<id>/) i jej statystyk — bez nowych zapytań
do Copernicus. Zawartość: płodozmian z plonami, mapki względne z każdego
roku, mapa stref z wielolecia, tabela stref z dawkami nawozu (jeśli
ustawione), próby glebowe per strefa, zastrzeżenia metodyczne.

Czcionka: DejaVu Sans (z pakietu matplotlib — zawsze dostępna, pełne
polskie znaki; standardowa Helvetica reportlaba ich nie ma)."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

import yield_mvi as ym
import yield_multi as ymu

_FONT = "PoplonDejaVu"


def _register_fonts():
    if _FONT in pdfmetrics.getRegisteredFontNames():
        return
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    pdfmetrics.registerFont(TTFont(_FONT, os.path.join(base, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT + "-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))


def _styles():
    st = getSampleStyleSheet()
    def add(name, **kw):
        st.add(ParagraphStyle(name=name, **kw))
    add("YH1", parent=st["Heading1"], fontName=_FONT + "-Bold", fontSize=15, spaceAfter=4)
    add("YH2", parent=st["Heading2"], fontName=_FONT + "-Bold", fontSize=11.5, spaceBefore=10, spaceAfter=4)
    add("YBody", parent=st["Normal"], fontName=_FONT, fontSize=9.5, leading=13)
    add("YCaveat", parent=st["Normal"], fontName=_FONT, fontSize=7.8, textColor=colors.grey, leading=10.5)
    return st


def _fmt(v, suffix="", nd=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        v = round(v, nd)
        if v == int(v):
            v = int(v)
    return f"{v}{suffix}"


def _read_tif(path):
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1), ds.bounds


def _map_png(adir, analysis, kind, year=None, dpi=160):
    """Renderuje mapę do PNG (ścieżka pliku tymczasowego w adir):
    kind='zones' (strefy 1..5) albo kind='rel' (mapa względna roku)."""
    import numpy as np
    if kind == "zones":
        arr, bounds = _read_tif(os.path.join(adir, "strefy.tif"))
        data = np.ma.masked_equal(arr, 0)
        cmap = ListedColormap(ym.ZONE_COLORS)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
        out = os.path.join(adir, "_pdf_zones.png")
    else:
        arr, bounds = _read_tif(os.path.join(adir, f"rel_{year}.tif"))
        data = np.ma.masked_invalid(arr)
        cmap, norm = "RdYlGn", None
        out = os.path.join(adir, f"_pdf_rel_{year}.png")
    fig, ax = plt.subplots(figsize=(4.4, 3.6), dpi=dpi)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    kw = {"vmin": 0.7, "vmax": 1.3} if kind == "rel" else {}
    ax.imshow(data, cmap=cmap, norm=norm, extent=extent, interpolation="nearest", **kw)
    _plot_outline(ax, analysis.get("geometry_wkt"))
    ax.set_aspect(1.0 / max(0.2, math.cos(math.radians((bounds.bottom + bounds.top) / 2))))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def _plot_outline(ax, wkt_str):
    if not wkt_str:
        return
    try:
        from shapely import wkt as shapely_wkt
        g = shapely_wkt.loads(wkt_str)
        geoms = getattr(g, "geoms", [g])
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="#222222", linewidth=1.0)
    except Exception:
        pass


def _zone_table(stats, styles):
    fert = stats.get("fertilizer")
    dose = {z["zone"]: z for z in fert["zones"]} if fert else {}
    head = ["", "Strefa", "Pow. [ha]", "Udział", "Plon wzgl.", "CV między- roczny"]
    if fert:
        head += ["Dawka [kg/ha]", "Razem [kg]"]
    rows = [head]
    swatches = []
    for z in stats["zones"]:
        if not z["area_ha"]:
            continue
        r = ["", z["label"], _fmt(z["area_ha"], nd=2), _fmt(z["share_pct"], "%"),
             _fmt(z["mean_rel_pct"], "%"), _fmt(z["interyear_cv_pct"], "%")]
        if fert:
            d = dose.get(z["zone"])
            r += [_fmt(d and d["dose_kg_ha"]), _fmt(d and d["dose_total_kg"], nd=0)]
        swatches.append((len(rows), colors.HexColor(z["color"])))
        rows.append(r)
    if fert:
        rows.append(["", "RAZEM", _fmt(fert["area_ha"], nd=2), "100%", "", "",
                     f"śr. {_fmt(fert['effective_avg_kg_ha'])}", _fmt(fert["total_kg"], nd=0)])
    t = Table(rows, colWidths=[7 * mm] + [None] * (len(head) - 1))
    style = [("FONTNAME", (0, 0), (-1, -1), _FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("FONTNAME", (0, 0), (-1, 0), _FONT + "-Bold"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
             ("ALIGN", (2, 1), (-1, -1), "CENTER"),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2ee"))]
    for ri, col in swatches:
        style.append(("BACKGROUND", (0, ri), (0, ri), col))
    if fert:
        style.append(("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), _FONT + "-Bold"))
    t.setStyle(TableStyle(style))
    return t


def _soil_tables(soil, styles):
    out = []
    zs = soil.get("zone_summary") or []
    if zs:
        rows = [["Strefa", "n prób", "pH", "P₂O₅", "K₂O", "Mg", "Próchnica"]]
        for r in zs:
            rows.append([r["label"], r["n"], _fmt(r.get("ph"), nd=2), _fmt(r.get("p2o5")),
                        _fmt(r.get("k2o")), _fmt(r.get("mg")), _fmt(r.get("humus"), nd=2)])
        t = Table(rows)
        t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), _FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("FONTNAME", (0, 0), (-1, 0), _FONT + "-Bold"),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                               ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2ee"))]))
        out.append(Paragraph("Próby glebowe — średnie per strefa (mg/100 g gleby):", styles["YH2"]))
        out.append(t)
        n_nozone = sum(1 for s in soil.get("samples", []) if s.get("zone") is None)
        if n_nozone:
            out.append(Spacer(1, 2 * mm))
            out.append(Paragraph(f"{n_nozone} prób bez współrzędnych w polu — nie przypisano do stref.",
                                 styles["YCaveat"]))
    return out


def build_analysis_pdf(analysis: dict) -> str:
    """analysis: słownik z yield_multi_api._public(with_geojson=True)."""
    _register_fonts()
    styles = _styles()
    aid = analysis["id"]
    stats = analysis["stats"] or {}
    adir = os.path.join(ymu.ANALYSES_DIR, str(aid))
    out_path = os.path.join(adir, "raport.pdf")
    story = []
    title = analysis.get("label") or analysis.get("parcel_id") or f"analiza #{aid}"
    story.append(Paragraph(f"Strefy plonu z wielolecia — {title}", styles["YH1"]))
    sub = []
    if analysis.get("parcel_id"):
        sub.append(f"działki: {analysis['parcel_id']}")
    sub.append(f"powierzchnia pola: {_fmt(stats.get('field_area_ha'), ' ha', 2)}")
    sub.append(f"analizowane wnętrze: {_fmt(stats.get('area_ha'), ' ha', 2)}")
    story.append(Paragraph(" • ".join(sub), styles["YBody"]))
    story.append(Spacer(1, 3 * mm))

    # Płodozmian
    story.append(Paragraph("Płodozmian i plony (dane rolnika):", styles["YH2"]))
    rows = [["Rok", "Uprawa", "Plon [t/ha]", "Okno analizy", "Sceny", "CV mapy", "Status"]]
    for y in stats.get("years", []):
        ok = y.get("status") == "ok"
        rows.append([y["year"], y.get("crop", "—"), _fmt(y.get("yield_t_ha")),
                     f"{y.get('date_from', '')} – {y.get('date_to', '')}",
                     _fmt(y.get("median_scenes"), nd=0) if ok else "—",
                     _fmt(y.get("cv_pct"), "%") if ok else "—",
                     "OK" if ok else f"pominięty ({(y.get('error') or '')[:60]})"])
    t = Table(rows)
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), _FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("FONTNAME", (0, 0), (-1, 0), _FONT + "-Bold"),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                           ("ALIGN", (0, 1), (2, -1), "CENTER"), ("ALIGN", (4, 1), (5, -1), "CENTER"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2ee"))]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # Mapa stref z wielolecia + tabela stref/dawek
    story.append(Paragraph("Mapa stref z wielolecia (średnia map względnych):", styles["YH2"]))
    img = _map_png(adir, analysis, "zones")
    story.append(Image(img, width=120 * mm, height=98 * mm, kind="proportional"))
    legend = "   ".join(f"{l}" for l in ym.ZONE_LABELS)
    story.append(Paragraph(f"Strefy względem średniej pola (=100%): {legend}. "
                           f"Lata w analizie: {stats.get('years_ok')} • CV mapy: {_fmt(stats.get('cv_pct'), '%')} "
                           f"• piksele stabilne między latami (CV<{ymu.STABLE_CV_PCT:.0f}%): "
                           f"{_fmt(stats.get('stable_share_pct'), '%')}", styles["YCaveat"]))
    story.append(Spacer(1, 2 * mm))
    story.append(_zone_table(stats, styles))

    fert = stats.get("fertilizer")
    if fert:
        strat = ("proporcjonalnie do potencjału strefy (pod wynos składników)"
                 if fert["strategy"] == "potencjal" else
                 "wyrównawczo — więcej w strefach słabych (odbudowa zasobności)")
        extra = []
        if fert.get("dose_min_kg_ha") is not None:
            extra.append(f"min {_fmt(fert['dose_min_kg_ha'])} kg/ha")
        if fert.get("dose_max_kg_ha") is not None:
            extra.append(f"max {_fmt(fert['dose_max_kg_ha'])} kg/ha")
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            f"Nawożenie: {fert.get('fert_name') or 'nawóz'} • dawka średnia zadana "
            f"{_fmt(fert['avg_dose_kg_ha'])} kg/ha • strategia: {strat} • zaokrąglenie do "
            f"{fert['rounding_kg']} kg{(' • ' + ', '.join(extra)) if extra else ''}. "
            f"Po zaokrągleniu średnia rzeczywista {_fmt(fert['effective_avg_kg_ha'])} kg/ha, "
            f"łącznie {_fmt(fert['total_kg'], ' kg', 0)} na {_fmt(fert['area_ha'], ' ha', 2)}.",
            styles["YBody"]))

    soil = analysis.get("soil")
    if soil:
        story.extend(_soil_tables(soil, styles))

    # Mapki roczne
    yrs = [y for y in stats.get("years", []) if y.get("status") == "ok"]
    if yrs:
        story.append(PageBreak())
        story.append(Paragraph("Mapy względne z poszczególnych lat:", styles["YH2"]))
        story.append(Paragraph("Skala wspólna: 70% (czerwień) – 130% (zieleń) średniej pola w danym roku. "
                               "Mapa względna pokazuje, GDZIE w polu było lepiej/gorzej — niezależnie od uprawy.",
                               styles["YCaveat"]))
        story.append(Spacer(1, 2 * mm))
        cells, row = [], []
        for y in yrs:
            img = _map_png(adir, analysis, "rel", y["year"])
            cap = f"{y['year']} — {y.get('crop', '')}, {_fmt(y.get('yield_t_ha'))} t/ha"
            row.append([Image(img, width=78 * mm, height=64 * mm, kind="proportional"),
                        Paragraph(cap, styles["YBody"])])
            if len(row) == 2:
                cells.append(row); row = []
        if row:
            cells.append(row + [""])
        t = Table(cells, colWidths=[85 * mm, 85 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(t)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Metodyka i zastrzeżenia: mapy względne z Sentinel-2 metodą MVI (mediana EVI2 × 75. percentyl "
        "NDMI w oknie sezonu każdej uprawy; średnia pola = plon zgłoszony przez rolnika). Strefy z "
        "wielolecia = średnia map względnych; piksel wymaga danych z co najmniej połowy lat. Mapa jest "
        "WZGLĘDNA — pokazuje rozkład w polu, nie bezwzględny pomiar plonu; do czasu porównania z mapą "
        "z kombajnu wartości traktować orientacyjnie. Dawki nawozu wyliczone wyłącznie z rozkładu "
        "potencjału plonowania wybraną strategią — nie uwzględniają automatycznie zasobności gleby "
        "(tabela prób glebowych służy do ręcznej korekty przez agronoma). Pas brzegowy pola "
        f"({ym.EDGE_BUFFER_M:.0f} m) pominięty w analizie.", styles["YCaveat"]))

    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title=f"Strefy plonu — {title}")
    doc.build(story)
    return out_path
