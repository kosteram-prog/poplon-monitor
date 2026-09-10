"""Testy report_pdf.py: budowa PDF z (zamockowanego) wyniku check_multi.
Sprawdzamy tylko, że dokument się buduje (poprawny nagłówek PDF, niepusty),
bez porównywania treści bajt-po-bajt (kruche, zależne od reportlab/matplotlib)."""
from datetime import date

import report_pdf


def _fake_check_multi_response(parcel_id="141201_1.0001.6509"):
    return {
        "parcel_id": parcel_id,
        "parcel_info": {"teryt": "141201_1", "wojewodztwo": "mazowieckie",
                        "powiat": "przykładowy", "gmina": "przykładowa"},
        "confidence": {"score": 78.0, "score_low": 68.0, "score_high": 85.0,
                       "grade": "B", "label": "Poplon prawdopodobny"},
        "fused_verdict": {"status": "OBECNY"},
        "fused_category": "green",
        "category": "green",
        "category_reasons": ["NDVI powyżej progu w oknie oceny."],
        "peak_record": {"date": "2026-09-15", "ndvi": 0.55, "evi": 0.45,
                        "savi": 0.40, "ndre": 0.30, "ndwi": 0.30},
        "multi_records": [
            {"date": "2026-09-01", "ndvi": 0.30, "evi": 0.25, "savi": 0.20, "ndre": 0.18},
            {"date": "2026-09-15", "ndvi": 0.55, "evi": 0.45, "savi": 0.40, "ndre": 0.30},
        ],
        "biomass": {"biomass_t_ha": 2.1, "biomass_low_t_ha": 1.5, "biomass_high_t_ha": 2.8,
                    "carbon_co2eq_t_ha": 3.8, "carbon_co2eq_low_t_ha": 2.7,
                    "carbon_co2eq_high_t_ha": 5.1, "at_model_cap": False},
        "agro_events": [
            {"date": "2026-09-05", "type": "tillage", "confidence": "medium"},
        ],
        "weather_context": {"frost_days": 1, "dry_spell_days": 0, "first_frost_date": None},
        "weather_error": None,
        "terrain_context": {"elevation_m": 150.0, "slope_deg": 4.2,
                            "aspect_deg": 180.0, "erosion_risk": "niskie"},
        "growing_conditions": {"window_days": 14, "cumulative_gdd": 120.5,
                               "gdd_base_temp_c": 5.0, "total_precip_mm": 22.0,
                               "total_radiation_mj": 210.0},
    }


def test_build_parcel_report_zwraca_poprawny_pdf():
    data = _fake_check_multi_response()
    pdf_bytes = report_pdf.build_parcel_report(data)
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 500


def test_build_parcel_report_dziala_bez_biomasy_i_zdarzen():
    """Raport nie może się wywalić, gdy brakuje opcjonalnych sekcji (np.
    biomass=None, brak zdarzeń, brak konteksu pogodowego) — dokładnie tak,
    jak może wyglądać odpowiedź check_multi dla działki z samymi błędami."""
    data = _fake_check_multi_response()
    data["biomass"] = None
    data["agro_events"] = []
    data["weather_context"] = {}
    data["terrain_context"] = None
    data["growing_conditions"] = None
    data["peak_record"] = None
    data["multi_records"] = []
    pdf_bytes = report_pdf.build_parcel_report(data)
    assert pdf_bytes[:4] == b"%PDF"


def test_build_farmer_report_laczy_wiele_dzialek():
    data1 = _fake_check_multi_response("141201_1.0001.6509")
    data2 = _fake_check_multi_response("141201_1.0001.6510")
    pdf_bytes = report_pdf.build_farmer_report(
        [data1, data2], farmer_name="Jan Kowalski", farmer_phone="600100200")
    assert pdf_bytes[:4] == b"%PDF"
    # Zbiorczy raport dla 2 działek (z wykresami matplotlib) musi być
    # zauważalnie większy niż raport pojedynczej działki.
    single = report_pdf.build_parcel_report(data1)
    assert len(pdf_bytes) > len(single)
