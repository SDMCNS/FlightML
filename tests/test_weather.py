from flightml.weather.metar_parser import parse_metar_line, compute_flight_category
from flightml.weather.wind_calc import compute_runway_wind, find_optimal_runways


def test_compute_flight_category():
    assert compute_flight_category(visibility_sm=10.0, ceiling_ft=5000.0) == "VFR"
    assert compute_flight_category(visibility_sm=4.0, ceiling_ft=2500.0) == "MVFR"
    assert compute_flight_category(visibility_sm=2.0, ceiling_ft=800.0) == "IFR"
    assert compute_flight_category(visibility_sm=0.5, ceiling_ft=300.0) == "LIFR"


def test_parse_metar_line():
    raw = "KORD 010051Z 30018G25KT 10SM SCT040 OVC080 08/02 A2992"
    obs = parse_metar_line(raw)
    assert obs.station_id == "KORD"
    assert obs.wind_dir_deg == 300.0
    assert obs.wind_speed_kts == 18.0
    assert obs.wind_gust_kts == 25.0
    assert obs.visibility_sm == 10.0
    assert obs.ceiling_ft == 8000.0
    assert obs.flight_category == "VFR"
    assert obs.temp_c == 8.0
    assert obs.dewpoint_c == 2.0
    assert obs.altimeter_inhg == 29.92


def test_compute_runway_wind_aligned():
    # Direct headwind
    comp = compute_runway_wind(
        runway_ident="27",
        runway_heading_deg=270.0,
        wind_dir_deg=270.0,
        wind_speed_kts=20.0,
    )
    assert round(comp.crosswind_kts, 1) == 0.0
    assert round(comp.headwind_kts, 1) == 20.0
    assert not comp.is_tailwind
    assert not comp.crosswind_limit_exceeded


def test_compute_runway_wind_crosswind():
    # 90 degree crosswind
    comp = compute_runway_wind(
        runway_ident="36",
        runway_heading_deg=360.0,
        wind_dir_deg=270.0,
        wind_speed_kts=20.0,
    )
    assert round(comp.crosswind_kts, 1) == 20.0
    assert round(comp.headwind_kts, 1) == 0.0
    assert comp.crosswind_side == "LEFT"

