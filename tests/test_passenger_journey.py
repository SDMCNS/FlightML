"""
Unit tests for the Passenger Journey & Public Travel Copilot engine.
"""
from __future__ import annotations

import json
import pytest

from flightml.data.dataset import FlightDataset
from flightml.eval.passenger_journey import (
    AirportWeatherForecast,
    ConnectionRisk,
    FlightLegRisk,
    PassengerItineraryAssessment,
    PassengerJourneyEngine,
    PassengerRightsAssessment,
)
from flightml.mcp.server import create_mcp_server


@pytest.fixture(scope="module")
def dataset() -> FlightDataset:
    return FlightDataset()


def test_passenger_itinerary_multi_leg(dataset: FlightDataset):
    engine = PassengerJourneyEngine(dataset)
    res = engine.assess_itinerary(
        origin="EINN",
        destination="EBBR",
        hub="EGLL",
        day_of_week="tuesday",
        time_window="morning",
        scheduled_layover_min=75,
        inbound_op="EIN",
        outbound_op="BAW",
    )

    assert isinstance(res, PassengerItineraryAssessment)
    assert res.origin == "EINN"
    assert res.destination == "EBBR"
    assert res.hub == "EGLL"
    assert res.day_of_week == "Tuesday"
    assert len(res.legs) == 2

    # Leg 1: EINN -> EGLL
    leg1 = res.legs[0]
    assert leg1.origin == "EINN"
    assert leg1.destination == "EGLL"
    assert leg1.dominant_operator in ("EIN", "BAW")
    assert leg1.mean_arr_delay_min is not None
    assert leg1.prob_delay_gt_15m_pct >= 0.0
    assert leg1.origin_weather.icao == "EINN"
    assert leg1.origin_weather.temperature_c is not None
    assert leg1.weather_risk_multiplier >= 1.0

    # Leg 2: EGLL -> EBBR
    leg2 = res.legs[1]
    assert leg2.origin == "EGLL"
    assert leg2.destination == "EBBR"
    assert leg2.dominant_operator in ("BAW", "BEL")

    # Connection Risk at EGLL
    assert res.connection is not None
    c = res.connection
    assert c.hub == "EGLL"
    assert c.scheduled_layover_min == 75
    assert c.minimum_connection_time_min >= 60
    assert c.misconnect_probability_pct > 0.0
    assert c.misconnect_risk_level in ("SECURE", "TIGHT", "HIGH_RISK", "CRITICAL")
    assert "Terminal" in c.terminal_transfer_note

    # Passenger Rights
    pr = res.passenger_rights
    assert pr.statutory_compensation_amount_eur == 250.0  # < 1500 km
    assert pr.delay_threshold_hours == 3.0
    assert len(pr.duty_of_care_entitlements) > 0

    # Departure Waves
    assert len(res.departure_waves) == 4
    first_wave = res.departure_waves[0]
    assert "Early Morning" in first_wave.wave_name
    assert first_wave.on_time_pct >= 0.0

    # Recommendations
    assert len(res.actionable_recommendations) >= 3

    # Dictionary conversion
    d = res.to_dict()
    assert d["origin"] == "EINN"
    assert d["destination"] == "EBBR"
    assert len(d["legs"]) == 2


def test_passenger_itinerary_direct(dataset: FlightDataset):
    engine = PassengerJourneyEngine(dataset)
    res = engine.assess_itinerary(
        origin="EINN",
        destination="EGLL",
        hub=None,
        day_of_week="friday",
        time_window="evening",
    )

    assert isinstance(res, PassengerItineraryAssessment)
    assert res.hub is None
    assert len(res.legs) == 1
    assert res.connection is None
    assert res.legs[0].origin == "EINN"
    assert res.legs[0].destination == "EGLL"
    assert res.passenger_rights.statutory_compensation_amount_eur == 250.0


def test_connection_tightness_sensitivity(dataset: FlightDataset):
    engine = PassengerJourneyEngine(dataset)

    # Tight 45 min connection at Heathrow (below 60-90 min MCT)
    res_tight = engine.assess_itinerary(
        origin="EINN",
        destination="EBBR",
        hub="EGLL",
        scheduled_layover_min=45,
    )
    # Relaxed 150 min connection at Heathrow (well above MCT)
    res_relaxed = engine.assess_itinerary(
        origin="EINN",
        destination="EBBR",
        hub="EGLL",
        scheduled_layover_min=150,
    )

    assert res_tight.connection is not None
    assert res_relaxed.connection is not None
    # Tight connection must have significantly higher misconnect risk than relaxed
    assert res_tight.connection.misconnect_probability_pct > res_relaxed.connection.misconnect_probability_pct
    assert res_tight.connection.slack_buffer_min < 0  # 45 min is in deficit against MCT


def test_passenger_rights_distance_tiers(dataset: FlightDataset):
    engine = PassengerJourneyEngine(dataset)

    # Tier 1: London to Paris (~340 km -> €250)
    pr_short = engine.calculate_passenger_rights("EGLL", "LFPG")
    assert pr_short.statutory_compensation_amount_eur == 250.0
    assert pr_short.total_journey_distance_km < 1500.0

    # Tier 2: London to Athens (~2400 km -> €400)
    pr_med = engine.calculate_passenger_rights("EGLL", "LGAV")
    assert pr_med.statutory_compensation_amount_eur == 400.0
    assert 1500.0 <= pr_med.total_journey_distance_km <= 3500.0


def test_mcp_passenger_itinerary_tool():
    server = create_mcp_server()
    # FastMCP stores registered tools in server._tool_manager._tools or server._tools
    tools = getattr(server, "_tool_manager", getattr(server, "_tools", None))
    assert tools is not None

    # Call tool function directly from module
    from flightml.mcp.server import get_dataset
    engine = PassengerJourneyEngine(get_dataset("202510"))
    res = engine.assess_itinerary(
        origin="EINN",
        destination="EBBR",
        hub="EGLL",
        day_of_week="tuesday",
        time_window="morning",
        scheduled_layover_min=75,
    )
    assert res.origin == "EINN"
    assert res.destination == "EBBR"
    assert res.connection is not None


def test_find_best_period_seasonality_and_recency():
    """Verify intelligent seasonal and modernity matching for Eurocontrol periods."""
    # November 2026 -> Closest modern autumn archive matching November is 202411
    p_nov26, reason_nov26 = FlightDataset.find_best_period(target_date="2026-11-17")
    assert p_nov26 == "202411"
    assert "November" in reason_nov26
    assert "2024" in reason_nov26

    # Exact calendar matches
    assert FlightDataset.find_best_period(target_date="2025-06-12")[0] == "202506"
    assert FlightDataset.find_best_period(target_date="2024-02-15")[0] == "202402"
    assert FlightDataset.find_best_period(target_date="2024-10-20")[0] == "202410"

    # Future dates map to matching seasonal modern archive
    assert FlightDataset.find_best_period(target_date="2026-07-20")[0] == "202507"
    assert FlightDataset.find_best_period(target_date="2026-01-10")[0] == "202501"

    # Historical baseline
    assert FlightDataset.find_best_period(target_date="2016-11-05")[0] == "201611"


def test_parse_flight_date_flexible_formats():
    """Verify flexible flight date parsing across ISO formats and relative phrases."""
    d_iso, src_iso = FlightDataset.parse_flight_date("2026-11-17")
    assert d_iso.isoformat() == "2026-11-17"
    assert src_iso == "explicit_date"

    d_euro, _ = FlightDataset.parse_flight_date("17/11/2026")
    assert d_euro.isoformat() == "2026-11-17"

    d_month, _ = FlightDataset.parse_flight_date("2025-06")
    assert d_month.year == 2025 and d_month.month == 6

    d_today, src_today = FlightDataset.parse_flight_date("today")
    assert src_today == "today"

    d_tomorrow, src_tomorrow = FlightDataset.parse_flight_date("tomorrow")
    assert src_tomorrow == "tomorrow"

    # Day of week fallback
    d_dow, src_dow = FlightDataset.parse_flight_date(None, day_of_week="tuesday")
    assert d_dow.weekday() == 1  # Tuesday is 1


def test_passenger_itinerary_auto_period_selection():
    """Verify that PassengerJourneyEngine automatically assigns the best fitting archive to the assessment."""
    engine = PassengerJourneyEngine()
    res = engine.assess_itinerary(
        origin="EHAM",
        destination="LFPG",
        target_date="2025-06-12",
        time_window="morning",
    )
    assert res.period == "202506"
    assert "202506" in res.period_selection_reason
    assert any("202506" in rec for rec in res.actionable_recommendations)


def test_passenger_itinerary_debug_audit():
    """Verify that debug=True populates full diagnostic accounting of records, weather, and MCT."""
    engine = PassengerJourneyEngine()
    res = engine.assess_itinerary(
        origin="EINN",
        destination="EBBR",
        hub="EGLL",
        target_date="2026-11-17",
        day_of_week="tuesday",
        time_window="morning",
        debug=True,
    )
    assert res.debug_audit is not None
    dbg = res.debug_audit
    assert "period_selection" in dbg
    assert "candidates" in dbg["period_selection"]
    assert len(dbg["period_selection"]["candidates"]) > 0

    assert "legs_audit" in dbg
    assert len(dbg["legs_audit"]) == 2
    leg1 = dbg["legs_audit"][0]
    assert "sample_records" in leg1
    assert "delay_percentiles" in leg1
    assert "operator_breakdown" in leg1
    assert "query_executed" in leg1

    assert "weather_audit" in dbg
    assert len(dbg["weather_audit"]) >= 2
    for w in dbg["weather_audit"]:
        assert "api_endpoint" in w
        assert "temperature_c" in w
        assert "wind_speed_kts" in w

    assert "connection_audit" in dbg
    assert dbg["connection_audit"]["hub"] == "EGLL"
    assert "effective_mct_min" in dbg["connection_audit"]

    assert "passenger_rights_audit" in dbg
    assert dbg["passenger_rights_audit"]["statutory_compensation_amount_eur"] == 250.0


def test_hub_mct_database_and_connection_types():
    """Verify global hub MCT profiles and automatic connection type inference (D_to_D, I_to_D, etc.)."""
    from flightml.eval.mct_database import get_hub_mct_profile, infer_connection_type, get_applicable_mct

    # 1. Profile retrieval
    jfk_prof = get_hub_mct_profile("KJFK")
    assert jfk_prof is not None
    assert jfk_prof.iata == "JFK"
    assert jfk_prof.d_to_d_min == 45
    assert jfk_prof.i_to_d_min == 135  # max of 120-135 min
    assert "AirTrain" in jfk_prof.notes

    lhr_prof = get_hub_mct_profile("LHR")
    assert lhr_prof is not None
    assert lhr_prof.icao == "EGLL"
    assert lhr_prof.d_to_d_min == 60
    assert lhr_prof.i_to_i_min == 90

    # 2. Connection type inference
    # BOS (US) -> JFK (US) -> LAX (US) => D_to_D
    c_type = infer_connection_type("US", "US", "US")
    assert c_type == "D_to_D"

    # LHR (GB) -> JFK (US) -> LAX (US) => I_to_D
    c_type = infer_connection_type("GB", "US", "US")
    assert c_type == "I_to_D"

    # BOS (US) -> JFK (US) -> LHR (GB) => D_to_I
    c_type = infer_connection_type("US", "US", "GB")
    assert c_type == "D_to_I"

    # CDG (FR) -> JFK (US) -> MEX (MX) => I_to_I
    c_type = infer_connection_type("FR", "US", "MX")
    assert c_type == "I_to_I"

    # 3. Applicable MCT
    mct_min, matched_type, matched_prof = get_applicable_mct("KJFK", origin_country="GB", dest_country="US")
    assert matched_type == "I_to_D"
    assert mct_min == 135
    assert matched_prof is not None
    assert "AirTrain" in matched_prof.notes

    # 4. Unlisted Hub Placeholder Common Sense Numbers
    unlisted_prof = get_hub_mct_profile("ZZZZ")
    assert unlisted_prof is not None
    assert unlisted_prof.is_placeholder is True
    assert unlisted_prof.d_to_d_min == 45
    assert unlisted_prof.d_to_i_min == 60
    assert unlisted_prof.i_to_d_min == 90
    assert unlisted_prof.i_to_i_min == 60
    assert "Standard industry MCT baseline" in unlisted_prof.notes

    # Unlisted hub with domestic connection
    mct_unlisted_d, type_d, prof_d = get_applicable_mct("ZZZZ", origin_country="US", dest_country="US", hub_country="US")
    assert type_d == "D_to_D"
    assert mct_unlisted_d == 45
    assert prof_d.is_placeholder is True

    # Unlisted hub with international to domestic connection
    mct_unlisted_id, type_id, prof_id = get_applicable_mct("ZZZZ", origin_country="GB", dest_country="US", hub_country="US")
    assert type_id == "I_to_D"
    assert mct_unlisted_id == 90



