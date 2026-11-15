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
