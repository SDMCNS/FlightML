"""Tests for FlightML Model Context Protocol (MCP) server integration."""
from __future__ import annotations

import json
import pytest
from flightml.mcp.server import create_mcp_server


@pytest.mark.asyncio
async def test_mcp_server_registration() -> None:
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    assert len(tools) >= 8
    assert "get_flight_details" in tool_names
    assert "search_flights" in tool_names
    assert "find_operational_twins" in tool_names
    assert "simulate_delay_contagion" in tool_names
    assert "simulate_fir_closure" in tool_names
    assert "simulate_marl_slot_gaming" in tool_names
    assert "evaluate_contrail_climate_forcing" in tool_names
    assert "get_airport_capacity" in tool_names

    prompts = await server.list_prompts()
    prompt_names = [p.name for p in prompts]
    assert "analyze_airport_delay_shock" in prompt_names


@pytest.mark.asyncio
async def test_mcp_tool_get_airport_capacity() -> None:
    server = create_mcp_server()
    res = await server.call_tool("get_airport_capacity", {"airport_icao": "EGLL"})
    assert len(res) > 0
    text_content = res[0][0].text
    data = json.loads(text_content)
    assert data["airport_icao"] == "EGLL"
    assert data["nominal_hourly_capacity_movements"] == 90


@pytest.mark.asyncio
async def test_mcp_tool_search_and_get_flight() -> None:
    server = create_mcp_server()
    search_res = await server.call_tool("search_flights", {"ades": "EGLL", "limit": 2})
    search_data = json.loads(search_res[0][0].text)
    assert search_data["total_matches"] > 0
    first_fid = search_data["flights"][0]["ectrl_id"]

    # Now get details by flight ID
    detail_res = await server.call_tool("get_flight_details", {"flight_id": first_fid})
    detail_data = json.loads(detail_res[0][0].text)
    assert str(detail_data["ectrl_id"]) == str(first_fid)
    assert detail_data["ades"] == "EGLL"


@pytest.mark.asyncio
async def test_mcp_tool_contrail_forcing() -> None:
    server = create_mcp_server()
    res = await server.call_tool(
        "evaluate_contrail_climate_forcing",
        {"temperature_c": -54.0, "rh_ice_pct": 120.0, "flight_level": 360.0},
    )
    data = json.loads(res[0][0].text)
    assert data["temperature_c"] == -54.0
    assert data["is_persistent_issr"] is True
    assert data["instantaneous_rf_mw_m2"] > 0.0


@pytest.mark.asyncio
async def test_mcp_tool_operational_twins() -> None:
    server = create_mcp_server()
    search_res = await server.call_tool("search_flights", {"ades": "EGLL", "limit": 1})
    search_data = json.loads(search_res[0][0].text)
    first_fid = search_data["flights"][0]["ectrl_id"]

    twins_res = await server.call_tool("find_operational_twins", {"flight_id": first_fid, "top_k": 2})
    twins_data = json.loads(twins_res[0][0].text)
    assert "twins" in twins_data
    assert len(twins_data["twins"]) > 0


@pytest.mark.asyncio
async def test_mcp_tool_fir_closure() -> None:
    server = create_mcp_server()
    res = await server.call_tool(
        "simulate_fir_closure",
        {"target_fir": "LSASFIR", "duration_hours": 2.0},
    )
    data = json.loads(res[0][0].text)
    assert data["target_fir"] == "LSASFIR"
    assert data["total_flights_impacted"] > 0
    assert data["rerouted_flights_count"] > 0


@pytest.mark.asyncio
async def test_mcp_tool_marl_slot_gaming() -> None:
    server = create_mcp_server()
    res = await server.call_tool(
        "simulate_marl_slot_gaming",
        {"regulated_airport": "EGLL", "capacity_cut_pct": 30.0, "duration_hours": 2.0},
    )
    data = json.loads(res[0][0].text)
    assert data["regulated_airport"] == "EGLL"
    assert len(data["airline_strategies"]) == 4
    assert len(data["decision_log"]) >= 8


@pytest.mark.asyncio
async def test_mcp_tool_assess_passenger_itinerary():
    server = create_mcp_server()
    res = await server.call_tool(
        "assess_passenger_itinerary",
        {
            "origin": "EINN",
            "destination": "EBBR",
            "hub": "EGLL",
            "day_of_week": "tuesday",
            "time_window": "morning",
            "scheduled_layover_min": 75,
        },
    )
    data = json.loads(res[0][0].text)
    assert data["origin"] == "EINN"
    assert data["destination"] == "EBBR"
    assert data["hub"] == "EGLL"
    assert len(data["legs"]) == 2
    assert data["connection"]["hub"] == "EGLL"
    assert data["passenger_rights"]["statutory_compensation_amount_eur"] == 250.0
