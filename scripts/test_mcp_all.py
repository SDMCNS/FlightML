import json
from flightml.mcp.server import create_mcp_server
server = create_mcp_server()
eid = 291262109

test_cases = [
    ('get_pru_scorecard', {'flight_id': eid}),
    ('get_flight_doc', {'flight_id': eid}),
    ('get_crco_charges', {'flight_id': eid}),
    ('get_fuel_and_emissions', {'flight_id': eid}),
    ('get_cdo_descent_profile', {'flight_id': eid}),
    ('get_cco_climb_profile', {'flight_id': eid}),
    ('get_fra_adherence', {'flight_id': eid}),
    ('diagnose_flight_bottlenecks', {'flight_id': eid}),
    ('get_fir_saturation', {'fir_id': 'EDUUUIR'}),
    ('get_flight_fir_dwell', {'flight_id': eid}),
    ('get_airport_taxi_out', {'airport_icao': 'EGLL'}),
    ('get_flight_taxi_out', {'flight_id': eid}),
    ('assess_runway_safety', {'flight_id': eid}),
    ('analyze_reactionary_delay', {'operator': 'BAW'}),
    ('estimate_radio_workload', {'airport_icao': 'EGLL'}),
    ('scan_airspace_conflicts', {'limit': 5}),
    ('get_jet_fuel_prices', {'period': '202510'}),
    ('locate_fir', {'lat': 51.5, 'lon': -0.1}),
    ('calculate_runway_winds', {'airport': 'EGLL', 'wind_dir': 270.0, 'wind_speed': 15.0}),
    ('predict_flight_delay', {'adep': 'EGLL', 'ades': 'EDDF'}),
    ('get_fra_boundary_rankings', {'limit': 5}),
    ('diagnose_fra_boundary', {'boundary_pair': 'EGTTUIR <-> LFFFUIR'}),
    ('get_fra_worst_flights', {'boundary_pair': 'EGTTUIR <-> LFFFUIR', 'limit': 3}),
    ('forecast_fra_benefits', {'scope': 'prime_only', 'annual_growth_rate_pct': 2.0}),
]

for tool_name, kwargs in test_cases:
    tool = server._tool_manager.get_tool(tool_name)
    res = tool.fn(**kwargs)
    parsed = json.loads(res)
    has_err = "error" in parsed if isinstance(parsed, dict) else False
    if has_err:
        print(f"❌ {tool_name:30s} -> ERROR: {parsed['error']}")
    else:
        print(f"✅ {tool_name:30s} -> OK")
