from __future__ import annotations

import os
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flightml.data.dataset import FlightDataset
from flightml.geo.fir_engine import FIREngine
from flightml.geo.trajectory import analyze_trajectory
from flightml.weather.metar_client import MetarClient
from flightml.weather.wind_calc import find_optimal_runways
from flightml.weather.delay_weather import analyze_delayed_flight_weather
from flightml.models.delay_predictor import DelayPredictor, MODEL_DIR
from flightml.models.reactionary_delay import ReactionaryDelayAnalyzer
from flightml.models.airline_stress import stress_test_airline
from flightml.viz.map_builder import create_trajectory_map
from flightml.eval.macro_comparison import compare_aviation_periods
from flightml.eval.cdo_analyzer import (
    analyze_cdo_descent,
    analyze_cco_climb,
    analyze_vertical_profile_efficiency,
)
from flightml.eval.pru_kpis import (
    compute_flight_pru_kpis,
    analyze_network_level_capping,
)
from flightml.geo.fra_analyzer import (
    analyze_flight_fra_adherence,
    compare_fra_evolution,
)
from flightml.features.radio_workload import estimate_airport_and_sector_radio_workload
from flightml.eval.crco_economics import (
    compute_flight_crco_charges,
    detect_route_charge_avoidance,
)
from flightml.geo.conflict_scanner import (
    scan_airspace_proximity_events,
    compute_sector_complexity_index,
    compute_hourly_grid_saturation,
)
from flightml.geo.rnp_navaid_analyzer import analyze_flight_navaids_and_rnp
from flightml.weather.runway_safety import assess_flight_runway_safety
from flightml.eval.contrail_climate import analyze_flight_wind_and_contrails
from flightml.sim.tail_swap_optimizer import optimize_hub_tail_swaps
from flightml.eval.passenger_journey import PassengerJourneyEngine
from flightml.eval.flight_twins import compare_twin_flights
from flightml.eval.diagnostics import diagnose_flight_bottlenecks
from flightml.data.archive_ingest import list_available_periods
from flightml.weather.network_weather_impact import analyze_network_weather_impact
from flightml.geo.airac_diff import compare_airac_cycles
from flightml.eval.recurring_series import analyze_recurring_flight_series
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
)
from flightml.features.aua_saturation import (
    calculate_aua_hourly_saturation,
    calculate_flight_aua_dwell_inefficiency,
)
from flightml.eval.surface_taxi import (
    analyze_airport_taxi_out_performance,
    analyze_flight_taxi_out,
)
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.aircraft_engine_perf import resolve_aircraft_engine_perf
from flightml.eval.multi_level_kpis import query_multi_level_kpis, get_dataset_date_bounds
from flightml.eval.jet_fuel_prices import (
    get_all_weekly_fuel_prices_df,
    get_weekly_fuel_price,
)
from flightml.eval.longitudinal import LongitudinalEngine
from flightml.models.airspace_forecaster import AirspaceForecaster
from flightml.eval.airline_economics import AirlineEconomicsEngine
from flightml.models.delay_contagion_graph import DelayContagionGraphEngine

app = FastAPI(
    title="FlightML API",
    description="Global Airspace Network Intelligence, 4D Trajectory Analytics, and Weather Fusion",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.isdir(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/cockpit")
    def cockpit_redirect():
        return RedirectResponse(url="/cockpit/")

    app.mount("/cockpit", StaticFiles(directory=frontend_dist, html=True), name="cockpit")

# Global singletons
_dataset: FlightDataset | None = None
_fir_engine: FIREngine | None = None
_metar_client: MetarClient | None = None
_predictor: DelayPredictor | None = None


def get_dataset() -> FlightDataset:
    global _dataset
    if _dataset is None:
        _dataset = FlightDataset()
    return _dataset


def get_fir_engine() -> FIREngine:
    global _fir_engine
    if _fir_engine is None:
        _fir_engine = FIREngine()
    return _fir_engine


def get_metar_client() -> MetarClient:
    global _metar_client
    if _metar_client is None:
        _metar_client = MetarClient()
    return _metar_client


def get_delay_predictor() -> DelayPredictor:
    global _predictor
    if _predictor is None:
        model_path = MODEL_DIR / "arr_delay_min_model.pkl"
        if model_path.exists():
            _predictor = DelayPredictor.load(model_path)
        else:
            ds = get_dataset()
            _predictor = DelayPredictor()
            df = _predictor.prepare_data(ds, sample_size=30000)
            _predictor.train(df)
            _predictor.save(model_path)
    return _predictor


class PredictDelayRequest(BaseModel):
    adep: str = Field(..., examples=["EGLL"], description="Departure airport ICAO code")
    ades: str = Field(..., examples=["KORD"], description="Destination airport ICAO code")
    ac_operator: str = Field(default="BAW", examples=["BAW"], description="Aircraft operator ICAO code")
    market_segment: str = Field(default="Traditional Scheduled", examples=["Traditional Scheduled"])
    requested_fl: float = Field(default=350.0, examples=[350.0], description="Cruising Flight Level")
    actual_distance_nm: float = Field(default=3400.0, examples=[3400.0], description="Distance in nautical miles")
    dep_hour: int = Field(default=10, ge=0, le=23, examples=[10], description="Scheduled departure hour (UTC)")
    dep_dayofweek: int = Field(default=2, ge=0, le=6, examples=[2], description="Day of week (0=Mon, 6=Sun)")
    filed_air_time_min: float = Field(default=480.0, examples=[480.0], description="Filed air time in minutes")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "FlightML API", "docs": "/docs", "cockpit": "/cockpit"}


@app.get("/api/v1/summary")
def get_summary() -> dict[str, Any]:
    ds = get_dataset()
    return ds.get_summary()


@app.get("/api/v1/airports/top")
def get_top_airports(limit: int = Query(default=15, ge=1, le=100)) -> list[dict[str, Any]]:
    ds = get_dataset()
    df = ds.get_top_airports(limit=limit)
    return df.to_dict(orient="records")


@app.get("/api/v1/airports/{ident}")
def get_airport(ident: str) -> dict[str, Any]:
    ds = get_dataset()
    ident = ident.upper()
    apt_df = ds.query("SELECT * FROM raw_airports WHERE ident = ?", [ident])
    if apt_df.empty:
        raise HTTPException(status_code=404, detail=f"Airport '{ident}' not found")

    apt_info = apt_df.iloc[0].to_dict()

    # Get runways
    runways_df = ds.query(
        "SELECT * FROM raw_runways WHERE airport_ident = ? ORDER BY length_ft DESC", [ident]
    )
    runways = runways_df.to_dict(orient="records")

    # Get recent weather via IEM ASOS client
    client = get_metar_client()
    metar = client.get_closest_metar(ident, "2016-11-01 12:00:00")

    runway_winds = []
    if metar and metar.wind_dir_deg is not None and metar.wind_speed_kts is not None:
        candidates = find_optimal_runways(
            runways, metar.wind_dir_deg, metar.wind_speed_kts, metar.wind_gust_kts
        )
        runway_winds = [c.to_dict() for c in candidates]

    return {
        "airport": apt_info,
        "runways": runways,
        "current_weather": metar.to_dict() if metar else None,
        "runway_wind_impact": runway_winds,
    }


@app.get("/api/v1/routes/top-delays")
def get_top_delay_routes(
    min_flights: int = Query(default=50, ge=10),
    limit: int = Query(default=15, ge=1, le=100),
) -> list[dict[str, Any]]:
    ds = get_dataset()
    df = ds.get_top_delayed_routes(min_flights=min_flights, limit=limit)
    return df.to_dict(orient="records")


@app.get("/api/v1/flights/search")
def search_flights_endpoint(
    q: str | None = Query(default=None, description="General search query (callsign, airport, operator, aircraft, ectrl_id)"),
    adep: str | None = Query(default=None, description="Departure airport ICAO (e.g. EGLL)"),
    ades: str | None = Query(default=None, description="Destination airport ICAO (e.g. KORD)"),
    operator: str | None = Query(default=None, description="Airline operator code (e.g. BAW)"),
    ac_type: str | None = Query(default=None, description="Aircraft type ICAO (e.g. B77W)"),
    tail: str | None = Query(default=None, description="Tail registration number"),
    time_of_day: str | None = Query(default=None, description="Time of day bank (Morning, Afternoon, Evening, Night)"),
    date_from: str | None = Query(default=None, description="Date from (YYYY-MM-DD)"),
    date_to: str | None = Query(default=None, description="Date to (YYYY-MM-DD)"),
    min_dist: float | None = Query(default=None, description="Min distance in nm"),
    max_dist: float | None = Query(default=None, description="Max distance in nm"),
    min_fl: float | None = Query(default=None, description="Min flight level"),
    max_fl: float | None = Query(default=None, description="Max flight level"),
    period: str = Query(default="202402", description="Data period ID"),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Search and filter flights with human-readable labels across any period."""
    ds = FlightDataset(period=period)

    # If general query string 'q' is passed, parse or apply multi-column match
    if q and q.strip():
        query_str = q.strip().upper()
        # Direct ECTRL ID lookup if integer
        if query_str.isdigit():
            single_f = ds.get_flight(int(query_str))
            if single_f:
                return [single_f]

        # Multi-attribute query across flights
        sql = """
            SELECT 
                ectrl_id,
                adep,
                ades,
                actual_off_block_time,
                actual_arrival_time,
                ac_operator,
                ac_type,
                ac_reg,
                requested_fl,
                actual_distance_nm,
                dep_delay_min,
                arr_delay_min,
                actual_air_time_min,
                flight_type,
                market_segment
            FROM flights
            WHERE 
                adep LIKE ? OR 
                ades LIKE ? OR 
                ac_operator LIKE ? OR 
                ac_type LIKE ? OR 
                ac_reg LIKE ?
            ORDER BY actual_off_block_time ASC
            LIMIT ?;
        """
        wild = f"%{query_str}%"
        df = ds.cursor.execute(sql, [wild, wild, wild, wild, wild, int(limit)]).df()
        if not df.empty:
            df["label"] = df.apply(ds.format_flight_label, axis=1)
        return df.to_dict(orient="records")

    df = ds.search_flights(
        adep=adep,
        ades=ades,
        ac_operator=operator,
        ac_type=ac_type,
        ac_reg=tail,
        time_of_day=time_of_day,
        date_from=date_from,
        date_to=date_to,
        min_distance_nm=min_dist,
        max_distance_nm=max_dist,
        min_fl=min_fl,
        max_fl=max_fl,
        limit=limit,
    )
    return df.to_dict(orient="records")


@app.get("/api/v1/flights/recurring")
def recurring_series_endpoint(
    adep: str = Query(..., description="Departure airport ICAO (e.g. EGLL)"),
    ades: str = Query(..., description="Destination airport ICAO (e.g. KORD)"),
    operator: str | None = Query(default=None, description="Airline operator code"),
    ac_type: str | None = Query(default=None, description="Aircraft type ICAO"),
    time_of_day: str | None = Query(default=None, description="Time of day bank"),
    period: str = Query(default="201611", description="Data period ID"),
    limit: int = Query(default=200, ge=10, le=1000),
) -> dict[str, Any]:
    """Longitudinal recurring series analysis comparing flights on a corridor day-over-day."""
    ds = FlightDataset(period=period)
    rep = analyze_recurring_flight_series(
        ds,
        adep=adep,
        ades=ades,
        operator=operator,
        ac_type=ac_type,
        time_of_day=time_of_day,
        limit=limit,
    )
    if not rep:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough recurring flights found on corridor {adep} -> {ades}",
        )
    return rep.to_dict()


@app.get("/api/v1/flights/{ectrl_id}")
def get_flight_details(
    ectrl_id: int,
    period: str | None = Query(default=None, description="Optional period ID (auto-detects if omitted)"),
) -> dict[str, Any]:
    if period:
        ds = FlightDataset(period=period)
    else:
        ds = get_dataset()

    flight = ds.get_flight(ectrl_id)
    if not flight and not period:
        # Auto-detect across all available periods
        for p in list_available_periods():
            if p != ds.period:
                test_ds = FlightDataset(period=p)
                test_f = test_ds.get_flight(ectrl_id)
                if test_f:
                    ds = test_ds
                    flight = test_f
                    break

    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight '{ectrl_id}' not found")

    trajs = ds.get_trajectory(ectrl_id)
    metrics = analyze_trajectory(trajs["actual"], trajs["planned"])

    actual_waypoints = trajs["actual"].to_dict(orient="records")
    planned_waypoints = trajs["planned"].to_dict(orient="records")

    return {
        "flight": flight,
        "metrics": metrics.to_dict(),
        "waypoints_actual_count": len(actual_waypoints),
        "waypoints_planned_count": len(planned_waypoints),
        "actual_waypoints": actual_waypoints,
        "actual_trajectory": actual_waypoints,
        "planned_waypoints": planned_waypoints,
        "planned_trajectory": planned_waypoints,
    }


@app.get("/api/v1/flights/{ectrl_id}/weather")
def get_delayed_flight_weather(ectrl_id: int) -> dict[str, Any]:
    """Retrieve on-demand IEM METAR observations for origin and destination of a delayed flight."""
    ds = get_dataset()
    client = get_metar_client()
    res = analyze_delayed_flight_weather(ds, client, ectrl_id)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res


@app.get("/api/v1/flights/{ectrl_id}/map", response_class=HTMLResponse)
def get_flight_map(ectrl_id: int) -> str:
    """Render interactive 4D trajectory, holding loops, and airports map."""
    ds = get_dataset()
    flight = ds.get_flight(ectrl_id)
    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight '{ectrl_id}' not found")

    trajs = ds.get_trajectory(ectrl_id)
    m = create_trajectory_map(flight, trajs["actual"], trajs["planned"])
    return m.get_root().render()


@app.get("/api/v1/reactionary")
def get_reactionary_delay_stats(
    operator: str | None = Query(default=None, description="Airline ICAO code (e.g. RYR, BAW, DLH)"),
    start_date: str = Query(default="2016-11-01"),
    end_date: str = Query(default="2016-11-04"),
) -> dict[str, Any]:
    """Calculate airline reactionary delay multiplier and ground buffer absorption rate."""
    ds = get_dataset()
    analyzer = ReactionaryDelayAnalyzer(ds)
    return analyzer.compute_network_reactionary_multiplier(
        operator=operator, date_start=start_date, date_end=end_date
    )


@app.post("/api/v1/predict/delay")
def predict_flight_delay(req: PredictDelayRequest) -> dict[str, Any]:
    predictor = get_delay_predictor()
    pred_delay = predictor.predict(req.model_dump())
    return {
        "flight_params": req.model_dump(),
        "predicted_arrival_delay_min": pred_delay,
        "delay_risk_level": "HIGH" if pred_delay > 15 else "MODERATE" if pred_delay > 5 else "LOW",
    }


@app.get("/api/v1/fir/locate")
def locate_fir(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    flight_level: float = Query(default=350.0, description="Flight Level"),
) -> dict[str, Any]:
    engine = get_fir_engine()
    firs = engine.locate_fir(lat, lon, flight_level)
    return {
        "latitude": lat,
        "longitude": lon,
        "flight_level": flight_level,
        "containing_firs": firs,
    }


@app.get("/api/v1/fir/saturation")
def get_fir_saturation(
    period: str = Query(default="202510", description="Eurocontrol period (e.g. 202510)"),
    fir_id: str | None = Query(default=None, description="Optional FIR ID filter (e.g. LTBBFIR)"),
    date: str | None = Query(default=None, description="Optional Date YYYY-MM-DD"),
    threshold: int = Query(default=40, ge=1, le=1000, description="Hourly flight saturation threshold"),
    top_n: int = Query(default=15, ge=1, le=100),
) -> dict[str, Any]:
    """Calculate hourly FIR sector saturation and identify congested European airspace chokepoints."""
    ds = FlightDataset(period=period)
    return calculate_fir_hourly_saturation(
        ds, fir_id=fir_id, date_str=date, threshold_flights_per_hr=threshold, top_n=top_n
    )


@app.get("/api/v1/periods")
def get_periods() -> dict[str, Any]:
    """List all available ingested historical Eurocontrol periods."""
    periods = list_available_periods()
    return {
        "periods": periods,
        "available_periods": periods,
        "total_periods": len(periods),
    }


@app.get("/api/v1/periods/compare")
def compare_periods_endpoint(
    p1: str = Query(default="201611", description="Baseline period (e.g. 201611)"),
    p2: str = Query(default="202510", description="Comparison period (e.g. 202510)"),
) -> dict[str, Any]:
    """Compare traffic growth, delay shifts, airline market share, and fleet modernization."""
    try:
        comp = compare_aviation_periods(period1=p1, period2=p2)
        return comp.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/flights/{ectrl_id}/fir-dwell")
def get_flight_fir_dwell(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Assess sector-by-sector en-route dwell efficiency, detecting airborne holding, vectoring, or shortcuts."""
    ds = FlightDataset(period=period)
    res = calculate_flight_fir_dwell_inefficiency(ds, ectrl_id=ectrl_id)
    if not res.get("has_data") and "not found" in res.get("message", "").lower():
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res


@app.get("/api/v1/fuel/prices")
def get_fuel_prices(
    period: str | None = Query(default=None, description="Optional period ID (201611, 202510)"),
) -> list[dict[str, Any]]:
    """Retrieve official weekly jet fuel spot prices (EIA / IATA benchmark) in €/gal, €/kg, and €/metric ton."""
    df = get_all_weekly_fuel_prices_df()
    if period:
        df = df[df["period_id"] == period.strip()]
    return df.to_dict(orient="records")


@app.get("/api/v1/flights/{ectrl_id}/fuel")
def get_flight_fuel_impact(
    ectrl_id: int,
    period: str = Query(default="201611", description="Eurocontrol period (e.g. 201611, 202510)"),
    price_per_kg: float | None = Query(default=None, description="Optional manual fuel price override in EUR/kg"),
    engine_override: str | None = Query(default=None, description="Optional aircraft engine model override (e.g. V2527-A5, CFM56-5B4, GE90-115B)"),
) -> dict[str, Any]:
    """Calculate excess fuel burn, CO2 emissions, and Euro (€) financial penalty with weekly EIA fuel spot price lookup and engine/altitude optimization."""
    ds = FlightDataset(period=period)
    try:
        impact = compute_flight_fuel_and_emissions(
            ds,
            ectrl_id=ectrl_id,
            fuel_price_per_kg=price_per_kg,
            engine_override=engine_override,
        )
        return impact.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/flights/{ectrl_id}/engines")
def get_flight_engine_options(
    ectrl_id: int,
    period: str = Query(default="201611", description="Eurocontrol period"),
) -> dict[str, Any]:
    """Retrieve available engine options, default engine, bypass ratio, pressure ratio, and ideal cruising altitude for a flight's aircraft."""
    ds = FlightDataset(period=period)
    flight = ds.get_flight(ectrl_id)
    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found")
    ac_type = str(flight.get("ac_type") or "A320").upper()
    perf = resolve_aircraft_engine_perf(ds, ac_type)
    return perf.to_dict()


@app.get("/api/v1/flights/{ectrl_id}/cdo")
def get_cdo_analysis(
    ectrl_id: int,
    period: str = Query(default="201611", description="Data period ID"),
) -> dict[str, Any]:
    """Continuous Descent Operations (CDO) profile and staircase level-off inefficiency analysis."""
    ds = FlightDataset(period=period)
    report = analyze_cdo_descent(ds, ectrl_id=ectrl_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"CDO descent profile not found for flight {ectrl_id}")
    return report.to_dict()


@app.get("/api/v1/flights/{ectrl_id}/diagnostics")
def get_flight_diagnostics(
    ectrl_id: int,
    period: str = Query(default="201611", description="Data period ID"),
) -> dict[str, Any]:
    """Multi-factor delay root-cause attribution (Reactionary vs Weather vs Holding vs Ground)."""
    ds = FlightDataset(period=period)
    client = get_metar_client()
    diag = diagnose_flight_bottlenecks(ds, client, ectrl_id=ectrl_id)
    return diag.to_dict()


@app.get("/api/v1/twins/compare")
def compare_twins_endpoint(
    adep: str = Query(..., description="Departure airport ICAO (e.g. RKSI)"),
    ades: str = Query(..., description="Destination airport ICAO (e.g. EGLL)"),
    ac_type: str = Query(default="B77W", description="Aircraft type ICAO (e.g. B77W)"),
    period: str = Query(default="201611", description="Data period ID"),
) -> dict[str, Any]:
    """Compare and contrast the most and least efficient flights between an airport pair."""
    ds = FlightDataset(period=period)
    comp = compare_twin_flights(ds, adep=adep.upper(), ades=ades.upper(), ac_type=ac_type.upper())
    if not comp:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough matching flights for twin comparison on {adep} -> {ades} ({ac_type})",
        )
    return comp.to_dict()


@app.get("/api/v1/stress-test")
def stress_test_endpoint(
    operator: str = Query(default="RYR", description="Airline operator ICAO (e.g. RYR, BAW)"),
    start_date: str = Query(default="2016-11-01"),
    end_date: str = Query(default="2016-11-04"),
    period: str = Query(default="201611", description="Data period ID"),
) -> dict[str, Any]:
    """Stress-test airline scheduling models under operational delay shock waves."""
    ds = FlightDataset(period=period)
    res = stress_test_airline(ds, operator=operator.upper(), start_date=start_date, end_date=end_date)
    return res.to_dict()


# -----------------------------------------------------------------------------
# Core Mathematical Library Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/v1/math/summary")
def get_math_library_summary() -> dict[str, Any]:
    """Retrieve catalog and function counts of the FlightML mathematical library."""
    import flightml.math as fmath
    return {
        "status": "ok",
        "total_functions_and_constants": len(fmath.__all__),
        "domains": [
            "Geodesy & Spherical Trigonometry",
            "Atmospheric Physics & ISA",
            "Aerodynamics & Kinematics",
            "Climate Microphysics & Emissions",
            "Aviation Economics & Passenger Liability",
            "Operational Efficiency & Navigation Infrastructure",
            "GNSS Kinematics & Electronic Warfare Shield",
        ],
        "exports": fmath.__all__,
    }


@app.get("/api/v1/math/isa")
def get_isa_math_endpoint(
    altitude_m: float = Query(10668.0, description="Altitude in meters (10668m = FL350)"),
) -> dict[str, Any]:
    """Compute exact International Standard Atmosphere (ISA) values for a given altitude."""
    import flightml.math as fmath
    fl = fmath.altitude_m_to_flight_level(altitude_m)
    return {
        "altitude_m": altitude_m,
        "flight_level": round(fl, 1),
        "temperature_c": round(fmath.isa_temperature_c(altitude_m), 2),
        "temperature_k": round(fmath.isa_temperature_k(altitude_m), 2),
        "pressure_hpa": round(fmath.isa_pressure_hpa(altitude_m), 2),
        "density_kg_m3": round(fmath.isa_density_kg_m3(altitude_m), 4),
        "speed_of_sound_kts": round(fmath.speed_of_sound_kts(altitude_m), 1),
        "speed_of_sound_mps": round(fmath.speed_of_sound_mps(altitude_m), 2),
    }


@app.get("/api/v1/math/geodesy/distance")
def get_geodesy_distance_endpoint(
    lat1: float = Query(51.4700, description="Latitude origin"),
    lon1: float = Query(-0.4543, description="Longitude origin"),
    lat2: float = Query(40.6413, description="Latitude destination"),
    lon2: float = Query(-73.7781, description="Longitude destination"),
) -> dict[str, Any]:
    """Compute Haversine distance, initial and final bearings between two coordinates."""
    import flightml.math as fmath
    dist_nm = fmath.haversine_distance_nm(lat1, lon1, lat2, lon2)
    dist_km = fmath.haversine_distance_km(lat1, lon1, lat2, lon2)
    init_b = fmath.initial_bearing_deg(lat1, lon1, lat2, lon2)
    final_b = fmath.final_bearing_deg(lat1, lon1, lat2, lon2)
    m_lat, m_lon = fmath.spherical_midpoint(lat1, lon1, lat2, lon2)
    return {
        "origin": [lat1, lon1],
        "destination": [lat2, lon2],
        "distance_nm": round(dist_nm, 2),
        "distance_km": round(dist_km, 2),
        "initial_bearing_deg": round(init_b, 1),
        "final_bearing_deg": round(final_b, 1),
        "spherical_midpoint": [round(m_lat, 4), round(m_lon, 4)],
    }


@app.get("/api/v1/math/climate/contrail")
def get_climate_contrail_endpoint(
    temp_c: float = Query(-45.0, description="Ambient air temperature in Celsius"),
    pressure_hpa: float = Query(250.0, description="Atmospheric pressure in hPa"),
    rh_ice_pct: float = Query(105.0, description="Relative humidity with respect to ice (%)"),
    propulsion_efficiency: float = Query(0.35, description="Overall propulsion efficiency (0.30 - 0.40)"),
) -> dict[str, Any]:
    """Evaluate Schmidt-Appleman Criterion and persistent contrail formation."""
    import flightml.math as fmath
    t_sac = fmath.schmidt_appleman_threshold_c(pressure_hpa, propulsion_efficiency)
    is_forming = fmath.is_persistent_contrail_forming(temp_c, pressure_hpa, rh_ice_pct, propulsion_efficiency)
    is_issr = fmath.is_ice_supersaturated(rh_ice_pct)
    return {
        "ambient_temp_c": temp_c,
        "pressure_hpa": pressure_hpa,
        "rh_ice_pct": rh_ice_pct,
        "t_sac_threshold_c": round(t_sac, 2),
        "is_ice_supersaturated_issr": is_issr,
        "is_persistent_contrail_forming": is_forming,
    }


@app.get("/api/v1/math/economics/crco")
def get_crco_charge_endpoint(
    distance_km: float = Query(1000.0, description="Charging distance in kilometers"),
    mtow_tonnes: float = Query(77.0, description="Aircraft Maximum Take-Off Weight (tonnes)"),
    unit_rate_eur: float = Query(75.0, description="National charging zone unit rate (EUR)"),
) -> dict[str, Any]:
    """Compute Eurocontrol CRCO En-Route route charge."""
    import flightml.math as fmath
    w_fac = fmath.crco_weight_factor(mtow_tonnes)
    d_fac = fmath.crco_distance_factor(distance_km)
    charge = fmath.crco_route_charge_eur(distance_km, mtow_tonnes, unit_rate_eur)
    return {
        "distance_km": distance_km,
        "mtow_tonnes": mtow_tonnes,
        "unit_rate_eur": unit_rate_eur,
        "distance_factor": round(d_fac, 3),
        "weight_factor": round(w_fac, 4),
        "route_charge_eur": round(charge, 2),
    }


# -----------------------------------------------------------------------------
# GNSS Jamming & Spoofing Network Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/v1/gnss/hotspots")
def get_gnss_hotspots() -> dict[str, Any]:
    """Retrieve known geopolitical electronic warfare and GNSS interference hotspot bounding boxes."""
    from flightml.eval.gnss_jamming_spoofing import GNSS_HOTSPOTS
    return GNSS_HOTSPOTS


@app.get("/api/v1/gnss/scan")
def scan_gnss_endpoint(
    period: str = Query("202402", description="Archive period"),
    min_score: float = Query(75.0, description="Minimum confirmation score threshold"),
    limit: int | None = Query(None, description="Pre-screening candidate flight limit (None for all flights in period)"),
    hotspot: str | None = Query(None, description="Optional hotspot zone filter key"),
) -> dict[str, Any]:
    """Scan the network for commercial flights displaying signatures of GNSS spoofing, speed teleportation, and vertical spikes."""
    from flightml.eval.gnss_jamming_spoofing import scan_network_gnss_interference
    ds = FlightDataset(period=period)
    rep = scan_network_gnss_interference(ds, sample_limit=limit, min_score=min_score, hotspot_filter=hotspot)
    return rep.to_dict()


@app.get("/api/v1/gnss/flight/{ectrl_id}")
def inspect_flight_gnss_endpoint(
    ectrl_id: int,
    period: str = Query("202402", description="Archive period"),
) -> dict[str, Any]:
    """Perform detailed kinematic envelope and terminal-proximity analysis for a single flight."""
    from flightml.eval.gnss_jamming_spoofing import analyze_flight_gnss_interference
    ds = FlightDataset(period=period)
    rep = analyze_flight_gnss_interference(ds, ectrl_id)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found or insufficient trajectory telemetry")
    return rep.to_dict()






@app.get("/api/v1/weather/impact")
def get_weather_impact_endpoint(
    start_date: str | None = Query(default=None, description="Start date (YYYY-MM-DD), defaults to period start"),
    end_date: str | None = Query(default=None, description="End date (YYYY-MM-DD), defaults to period end"),
    period: str = Query(default="201611", description="Data period ID"),
) -> dict[str, Any]:
    """Correlate historical European weather conditions with network-wide flight delays across the selected dataset time frame."""
    ds = FlightDataset(period=period)
    client = get_metar_client()
    try:
        report = analyze_network_weather_impact(ds, client, start_date=start_date, end_date=end_date)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/kpi/hierarchy")
def get_kpi_hierarchy_endpoint(
    level: str = Query(default="network", description="KPI tier: 'network', 'fleet', 'airport', or 'flight'"),
    group_by: str = Query(default="operator", description="Sub-aggregation: 'operator'/'aircraft' for fleet, 'departure'/'arrival' for airport"),
    start_date: str | None = Query(default=None, description="Start date (YYYY-MM-DD), optional"),
    end_date: str | None = Query(default=None, description="End date (YYYY-MM-DD), optional"),
    hour_start: int = Query(default=0, ge=0, le=23, description="Diurnal window start hour (0-23 UTC)"),
    hour_end: int = Query(default=23, ge=0, le=23, description="Diurnal window end hour (0-23 UTC)"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum entities/records to return"),
    period: str = Query(default="201611", description="Dataset period ID (e.g. 201611, 202510)"),
) -> dict[str, Any]:
    """Retrieve multi-level KPI scorecard across Flight, Fleet, Airport, or Network tiers with calendar and diurnal filtering."""
    ds = FlightDataset(period=period)
    try:
        summary = query_multi_level_kpis(
            ds=ds,
            level=level,
            group_by=group_by,
            start_date=start_date,
            end_date=end_date,
            hour_start=hour_start,
            hour_end=hour_end,
            limit=limit,
        )
        return summary.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/airac/diff")
def get_airac_diff_endpoint(
    c1: str = Query(default="2509", description="Baseline AIRAC cycle ID (e.g. 2509)"),
    c2: str = Query(default="2510", description="Target AIRAC cycle ID (e.g. 2510)"),
) -> dict[str, Any]:
    """Diff contiguous AIRAC cycle aeronautical configurations (airways and FIR airspace boundaries)."""
    try:
        report = compare_airac_cycles(cycle1=c1, cycle2=c2)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/aua/saturation")
def get_aua_saturation_endpoint(
    period: str = Query(default="202510", description="Eurocontrol period (e.g. 202510)"),
    aua_type: str | None = Query(default=None, description="Optional AUA type filter: TMA, CTA, UTA"),
    aua_id: str | None = Query(default=None, description="Optional AUA ID filter (e.g. EGTTCTA, EYVITMA)"),
    date: str | None = Query(default=None, description="Optional Date YYYY-MM-DD"),
    threshold: int = Query(default=30, ge=1, le=1000, description="Hourly flight saturation threshold"),
    top_n: int = Query(default=15, ge=1, le=100),
) -> dict[str, Any]:
    """Calculate hourly ATC Unit Airspace (TMA, CTA, UTA) saturation and sector congestion."""
    ds = FlightDataset(period=period)
    return calculate_aua_hourly_saturation(
        ds,
        aua_type=aua_type,
        aua_id=aua_id,
        date_str=date,
        threshold_flights_per_hr=threshold,
        top_n=top_n,
    )


@app.get("/api/v1/flights/{ectrl_id}/aua-dwell")
def get_flight_aua_dwell_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Assess sector-by-sector ATC Unit Airspace (TMA, CTA, UTA) dwell efficiency for a single flight."""
    ds = FlightDataset(period=period)
    res = calculate_flight_aua_dwell_inefficiency(ds, ectrl_id=ectrl_id)
    if not res.get("has_data") and "no aua traversal" in res.get("message", "").lower():
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res


@app.get("/api/v1/surface/taxi-out")
def get_airport_taxi_out_endpoint(
    airport: str | None = Query(default=None, description="Optional Airport ICAO (e.g. EGLL, LFPG)"),
    date: str | None = Query(default=None, description="Optional Date YYYY-MM-DD"),
    period: str = Query(default="202510", description="Eurocontrol period (e.g. 202510)"),
    min_flights: int = Query(default=50, ge=1, le=5000),
    top_n: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Compute ICAO KPI02 Additional Taxi-Out Time (ATOT), UTOT baselines, and LTO idle emissions across airports."""
    ds = FlightDataset(period=period)
    return analyze_airport_taxi_out_performance(
        ds, airport_icao=airport, date_str=date, min_flights=min_flights, top_n=top_n
    )


@app.get("/api/v1/flights/{ectrl_id}/taxi-out")
def get_flight_taxi_out_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Evaluate single-flight surface taxi-out queue delay (ICAO KPI02 ATOT) and ICAO EDB v32B idle emissions."""
    ds = FlightDataset(period=period)
    rep = analyze_flight_taxi_out(ds, ectrl_id=ectrl_id)
    if not rep:
        raise HTTPException(status_code=404, detail=f"TAXI_OUT record not found for flight {ectrl_id}")
    return rep.to_dict()


@app.get("/api/v1/flights/{ectrl_id}/pru-kpis")
def get_flight_pru_kpis_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Compute ICAO GANP & Eurocontrol PRU Scorecard (KPI04 KEP, KPI05 KEA, KPI08 ASMA, KPI19 Level-Capping)."""
    ds = FlightDataset(period=period)
    sc = compute_flight_pru_kpis(ds, ectrl_id=ectrl_id)
    if not sc:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return sc.to_dict()


@app.get("/api/v1/flights/{ectrl_id}/cco")
def get_flight_cco_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Analyze Continuous Climb Operations (KPI17 CCO) and departure climb level-offs."""
    ds = FlightDataset(period=period)
    rep = analyze_cco_climb(ds, ectrl_id=ectrl_id)
    if not rep:
        raise HTTPException(status_code=404, detail=f"CCO climb profile not found for flight {ectrl_id}")
    return rep.to_dict()


@app.get("/api/v1/flights/{ectrl_id}/vertical-profile")
def get_flight_vertical_profile_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Unified 3-phase vertical flight efficiency scorecard (CCO + Cruise Level-Capping + CDO)."""
    ds = FlightDataset(period=period)
    res = analyze_vertical_profile_efficiency(ds, ectrl_id=ectrl_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Vertical profile not found for flight {ectrl_id}")
    return res


@app.get("/api/v1/pru/level-capping")
def get_network_level_capping_endpoint(
    period: str = Query(default="202510", description="Data period ID"),
    operator: str | None = Query(default=None, description="Optional airline operator ICAO"),
    min_distance_nm: float = Query(default=250.0, ge=50.0),
    top_n: int = Query(default=15, ge=1, le=100),
) -> dict[str, Any]:
    """Scan the network for KPI19 Cruise Flight-Level Capping (actual max FL restricted >=2,000 ft below Requested FL)."""
    ds = FlightDataset(period=period)
    return analyze_network_level_capping(ds, operator=operator, min_distance_nm=min_distance_nm, top_n=top_n)


@app.get("/api/v1/flights/{ectrl_id}/fra")
def get_flight_fra_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
    snap_tolerance_nm: float = Query(default=2.5, ge=0.5, le=15.0),
) -> dict[str, Any]:
    """Snap 4D waypoints to ATS airways, 5-letter RNAV fixes, and radio navaids to compute Free Route Airspace (FRA) directness."""
    ds = FlightDataset(period=period)
    rep = analyze_flight_fra_adherence(ds, ectrl_id=ectrl_id, snap_tolerance_nm=snap_tolerance_nm)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return rep.to_dict()


@app.get("/api/v1/fra/evolution")
def get_fra_evolution_endpoint(
    p1: str = Query(default="201611", description="Baseline period ID"),
    p2: str = Query(default="202510", description="Comparison period ID"),
    sample_flights: int = Query(default=300, ge=50, le=2000),
) -> dict[str, Any]:
    """Compare Free Route Airspace (FRA) adoption between two historical Eurocontrol periods."""
    return compare_fra_evolution(period1=p1, period2=p2, sample_flights=sample_flights)


@app.get("/api/v1/airports/{ident}/radio-workload")
def get_airport_radio_workload_endpoint(
    ident: str,
    date: str | None = Query(default=None, description="Optional Date YYYY-MM-DD"),
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Estimate ATC Controller-Pilot VHF radio frequency workload and voice channel occupancy (%)."""
    ds = FlightDataset(period=period)
    return estimate_airport_and_sector_radio_workload(ds, airport_icao=ident, date_str=date)


@app.get("/api/v1/flights/{ectrl_id}/crco-cost")
def get_flight_crco_cost_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Compute Eurocontrol CRCO En-Route Navigation Charges (EUR) and Total Direct Operating Cost (EUR)."""
    ds = FlightDataset(period=period)
    rep = compute_flight_crco_charges(ds, ectrl_id=ectrl_id)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return rep.to_dict()


@app.get("/api/v1/economics/route-avoidance")
def get_route_charge_avoidance_endpoint(
    adep: str = Query(..., description="Origin airport ICAO"),
    ades: str = Query(..., description="Destination airport ICAO"),
    ac_type: str | None = Query(default=None, description="Optional aircraft type ICAO"),
    period: str = Query(default="202510", description="Data period ID"),
    limit: int = Query(default=25, ge=5, le=100),
) -> dict[str, Any]:
    """Compare CRCO route charges vs. fuel burn on a city pair to flag Route-Charge Avoidance detours."""
    ds = FlightDataset(period=period)
    return detect_route_charge_avoidance(ds, adep=adep, ades=ades, ac_type=ac_type, limit=limit)


@app.get("/api/v1/airspace/conflicts")
def get_airspace_conflicts_endpoint(
    date: str | None = Query(default=None, description="Date YYYY-MM-DD"),
    hour: int = Query(default=10, ge=0, le=23, description="UTC hour"),
    horiz_sep_nm: float = Query(default=5.0, ge=1.0, le=25.0),
    vert_sep_fl: float = Query(default=10.0, ge=5.0, le=40.0),
    period: str = Query(default="202510", description="Data period ID"),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    """Scan 4D trajectories for co-altitude aircraft proximity events and airspace complexity."""
    ds = FlightDataset(period=period)
    rep = scan_airspace_proximity_events(
        ds, date_str=date, hour=hour, horiz_sep_nm=horiz_sep_nm, vert_sep_fl=vert_sep_fl, limit=limit
    )
    comp = compute_sector_complexity_index(ds, date_str=rep.date_str, hour=hour)
    return {
        "proximity_report": rep.to_dict(),
        "sector_complexity": comp,
    }


@app.get("/api/v1/flights/{ectrl_id}/runway-safety")
def get_flight_runway_safety_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Assess arrival runway braking action, density altitude, crosswind limits, and landing distance safety margin."""
    ds = FlightDataset(period=period)
    rep = assess_flight_runway_safety(ds, metar_client=None, ectrl_id=ectrl_id, fetch_live_metar=False)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return rep.to_dict()


@app.get("/api/v1/flights/{ectrl_id}/contrails")
def get_flight_contrails_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Evaluate upper-air wind-optimal routing (TAS vs GS) and persistent contrail ISSR formation & avoidance."""
    ds = FlightDataset(period=period)
    rep = analyze_flight_wind_and_contrails(ds, ectrl_id=ectrl_id)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return rep.to_dict()


@app.get("/api/v1/airspace/grid-saturation")
def get_airspace_grid_saturation_endpoint(
    hour: int = Query(default=10, ge=0, le=23, description="UTC hour of day (0-23) averaged across the period"),
    period: str = Query(default="202510", description="Data period ID"),
    grid_size: float = Query(default=5.0, ge=0.5, le=10.0, description="Square resolution in degrees (from 0.5 to 5.0)"),
) -> dict[str, Any]:
    """Compute average hourly flight density and complexity per geographic square across the period."""
    ds = FlightDataset(period=period)
    return compute_hourly_grid_saturation(ds, hour_utc=hour, grid_size_deg=grid_size)


@app.get("/api/v1/flights/{ectrl_id}/navaids-rnp")
def get_flight_navaids_rnp_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Data period ID"),
    corridor_nm: float = Query(default=35.0, ge=5.0, le=100.0, description="Corridor half-width (nm) for likely used navaids"),
) -> dict[str, Any]:
    """Analyze likely used radio navaids (DME, VOR, ILS/LOC, NDB) along a flight's 4D trajectory and point-by-point anticipated RNP/RNAV capability."""
    ds = FlightDataset(period=period)
    rep = analyze_flight_navaids_and_rnp(ds, ectrl_id=ectrl_id, corridor_half_width_nm=corridor_nm)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return rep.to_dict()


@app.get("/api/v1/sim/tail-swaps")
def get_tail_swaps_endpoint(
    operator: str | None = Query(default=None, description="Optional airline ICAO code (omit for network-wide scan)"),
    airport: str | None = Query(default=None, description="Optional hub airport ICAO"),
    date: str | None = Query(default=None, description="Optional date YYYY-MM-DD"),
    period: str = Query(default="202510", description="Data period ID"),
) -> dict[str, Any]:
    """Prescriptive AI optimizer recommending same-type aircraft tail swaps at hubs to absorb reactionary delays."""
    ds = FlightDataset(period=period)
    res = optimize_hub_tail_swaps(ds, operator=operator, hub_airport=airport, date_str=date)
    return res.to_dict()





# ----------------- v3 Endpoints: Longitudinal, Predictive & Economics -----------------

@app.get("/api/v3/longitudinal/timeline")
def get_longitudinal_timeline_endpoint() -> dict[str, Any]:
    """Retrieve multi-period macro evolution timeline across all ingested Eurocontrol archives."""
    engine = LongitudinalEngine()
    tl_df = engine.get_timeline_dataframe()
    return {
        "periods_count": len(tl_df),
        "timeline": tl_df.to_dict(orient="records"),
    }


@app.get("/api/v3/longitudinal/compare")
def get_longitudinal_compare_endpoint(
    baseline: str = Query(default="202410", description="Baseline period ID (e.g. 202410)"),
    target: str = Query(default="202510", description="Target period ID (e.g. 202510)"),
) -> dict[str, Any]:
    """Controlled Year-over-Year (YoY) comparison of traffic, delays, fleet shifts, and city pairs."""
    engine = LongitudinalEngine()
    yoy = engine.compare_yoy(baseline, target)
    return yoy.to_dict()


@app.get("/api/v3/longitudinal/seasonality")
def get_longitudinal_seasonality_endpoint(
    year: int = Query(default=2024, description="Analysis year (e.g. 2024 or 2025)"),
) -> dict[str, Any]:
    """Three-season network decomposition (Winter, Summer, Autumn) with surge rates and delay elasticity."""
    engine = LongitudinalEngine()
    decomp = engine.decompose_seasonality(year)
    return decomp.to_dict()


@app.get("/api/v3/predict/airspace")
def get_predict_airspace_endpoint(
    year: int = Query(default=2026, description="Forecast horizon year"),
    season: str = Query(default="summer", description="Season archetype ('summer', 'winter', 'autumn')"),
    growth: float = Query(default=3.5, description="Projected traffic growth rate (%)"),
    grid_size: float = Query(default=1.0, ge=0.5, le=5.0, description="Spatial grid resolution in degrees"),
) -> dict[str, Any]:
    """Forward forecast of European airspace sector capacity breaches and spatial grid hotspots."""
    forecaster = AirspaceForecaster()
    rep = forecaster.generate_comprehensive_forecast(
        forecast_year=year,
        season=season,
        growth_scenario_pct=growth,
        grid_size_deg=grid_size,
    )
    return rep.to_dict()


@app.get("/api/v3/economics/flight-doc/{ectrl_id}")
def get_flight_doc_endpoint(
    ectrl_id: int,
    period: str = Query(default="202510", description="Eurocontrol period"),
    time_cost: float = Query(default=850.0, description="Cost of flight operational time (€/hour)"),
) -> dict[str, Any]:
    """Total Direct Operating Cost (DOC = Fuel € + Crew/Time € + CRCO Navigation Charges €) breakdown."""
    ds = FlightDataset(period=period)
    engine = AirlineEconomicsEngine(ds)
    doc = engine.evaluate_flight_doc(ectrl_id, cost_of_time_eur_per_hour=time_cost)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Flight {ectrl_id} not found in period {period}")
    return doc.to_dict()


@app.get("/api/v3/economics/avoidance-corridors")
def get_avoidance_corridors_endpoint(
    period: str = Query(default="202510", description="Eurocontrol period"),
    min_flights: int = Query(default=50, description="Minimum corridor flight volume"),
    top_n: int = Query(default=10, description="Top corridors to return"),
) -> dict[str, Any]:
    """Scan European network city pairs for systematic FIR route charge avoidance detours."""
    ds = FlightDataset(period=period)
    engine = AirlineEconomicsEngine(ds)
    avoid_list = engine.scan_network_route_charge_avoidance(min_corridor_flights=min_flights, top_n=top_n)
    return {
        "period": period,
        "avoidance_corridors_count": len(avoid_list),
        "corridors": [a.to_dict() for a in avoid_list],
    }


@app.get("/api/v3/economics/simulate-harmonization")
def get_simulate_harmonization_endpoint(
    period: str = Query(default="202510", description="Eurocontrol period"),
    rate: float = Query(default=72.50, description="Harmonized CRCO unit rate (€/SU)"),
    sample: int = Query(default=25, description="Sample size for Monte Carlo flight scaling"),
) -> dict[str, Any]:
    """Simulate Single European Sky (SES) unified CRCO en-route navigation rate policy impact."""
    ds = FlightDataset(period=period)
    engine = AirlineEconomicsEngine(ds)
    sim = engine.simulate_harmonized_eurocontrol_rate(harmonized_rate_eur=rate, sample_size=sample)
    return sim.to_dict()


@app.get("/api/v3/delay/spreaders")
def get_delay_spreaders_endpoint(
    period: str = Query(default="202510", description="Eurocontrol period"),
    date: str = Query(default="2025-10-15", description="Analysis date (YYYY-MM-DD)"),
    min_delay: float = Query(default=20.0, description="Minimum initial departure delay (min)"),
    top_n: int = Query(default=10, description="Number of super-spreader flights to return"),
) -> dict[str, Any]:
    """Identify super-spreader flights where morning departure delays infected downstream aircraft rotations."""
    ds = FlightDataset(period=period)
    engine = DelayContagionGraphEngine(ds)
    spreaders = engine.identify_super_spreaders(date_str=date, min_primary_delay_min=min_delay, top_n=top_n)
    return {
        "period": period,
        "date": date,
        "spreaders_count": len(spreaders),
        "super_spreaders": [s.to_dict() for s in spreaders],
    }


@app.get("/api/v3/delay/repeat-offenders/routes")
def get_repeat_offender_routes_endpoint(
    period: str = Query(default="202510", description="Eurocontrol period"),
    min_delay: float = Query(default=25.0, description="Minimum initial departure delay (min)"),
    min_propagated: float = Query(default=10.0, description="Minimum propagated reactionary delay (min)"),
    min_incidents: int = Query(default=3, description="Minimum delay contagion days across period"),
    top_n: int = Query(default=15, description="Number of repeat offender routes to return"),
) -> dict[str, Any]:
    """Scan reporting period for chronic repeat offender routes causing downstream delay contagion."""
    ds = FlightDataset(period=period)
    engine = DelayContagionGraphEngine(ds)
    start_date, end_date = get_dataset_date_bounds(ds)
    routes = engine.scan_repeat_offender_routes(
        min_primary_delay_min=min_delay,
        min_propagated_delay_min=min_propagated,
        min_incidents=min_incidents,
        top_n=top_n,
    )
    return {
        "period": period,
        "start_date": start_date,
        "end_date": end_date,
        "routes_count": len(routes),
        "repeat_offender_routes": [r.to_dict() for r in routes],
    }


@app.get("/api/v3/delay/repeat-offenders/operators")
def get_repeat_offender_operators_endpoint(
    period: str = Query(default="202510", description="Eurocontrol period"),
    min_delay: float = Query(default=25.0, description="Minimum initial departure delay (min)"),
    min_propagated: float = Query(default=10.0, description="Minimum propagated reactionary delay (min)"),
    min_spreaders: int = Query(default=5, description="Minimum root spreader flights across period"),
    top_n: int = Query(default=15, description="Number of repeat offender operators to return"),
) -> dict[str, Any]:
    """Scan reporting period for airline operators generating systemic delay contagion across airframe chains."""
    ds = FlightDataset(period=period)
    engine = DelayContagionGraphEngine(ds)
    start_date, end_date = get_dataset_date_bounds(ds)
    operators = engine.scan_repeat_offender_operators(
        min_primary_delay_min=min_delay,
        min_propagated_delay_min=min_propagated,
        min_spreader_flights=min_spreaders,
        top_n=top_n,
    )
    return {
        "period": period,
        "start_date": start_date,
        "end_date": end_date,
        "operators_count": len(operators),
        "repeat_offender_operators": [o.to_dict() for o in operators],
    }


@app.get("/api/v3/sim/network-shock")
def get_network_shock_endpoint(
    target: str = Query(default="EGLL", description="Disruption target (Airport e.g. EGLL or FIR e.g. EDUUUIR)"),
    target_type: str = Query(default="AIRPORT", description="Target type ('AIRPORT' or 'FIR')"),
    duration: float = Query(default=2.0, description="Disruption duration in hours"),
    capacity_cut: float = Query(default=50.0, description="Capacity reduction percentage (0 to 100)"),
    date: str = Query(default="2025-10-15", description="Simulation date (YYYY-MM-DD)"),
    period: str = Query(default="202510", description="Eurocontrol period"),
) -> dict[str, Any]:
    """Simulate systemic delay contagion from an airport runway or FIR capacity shock."""
    ds = FlightDataset(period=period)
    engine = DelayContagionGraphEngine(ds)
    shock = engine.simulate_network_shock(
        disruption_target=target,
        target_type=target_type,
        duration_hours=duration,
        capacity_cut_pct=capacity_cut,
        date_str=date,
    )
    return shock.to_dict()


# =============================================================================
# PASSENGER JOURNEY & PUBLIC TRAVEL RISK ENDPOINTS
# =============================================================================

@app.get("/api/v1/passenger/itinerary-risk")
def get_passenger_itinerary_risk_endpoint(
    origin: str = Query(default="EINN", description="Departure airport 4-letter ICAO code (e.g. EINN, EHAM)"),
    destination: str = Query(default="EBBR", description="Arrival airport 4-letter ICAO code (e.g. EBBR, EGLL)"),
    hub: str | None = Query(default=None, description="Optional connection hub 4-letter ICAO code (leave empty for direct flights)"),
    day_of_week: str = Query(default="tuesday", description="Day of week (e.g. tuesday, monday, friday)"),
    time_window: str = Query(default="morning", description="Departure wave ('morning', 'midday', 'afternoon', 'evening')"),
    scheduled_layover_min: int = Query(default=75, description="Scheduled layover duration in minutes at transfer hub"),
    date: str | None = Query(default=None, description="Target flight date YYYY-MM-DD (defaults to upcoming date/live 7-day forecast)"),
    period: str = Query(default="auto", description="Eurocontrol baseline data period (default: 'auto' to select best fitting archive for flight date)"),
    debug: bool = Query(default=False, description="Include detailed diagnostic audit accounting of records, weather endpoints, and formulas"),
) -> dict[str, Any]:
    """Assess flight itinerary delay risk, missed connection probability, live weather impact, and EU261 statutory compensation for the travelling public."""
    parsed_date, _ = FlightDataset.parse_flight_date(date, day_of_week)
    resolved_date_str = parsed_date.isoformat()

    if str(period).lower().strip() in ("auto", "none", ""):
        resolved_period, _ = FlightDataset.find_best_period(target_date=resolved_date_str, day_of_week=day_of_week)
    else:
        resolved_period = str(period).strip()

    ds = FlightDataset(period=resolved_period)
    engine = PassengerJourneyEngine(ds)
    clean_hub = hub.strip().upper() if hub and hub.strip().upper() not in (origin.strip().upper(), destination.strip().upper(), "NONE", "") else None
    res = engine.assess_itinerary(
        origin=origin.strip().upper(),
        destination=destination.strip().upper(),
        hub=clean_hub,
        day_of_week=day_of_week,
        time_window=time_window,
        scheduled_layover_min=scheduled_layover_min,
        target_date=resolved_date_str,
        period=resolved_period,
        debug=debug,
    )
    return res.to_dict()


@app.get("/api/v1/passenger/mct/{hub_ident}")
def get_hub_mct_endpoint(
    hub_ident: str,
    fallback_to_default: bool = Query(True, description="Fallback to common-sense baseline if unlisted"),
) -> dict[str, Any]:
    """Retrieve minimum connection time (MCT) profile for a hub across D_to_D, D_to_I, I_to_D, and I_to_I."""
    from flightml.eval.mct_database import resolve_hub_mct
    profile = resolve_hub_mct(hub_ident, fallback_to_default=fallback_to_default)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Hub MCT profile not found for '{hub_ident}'")
    return profile.to_dict()


@app.get("/api/v1/passenger/mct-database")
def get_all_mct_database_endpoint() -> dict[str, Any]:
    """Retrieve complete database of standardized global and European hub Minimum Connection Times."""
    from flightml.eval.mct_database import RAW_HUB_MCT_DATA, HUB_MCT_INDEX, create_placeholder_mct_profile
    return {
        "total_hubs": len(RAW_HUB_MCT_DATA),
        "hubs": RAW_HUB_MCT_DATA,
        "placeholder_defaults": create_placeholder_mct_profile("DEFAULT").to_dict(),
        "supported_codes": sorted(list(HUB_MCT_INDEX.keys())),
    }





