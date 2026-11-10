from __future__ import annotations

from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
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
from flightml.eval.cdo_analyzer import analyze_cdo_descent
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
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.jet_fuel_prices import (
    get_all_weekly_fuel_prices_df,
    get_weekly_fuel_price,
)

app = FastAPI(
    title="FlightML API",
    description="Global Airspace Network Intelligence, 4D Trajectory Analytics, and Weather Fusion",
    version="0.1.0",
)

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
    return {"status": "ok", "service": "FlightML API", "docs": "/docs"}


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
    period: str = Query(default="201611", description="Data period ID"),
    limit: int = Query(default=25, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Search and filter flights with human-readable labels across any period."""
    ds = FlightDataset(period=period)
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
def get_flight_details(ectrl_id: int) -> dict[str, Any]:
    ds = get_dataset()
    flight = ds.get_flight(ectrl_id)
    if not flight:
        raise HTTPException(status_code=404, detail=f"Flight '{ectrl_id}' not found")

    trajs = ds.get_trajectory(ectrl_id)
    metrics = analyze_trajectory(trajs["actual"], trajs["planned"])

    return {
        "flight": flight,
        "metrics": metrics.to_dict(),
        "waypoints_actual_count": len(trajs["actual"]),
        "waypoints_planned_count": len(trajs["planned"]),
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
def get_periods() -> dict[str, list[str]]:
    """List all available ingested historical Eurocontrol periods."""
    return {"available_periods": list_available_periods()}


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
    """Retrieve official weekly jet fuel spot prices (EIA / IATA benchmark) in $/gal, $/kg, and $/metric ton."""
    df = get_all_weekly_fuel_prices_df()
    if period:
        df = df[df["period_id"] == period.strip()]
    return df.to_dict(orient="records")


@app.get("/api/v1/flights/{ectrl_id}/fuel")
def get_flight_fuel_impact(
    ectrl_id: int,
    period: str = Query(default="201611", description="Eurocontrol period (e.g. 201611, 202510)"),
    price_per_kg: float | None = Query(default=None, description="Optional manual fuel price override in USD/kg"),
) -> dict[str, Any]:
    """Calculate excess fuel burn, CO2 emissions, and financial penalty with weekly EIA fuel spot price lookup."""
    ds = FlightDataset(period=period)
    try:
        impact = compute_flight_fuel_and_emissions(ds, ectrl_id=ectrl_id, fuel_price_per_kg=price_per_kg)
        return impact.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


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


@app.get("/api/v1/weather/impact")
def get_weather_impact_endpoint(
    start_date: str = Query(default="2016-11-01", description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(default="2016-11-05", description="End date (YYYY-MM-DD)"),
    period: str = Query(default="201611", description="Data period ID"),
) -> dict[str, Any]:
    """Correlate historical European weather conditions with network-wide flight delays."""
    ds = FlightDataset(period=period)
    client = get_metar_client()
    try:
        report = analyze_network_weather_impact(ds, client, start_date=start_date, end_date=end_date)
        return report.to_dict()
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
