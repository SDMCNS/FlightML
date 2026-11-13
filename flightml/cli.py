from __future__ import annotations

from pathlib import Path
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
import uvicorn

from flightml.data.dataset import FlightDataset
from flightml.data.parquet_converter import convert_all
from flightml.geo.fir_engine import FIREngine
from flightml.geo.trajectory import analyze_trajectory
from flightml.weather.metar_client import MetarClient
from flightml.weather.wind_calc import find_optimal_runways
from flightml.weather.delay_weather import analyze_delayed_flight_weather
from flightml.models.delay_predictor import DelayPredictor, MODEL_DIR
from flightml.models.reactionary_delay import ReactionaryDelayAnalyzer
from flightml.models.gnn_delay import AirportGraphDelayModel
from flightml.features.sector_bottlenecks import analyze_fir_bottlenecks
from flightml.eval.diagnostics import diagnose_flight_bottlenecks
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.flight_twins import compare_twin_flights, render_twin_comparison_map
from flightml.models.airline_stress import stress_test_airline
from flightml.sim.what_if import simulate_turnaround_mitigation, simulate_runway_weather_scenario
from flightml.sim.gdp_optimizer import optimize_ground_delay_program
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
)
from flightml.weather.runway_safety import assess_flight_runway_safety
from flightml.eval.contrail_climate import analyze_flight_wind_and_contrails
from flightml.sim.tail_swap_optimizer import optimize_hub_tail_swaps
from flightml.data.archive_ingest import EurocontrolArchiveIngestor, list_available_periods
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
from flightml.eval.jet_fuel_prices import (
    get_all_weekly_fuel_prices_df,
    get_weekly_fuel_price,
)

app = typer.Typer(
    name="flightml",
    help="FlightML: Global Aviation Network Intelligence, 4D Trajectory ML & Weather Fusion",
    add_completion=False,
)
console = Console()


@app.command("summary")
def summary_cmd() -> None:
    """Display comprehensive dataset summary metrics."""
    with console.status("[cyan]Loading dataset metrics via DuckDB..."):
        ds = FlightDataset()
        stats = ds.get_summary()

    table = Table(title="✈️  FlightML Dataset Overview", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Processed Flights", f"{stats['total_flights']:,}")
    table.add_row("Unique Aircraft Operators", f"{stats['total_operators']:,}")
    table.add_row("Origin Airports (ADEP)", f"{stats['unique_dep_airports']:,}")
    table.add_row("Destination Airports (ADES)", f"{stats['unique_arr_airports']:,}")
    table.add_row("Date Coverage", stats["date_range"])
    table.add_row("Mean Departure Delay", f"{stats['avg_dep_delay_min']} min")
    table.add_row("Mean Arrival Delay", f"{stats['avg_arr_delay_min']} min")
    table.add_row("Airports Catalog", f"{stats['airports_catalog']:,}")
    table.add_row("Runways Catalog", f"{stats['runways_catalog']:,}")
    table.add_row("FIR Airspaces Catalog", f"{stats['fir_airspaces']:,}")

    console.print(table)


@app.command("convert")
def convert_cmd(
    include_trajectories: bool = typer.Option(
        True, "--include-trajectories", "-t", help="Convert large 4D trajectory CSVs"
    )
) -> None:
    """Convert CSV datasets to high-speed compressed Apache Parquet."""
    console.print("[bold yellow]Starting High-Speed Parquet Conversion...[/bold yellow]")
    convert_all(include_trajectories=include_trajectories)
    console.print("[bold green]Conversion completed successfully![/bold green]")


@app.command("top-airports")
def top_airports_cmd(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of airports to show")
) -> None:
    """List busiest airports by departures and average delay."""
    ds = FlightDataset()
    df = ds.get_top_airports(limit=limit)

    table = Table(title=f"Top {limit} Busiest Airports (Departures)", header_style="bold cyan")
    table.add_column("ICAO", style="bold")
    table.add_column("Name", style="white")
    table.add_column("City", style="dim")
    table.add_column("Country", style="dim")
    table.add_column("Departures", justify="right", style="green")
    table.add_column("Avg Dep Delay", justify="right", style="yellow")

    for _, row in df.iterrows():
        table.add_row(
            row["ident"],
            str(row["name"]),
            str(row["municipality"]),
            str(row["iso_country"]),
            f"{row['departure_count']:,}",
            f"{row['avg_dep_delay_min']} min",
        )
    console.print(table)


@app.command("top-delays")
def top_delays_cmd(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of routes to show"),
    min_flights: int = typer.Option(50, "--min-flights", "-m", help="Minimum flight threshold"),
) -> None:
    """Identify network routes with the worst average arrival delays."""
    ds = FlightDataset()
    df = ds.get_top_delayed_routes(min_flights=min_flights, limit=limit)

    table = Table(title=f"Worst Delayed Corridors (min {min_flights} flights)", header_style="bold red")
    table.add_column("ADEP", style="bold cyan")
    table.add_column("ADES", style="bold cyan")
    table.add_column("Flights", justify="right")
    table.add_column("Avg Dep Delay", justify="right", style="yellow")
    table.add_column("Avg Arr Delay", justify="right", style="bold red")
    table.add_column("Avg Dist (nm)", justify="right")

    for _, row in df.iterrows():
        table.add_row(
            row["adep"],
            row["ades"],
            str(row["flight_count"]),
            f"{row['avg_dep_delay']} min",
            f"{row['avg_arr_delay']} min",
            f"{row['avg_distance_nm']} nm",
        )
    console.print(table)


@app.command("trajectory")
def trajectory_cmd(
    ectrl_id: int = typer.Argument(..., help="ECTRL ID of the flight to analyze")
) -> None:
    """Analyze 4D trajectory, path extension ratio, and holding loops."""
    ds = FlightDataset()
    fl = ds.get_flight(ectrl_id)
    if not fl:
        console.print(f"[red]Flight with ECTRL ID {ectrl_id} not found![/red]")
        raise typer.Exit(code=1)

    trajs = ds.get_trajectory(ectrl_id)
    metrics = analyze_trajectory(trajs["actual"], trajs["planned"])

    console.print(f"\n[bold green]Flight Analysis:[/bold green] ECTRL {ectrl_id}")
    console.print(f"Route: [cyan]{fl['adep']}[/cyan] -> [cyan]{fl['ades']}[/cyan] | Operator: [magenta]{fl['ac_operator']}[/magenta] | Aircraft: [yellow]{fl['ac_type']}[/yellow]")
    console.print(f"Off-block Delay: [yellow]{fl['dep_delay_min']} min[/yellow] | Arrival Delay: [yellow]{fl['arr_delay_min']} min[/yellow]")

    table = Table(title="Trajectory Performance Metrics", header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Actual Distance Flown", f"{metrics.actual_distance_nm:.1f} nm")
    table.add_row("Planned Distance", f"{metrics.planned_distance_nm:.1f} nm")
    table.add_row("Great Circle Distance", f"{metrics.great_circle_distance_nm:.1f} nm")
    table.add_row("Horizontal Inefficiency Ratio", f"{metrics.horizontal_extension_ratio:.3f}x")
    table.add_row("Max Vertical Deviation", f"{metrics.max_vertical_deviation_fl:.0f} FL")
    table.add_row("Holding Patterns Detected", str(len(metrics.holding_events)))
    table.add_row("Total Holding Duration", f"{metrics.total_holding_duration_min:.1f} min")

    console.print(table)

    if metrics.holding_events:
        h_table = Table(title="Detected Airborne Holding Loops", header_style="bold yellow")
        h_table.add_column("Start Time")
        h_table.add_column("End Time")
        h_table.add_column("Duration")
        h_table.add_column("Center Lat / Lon")
        h_table.add_column("Flight Level")

        for h in metrics.holding_events:
            h_table.add_row(
                h.start_time,
                h.end_time,
                f"{h.duration_min:.1f} min",
                f"{h.center_lat:.3f}, {h.center_lon:.3f}",
                f"FL{h.flight_level:.0f}",
            )
        console.print(h_table)


@app.command("diagnose")
def diagnose_cmd(
    ectrl_id: int = typer.Argument(..., help="ECTRL ID of the flight to diagnose")
) -> None:
    """Diagnose multi-factor root causes of flight delay (Reactionary, Weather, Holding, Ground)."""
    ds = FlightDataset()
    client = MetarClient()
    with console.status("[cyan]Running multi-modal delay diagnostic engine..."):
        diag = diagnose_flight_bottlenecks(ds, client, ectrl_id)

    console.print(f"\n[bold]Flight Delay Diagnostic Report:[/bold] ECTRL {ectrl_id} ({diag.ac_operator} - {diag.ac_reg})")
    console.print(f"Route: [cyan]{diag.adep}[/cyan] -> [cyan]{diag.ades}[/cyan]")
    console.print(f"Total Arrival Delay: [bold yellow]{diag.total_arr_delay_min} mins[/bold yellow] | Departure Delay: [yellow]{diag.total_dep_delay_min} mins[/yellow]")

    table = Table(title="Delay Root Cause Decomposition", header_style="bold magenta")
    table.add_column("Factor", style="cyan")
    table.add_column("Impact", style="yellow")

    table.add_row("Reactionary Delay (Inbound Tail)", f"{diag.reactionary_delay_min} mins")
    table.add_row("Airborne Holding Delay", f"{diag.holding_delay_min} mins")
    table.add_row("Weather Attribution", diag.weather_attribution)
    table.add_row("Primary Bottleneck Factor", f"[bold red]{diag.primary_bottleneck_factor}[/bold red]")

    console.print(table)
    console.print(f"\n[bold green]Actionable Mitigation Recommendation:[/bold green]")
    console.print(f"[white]{diag.mitigation_recommendation}[/white]\n")


@app.command("bottlenecks")
def bottlenecks_cmd(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of FIR sectors to show")
) -> None:
    """Identify the most congested Flight Information Region (FIR) airspaces."""
    ds = FlightDataset()
    with console.status("[cyan]Analyzing European FIR traffic density and waypoint traversals..."):
        df = analyze_fir_bottlenecks(ds, limit=limit)

    table = Table(title=f"Top {limit} Congested European Airspace Sectors", header_style="bold red")
    table.add_column("Airspace ID (FIR)", style="bold cyan")
    table.add_column("Unique Flights", justify="right", style="green")
    table.add_column("Waypoint Density", justify="right", style="yellow")
    table.add_column("Vertical Strata (FL)", justify="center", style="dim")

    for _, row in df.iterrows():
        table.add_row(
            str(row["airspace_id"]),
            f"{int(row['unique_flights']):,}",
            f"{int(row['waypoint_traversals']):,}",
            f"FL{int(row['min_fl'])} - FL{int(row['max_fl'])}",
        )
    console.print(table)


@app.command("train-gnn")
def train_gnn_cmd(
    epochs: int = typer.Option(40, "--epochs", "-e", help="Training epochs"),
    top_n: int = typer.Option(20, "--top-n", "-n", help="Top airports to include in graph"),
) -> None:
    """Train the Spatio-Temporal Graph Neural Network (ST-GNN) on airport network graph."""
    ds = FlightDataset()
    model = AirportGraphDelayModel(top_n_airports=top_n)
    console.print(f"[bold cyan]Constructing airport network graph for top {top_n} hubs...[/bold cyan]")
    with console.status(f"[cyan]Training PyTorch ST-GNN on {model.device}..."):
        res = model.train_baseline(ds, epochs=epochs)

    console.print(f"[bold green]✓ ST-GNN Training Complete![/bold green]")
    console.print(f"Device: [cyan]{res['device']}[/cyan] | Final MSE Loss: [yellow]{res['final_loss']}[/yellow] across {res['epochs']} epochs")


@app.command("map")
def map_cmd(
    ectrl_id: int = typer.Argument(..., help="ECTRL ID of the flight to visualize"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save HTML map"),
) -> None:
    """Generate interactive 4D trajectory HTML map with holding stacks (unblocked tiles)."""
    ds = FlightDataset()
    fl = ds.get_flight(ectrl_id)
    if not fl:
        console.print(f"[red]Flight with ECTRL ID {ectrl_id} not found![/red]")
        raise typer.Exit(code=1)

    trajs = ds.get_trajectory(ectrl_id)
    out_file = output or f"flight_{ectrl_id}_map.html"
    create_trajectory_map(fl, trajs["actual"], trajs["planned"], output_html_path=out_file)
    console.print(f"[bold green]Interactive trajectory map created:[/bold green] [cyan]{out_file}[/cyan]")


@app.command("weather-delayed")
def weather_delayed_cmd(
    ectrl_id: int = typer.Argument(..., help="ECTRL ID of delayed flight")
) -> None:
    """Fetch on-demand IEM METAR weather and attribute delay root cause."""
    ds = FlightDataset()
    client = MetarClient()
    with console.status("[cyan]Retrieving on-demand METAR observations via IEM..."):
        res = analyze_delayed_flight_weather(ds, client, ectrl_id)

    if "error" in res:
        console.print(f"[red]{res['error']}[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Delayed Flight Weather Analysis:[/bold] ECTRL {ectrl_id}")
    console.print(f"Route: [cyan]{res['adep']}[/cyan] -> [cyan]{res['ades']}[/cyan] | Arr Delay: [yellow]{res['arr_delay_min']} min[/yellow]")

    if res["adverse_weather_detected"]:
        console.print("[bold red]⚠️ Adverse Weather Detected at Aerodromes:[/bold red]")
        for reason in res["weather_attribution_reasons"]:
            console.print(f" - [red]{reason}[/red]")
    else:
        console.print("[bold green]✓ Normal weather observed at endpoints (likely non-weather delay).[/bold green]")

    if res["arr_metar"]:
        console.print(f"Destination METAR ({res['ades']}): [dim]{res['arr_metar']['raw_metar']}[/dim]")


@app.command("reactionary")
def reactionary_cmd(
    operator: str = typer.Option(None, "--operator", "-op", help="Filter by airline (e.g. RYR, BAW, DLH)"),
    start: str = typer.Option("2016-11-01", "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2016-11-04", "--end", help="End date (YYYY-MM-DD)"),
) -> None:
    """Analyze aircraft tail rotations and reactionary delay propagation."""
    ds = FlightDataset()
    analyzer = ReactionaryDelayAnalyzer(ds)
    with console.status("[cyan]Analyzing aircraft rotations and turnaround delay propagation..."):
        res = analyzer.compute_network_reactionary_multiplier(operator=operator, date_start=start, date_end=end)

    if "error" in res:
        console.print(f"[red]{res['error']}[/red]")
        raise typer.Exit(code=1)

    title = f"Reactionary Delay Analysis ({operator.upper() if operator else 'All Carriers'})"
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Turnarounds Analyzed", f"{res['total_turnarounds_analyzed']:,}")
    table.add_row("Reactionary Incidents", f"{res['reactionary_delay_incidents']:,}")
    table.add_row("Reactionary Incident Rate", f"{res['reactionary_incident_rate']}%")
    table.add_row("Total Inbound Delay", f"{res['total_incoming_delay_min']:,} min")
    table.add_row("Propagated Downstream Delay", f"{res['total_propagated_delay_min']:,} min")
    table.add_row("Ground Buffer Absorbed Delay", f"{res['total_absorbed_delay_min']:,} min")
    table.add_row("Delay Propagation Ratio", f"{res['delay_propagation_ratio'] * 100:.1f}%")
    table.add_row("Buffer Absorption Rate", f"{res['ground_buffer_absorption_rate'] * 100:.1f}%")

    console.print(table)


@app.command("weather-winds")
def weather_winds_cmd(
    airport: str = typer.Option(..., "--airport", "-a", help="Airport ICAO code"),
    wind_dir: float = typer.Option(..., "--wind-dir", "-d", help="Wind direction in degrees (0-360)"),
    wind_speed: float = typer.Option(..., "--wind-speed", "-s", help="Wind speed in knots"),
    gust: float = typer.Option(None, "--gust", "-g", help="Wind gust in knots"),
) -> None:
    """Calculate crosswinds and determine optimal active runways."""
    ds = FlightDataset()
    airport = airport.upper()
    runways_df = ds.query("SELECT * FROM raw_runways WHERE airport_ident = ?", [airport])
    if runways_df.empty:
        console.print(f"[red]No runways cataloged for {airport}![/red]")
        raise typer.Exit(code=1)

    candidates = find_optimal_runways(runways_df.to_dict(orient="records"), wind_dir, wind_speed, gust)

    table = Table(title=f"Runway Wind Impact for {airport} (Winds: {wind_dir:.0f}° @ {wind_speed:.0f} kts)", header_style="bold cyan")
    table.add_column("Runway", style="bold")
    table.add_column("Heading", justify="right")
    table.add_column("Crosswind", justify="right")
    table.add_column("Side")
    table.add_column("Head/Tailwind", justify="right")
    table.add_column("Status", justify="center")

    for c in candidates:
        cw_color = "red" if c.crosswind_limit_exceeded else "yellow" if c.crosswind_kts > 15 else "green"
        status = "[red]EXCEEDED[/red]" if (c.crosswind_limit_exceeded or c.tailwind_limit_exceeded) else "[green]ACCEPTABLE[/green]"
        hw_desc = f"[red]{c.tailwind_kts:.1f} kts Tailwind[/red]" if c.is_tailwind else f"[green]{c.headwind_kts:.1f} kts Headwind[/green]"

        table.add_row(
            c.runway_ident,
            f"{c.runway_heading_deg:.0f}°",
            f"[{cw_color}]{c.crosswind_kts:.1f} kts[/{cw_color}]",
            c.crosswind_side,
            hw_desc,
            status,
        )
    console.print(table)


@app.command("fir-locate")
def fir_locate_cmd(
    lat: float = typer.Option(..., "--lat", help="Latitude in decimal degrees"),
    lon: float = typer.Option(..., "--lon", help="Longitude in decimal degrees"),
    flight_level: float = typer.Option(350.0, "--fl", help="Flight Level (default: 350)"),
) -> None:
    """Identify which FIR airspace sector contains a given 3D coordinate."""
    engine = FIREngine()
    firs = engine.locate_fir(lat, lon, flight_level)
    if firs:
        console.print(f"[green]Coordinate ({lat:.4f}, {lon:.4f}) at FL{flight_level:.0f} is inside:[/green] [bold cyan]{', '.join(firs)}[/bold cyan]")
    else:
        console.print(f"[yellow]No enclosing FIR boundary found for ({lat:.4f}, {lon:.4f}) at FL{flight_level:.0f}.[/yellow]")


@app.command("predict-delay")
def predict_delay_cmd(
    adep: str = typer.Option(..., "--adep", help="Departure airport ICAO"),
    ades: str = typer.Option(..., "--ades", help="Destination airport ICAO"),
    operator: str = typer.Option("BAW", "--operator", help="Airline operator ICAO"),
    market_segment: str = typer.Option("Traditional Scheduled", "--segment"),
    requested_fl: float = typer.Option(350.0, "--fl", help="Cruising flight level"),
    distance: float = typer.Option(3400.0, "--dist", help="Distance in nautical miles"),
    hour: int = typer.Option(10, "--hour", help="Departure hour (0-23)"),
    dayofweek: int = typer.Option(2, "--dow", help="Day of week (0-6)"),
) -> None:
    """Predict flight delay using trained gradient-boosted ML model."""
    model_path = MODEL_DIR / "arr_delay_min_model.pkl"
    if model_path.exists():
        predictor = DelayPredictor.load(model_path)
    else:
        with console.status("[cyan]Training baseline delay model..."):
            ds = FlightDataset()
            predictor = DelayPredictor()
            df = predictor.prepare_data(ds, sample_size=40000)
            predictor.train(df)
            predictor.save(model_path)

    pred = predictor.predict({
        "adep": adep.upper(),
        "ades": ades.upper(),
        "ac_operator": operator.upper(),
        "market_segment": market_segment,
        "requested_fl": requested_fl,
        "actual_distance_nm": distance,
        "dep_hour": hour,
        "dep_dayofweek": dayofweek,
        "filed_air_time_min": (distance / 450.0) * 60.0,
    })

    risk = "HIGH" if pred > 15 else "MODERATE" if pred > 5 else "LOW"
    risk_color = "red" if risk == "HIGH" else "yellow" if risk == "MODERATE" else "green"

    console.print(f"\n[bold]Flight Delay Prediction:[/bold] [cyan]{adep.upper()}[/cyan] -> [cyan]{ades.upper()}[/cyan] ({operator.upper()})")
    console.print(f"Predicted Arrival Delay: [bold {risk_color}]{pred} minutes[/bold {risk_color}]")
    console.print(f"Operational Risk Level: [{risk_color}]{risk}[/{risk_color}]")


@app.command("fuel-impact")
def fuel_impact_cmd(
    ectrl_id: int = typer.Argument(..., help="ECTRL ID of the flight to evaluate"),
    period: str = typer.Option("201611", "--period", "-P", help="Eurocontrol dataset period (e.g. 201611, 202510)"),
    price_per_kg: float | None = typer.Option(None, "--price", "-p", help="Fuel price per kg in EUR (defaults to official EIA weekly spot price in EUR for flight date)"),
) -> None:
    """Calculate excess fuel burn, CO2 emissions, and Euro (€) financial penalty from holdings and detour."""
    ds = FlightDataset(period=period)
    with console.status("[cyan]Computing flight trajectory fuel burn and carbon metrics..."):
        impact = compute_flight_fuel_and_emissions(ds, ectrl_id, fuel_price_per_kg=price_per_kg)

    console.print(f"\n[bold green]Environmental & Fuel Impact:[/bold green] ECTRL {ectrl_id} ({impact.ac_type}) - Period {period}")
    table = Table(title="Fuel & Carbon Inefficiency Analysis", header_style="bold yellow")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Excess Distance Flown", f"{impact.excess_distance_nm:.1f} nm")
    table.add_row("Terminal Holding Duration", f"{impact.holding_duration_min:.1f} min")
    table.add_row("Holding Stack Fuel Burned", f"{impact.excess_holding_fuel_kg:.1f} kg")
    table.add_row("Route Detour Fuel Burned", f"{impact.excess_routing_fuel_kg:.1f} kg")
    table.add_row("Total Excess Fuel Burned", f"[bold red]{impact.total_excess_fuel_kg:,.1f} kg[/bold red]")
    table.add_row("Total Excess CO₂ Emitted", f"[bold red]{impact.excess_co2_kg:,.1f} kg ({impact.excess_co2_kg/1000.0:.2f} metric tons)[/bold red]")
    table.add_row("Weekly Jet Fuel Price", f"€{impact.fuel_price_eur_per_gallon:.3f}/gal (€{impact.fuel_price_eur_per_kg:.4f}/kg)")
    table.add_row("Price Week & Benchmark", f"{impact.fuel_price_week} [{impact.fuel_price_source}]")
    table.add_row("Estimated Excess Fuel Cost", f"[bold yellow]€{impact.estimated_excess_fuel_cost_eur:,.2f} EUR[/bold yellow]")

    console.print(table)


@app.command("fuel-prices")
def fuel_prices_cmd(
    period: str | None = typer.Option(None, "--period", "-P", help="Filter by Eurocontrol period (e.g. 201611, 202510)"),
) -> None:
    """Display official EIA / IATA historical weekly jet fuel spot prices (in Euros EUR) across dataset periods."""
    df = get_all_weekly_fuel_prices_df()
    if period:
        df = df[df["period_id"] == period.strip()]

    title = f"⛽ Weekly Aviation Jet Fuel Spot Prices in Euros (Source: EIA / IATA Benchmark)"
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Period", style="cyan", justify="center")
    table.add_column("Week Range", style="bold white")
    table.add_column("Price (€/gal)", justify="right", style="bold green")
    table.add_column("Price (€/kg)", justify="right", style="yellow")
    table.add_column("Price (€/metric ton)", justify="right", style="bold yellow")
    table.add_column("Benchmark Source", style="dim")

    for _, row in df.iterrows():
        table.add_row(
            str(row["period_id"]),
            f"{row['year']}-W{int(row['week_number']):02d} ({row['week_start']} ➔ {row['week_end']})",
            f"€{row['price_eur_per_gallon']:.3f}",
            f"€{row['price_eur_per_kg']:.4f}",
            f"€{row['price_eur_per_metric_ton']:,.2f}",
            str(row["source"]),
        )
    console.print(table)


@app.command("simulate-buffer")
def simulate_buffer_cmd(
    operator: str = typer.Option("RYR", "--operator", "-op", help="Airline ICAO code (e.g. RYR, DLH, BAW)"),
    extra_min: float = typer.Option(10.0, "--extra-min", "-m", help="Extra scheduled turnaround buffer in minutes"),
    start: str = typer.Option("2016-11-01", "--start"),
    end: str = typer.Option("2016-11-04", "--end"),
) -> None:
    """Simulate how expanding turnaround buffers reduces reactionary delays across a fleet."""
    ds = FlightDataset()
    with console.status(f"[cyan]Simulating +{extra_min:.0f} min turnaround buffer for {operator}..."):
        res = simulate_turnaround_mitigation(
            ds, operator=operator, extra_buffer_min=extra_min, start_date=start, end_date=end
        )

    if "error" in res:
        console.print(f"[red]{res['error']}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"What-If Simulation: +{extra_min:.0f} min Turnaround Buffer for {operator.upper()}", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Baseline Reactionary Incidents", f"{res['baseline_reactionary_incidents']:,}")
    table.add_row("Incidents Prevented", f"[bold green]{res['incidents_prevented']:,}[/bold green]")
    table.add_row("Baseline Propagated Delay", f"{res['baseline_propagated_delay_min']:,.1f} min")
    table.add_row("Delay Minutes Saved", f"[bold green]{res['delay_minutes_saved']:,.1f} min[/bold green]")
    table.add_row("Reactionary Delay Reduction", f"[bold yellow]{res['delay_reduction_percentage']}%[/bold yellow]")

    console.print(table)


@app.command("gdp")
def gdp_cmd(
    airport: str = typer.Option("EGLL", "--airport", "-a", help="Destination airport subject to capacity degradation"),
    date: str = typer.Option("2016-11-01", "--date", help="Program date (YYYY-MM-DD)"),
    start_hour: int = typer.Option(6, "--start", help="Program start hour (0-23)"),
    end_hour: int = typer.Option(12, "--end", help="Program end hour (0-23)"),
    rate: int = typer.Option(25, "--rate", "-r", help="Degraded arrival capacity rate (movements / hr)"),
    price: float | None = typer.Option(None, "--price", "-p", help="Fuel price per kg in EUR"),
) -> None:
    """Run Ration-by-Schedule (RBS) Ground Delay Program (GDP) optimization to prevent airborne holdings."""
    ds = FlightDataset()
    with console.status(f"[cyan]Optimizing Ground Delay Program (GDP) for {airport} ({rate} arr/hr)..."):
        res = optimize_ground_delay_program(
            ds,
            target_airport=airport,
            program_date=date,
            start_hour=start_hour,
            end_hour=end_hour,
            degraded_capacity_rate_per_hr=rate,
            fuel_price_per_kg=price,
        )

    console.print(f"\n[bold green]Ground Delay Program (GDP) Analysis:[/bold green] {airport} (Rate: {rate} movements/hr)")
    table = Table(title=f"GDP Optimization Impact ({res.program_start} to {res.program_end})", header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Flights Regulated", f"{res.total_flights_regulated:,}")
    table.add_row("Ground Delay Assigned (at Origin)", f"{res.total_ground_delay_assigned_min:,.1f} min")
    table.add_row("Airborne Holding Delay Prevented", f"[bold green]{res.total_airborne_delay_prevented_min:,.1f} min[/bold green]")
    table.add_row("Total Jet-A Fuel Saved", f"[bold green]{res.total_fuel_saved_kg:,.1f} kg ({res.total_fuel_saved_kg/1000.0:,.1f} metric tons)[/bold green]")
    table.add_row("Total CO₂ Emissions Prevented", f"[bold green]{res.total_co2_saved_kg:,.1f} kg ({res.total_co2_saved_kg/1000.0:,.1f} metric tons)[/bold green]")
    table.add_row("Direct Fuel Cost Saved", f"[bold yellow]€{res.total_fuel_cost_saved_eur:,.2f} EUR[/bold yellow]")

    console.print(table)

    if res.slot_assignments:
        s_table = Table(title="Sample Controlled Time of Arrival (CTA) Slot Assignments", header_style="bold yellow")
        s_table.add_column("Flight ID")
        s_table.add_column("Operator")
        s_table.add_column("Origin")
        s_table.add_column("Aircraft")
        s_table.add_column("Original ETA")
        s_table.add_column("Assigned Slot")
        s_table.add_column("Ground Delay")
        s_table.add_column("Fuel Saved")

        for s in res.slot_assignments[:8]:
            s_table.add_row(
                str(s.ectrl_id),
                s.ac_operator,
                s.adep,
                s.ac_type,
                s.original_arrival_time[-8:],
                s.assigned_slot_time[-8:],
                f"{s.ground_delay_assigned_min:.1f} min",
                f"{s.fuel_saved_kg:.1f} kg",
            )
        console.print(s_table)


@app.command("compare-twins")
def compare_twins_cmd(
    adep: str = typer.Option("RKSI", "--adep", help="Departure airport ICAO"),
    ades: str = typer.Option("EGLL", "--ades", help="Destination airport ICAO"),
    ac_type: str = typer.Option("B77W", "--ac", help="Aircraft type ICAO code"),
    map_out: str = typer.Option(None, "--map", "-m", help="Path to save dual-trajectory comparison map HTML"),
) -> None:
    """Compare and contrast the most and least efficient flights of identical complexity."""
    ds = FlightDataset()
    with console.status(f"[cyan]Benchmarking twin flight efficiency for {adep} -> {ades} ({ac_type})..."):
        comp = compare_twin_flights(ds, adep=adep, ades=ades, ac_type=ac_type)

    if not comp:
        console.print(f"[red]Not enough matching flights for {adep} -> {ades} ({ac_type})[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]Twin-Flight Efficiency Benchmark:[/bold green] [cyan]{adep}[/cyan] ➔ [cyan]{ades}[/cyan] ({ac_type})")
    
    table = Table(title="Most Efficient vs Least Efficient Operational Comparison", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("High-Efficiency Flight", style="green")
    table.add_column("Low-Efficiency Flight", style="red")
    table.add_column("Disparity Delta", style="bold yellow")

    table.add_row("ECTRL Flight ID", str(comp.fast_id), str(comp.slow_id), "-")
    table.add_row("Operator", comp.fast_operator, comp.slow_operator, "-")
    table.add_row("Actual Distance", f"{comp.fast_actual_distance_nm:.1f} nm", f"{comp.slow_actual_distance_nm:.1f} nm", f"+{comp.distance_penalty_nm:.1f} nm")
    table.add_row("Actual Air Time", f"{comp.fast_air_time_min:.1f} min", f"{comp.slow_air_time_min:.1f} min", f"+{comp.air_time_penalty_min:.1f} min")
    table.add_row("Extension Ratio (KEA)", f"{comp.fast_extension_ratio:.3f}x", f"{comp.slow_extension_ratio:.3f}x", f"+{comp.slow_extension_ratio - comp.fast_extension_ratio:.3f}x")
    table.add_row("Fuel Burned", f"{comp.fast_fuel_kg:,.1f} kg", f"{comp.slow_fuel_kg:,.1f} kg", f"+{comp.excess_fuel_burned_kg:,.1f} kg")
    table.add_row("CO₂ Emitted", f"{comp.fast_fuel_kg*3.16/1000:,.1f} tons", f"{comp.slow_fuel_kg*3.16/1000:,.1f} tons", f"+{comp.excess_co2_kg/1000:,.1f} tons")
    table.add_row("Excess Fuel Cost", "€0.00", f"€{comp.excess_fuel_cost_eur:,.2f}", f"€{comp.excess_fuel_cost_eur:,.2f} EUR")

    console.print(table)

    if comp.divergence_lat and comp.divergence_lon:
        console.print(f"[yellow]Major route divergence detected at coordinates:[/yellow] ({comp.divergence_lat:.3f}°, {comp.divergence_lon:.3f}°)")

    if map_out:
        render_twin_comparison_map(ds, comp, output_html_path=map_out)
        console.print(f"[bold green]Dual-trajectory map saved:[/bold green] [cyan]{map_out}[/cyan]")


@app.command("stress-test")
def stress_test_cmd(
    operator: str = typer.Option("RYR", "--operator", "-op", help="Airline operator ICAO (e.g. RYR, DLH, BAW)"),
    start: str = typer.Option("2016-11-01", "--start"),
    end: str = typer.Option("2016-11-04", "--end"),
) -> None:
    """Stress-test airline scheduling models under operational delay shock waves."""
    ds = FlightDataset()
    with console.status(f"[cyan]Running fleet operational stress simulation for {operator}..."):
        res = stress_test_airline(ds, operator=operator, start_date=start, end_date=end)

    if not res.fragility_curve:
        console.print(f"[red]No flight rotation data found for {operator}[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]Airline Operational Stress Test:[/bold green] [cyan]{operator.upper()}[/cyan]")
    console.print(f"Rotations Analyzed: {res.total_rotations:,} | Mean Turnaround Buffer: {res.mean_scheduled_turnaround_min:.1f} min")
    console.print(f"Resilience Rating: [bold yellow]{res.resilience_grade}[/bold yellow] | 50% Infection Threshold: [bold red]{res.resilience_threshold_shock_min:.0f} min shock[/bold red]\n")

    table = Table(title="Injected Shock vs Downstream Contagion Curve", header_style="bold magenta")
    table.add_column("Injected Shock", style="cyan")
    table.add_column("Fleet Infection Rate", justify="right")
    table.add_column("Total Propagated Delay", justify="right", style="yellow")
    table.add_column("Cascading Multiplier", justify="right", style="bold red")

    for pt in res.fragility_curve:
        inf_color = "red" if pt["infection_rate_pct"] >= 50 else "yellow" if pt["infection_rate_pct"] >= 25 else "green"
        table.add_row(
            f"+{pt['injected_shock_min']:.0f} min",
            f"[{inf_color}]{pt['infection_rate_pct']:.1f}%[/{inf_color}]",
            f"{pt['propagated_delay_total_min']:,.1f} min",
            f"{pt['cascade_multiplier']:.3f}x",
        )
    console.print(table)


@app.command("list-periods")
def list_periods_cmd() -> None:
    """List all available ingested Eurocontrol data periods."""
    periods = list_available_periods()
    table = Table(title="Available Aviation Data Periods", header_style="bold magenta")
    table.add_column("Period ID", style="cyan")
    table.add_column("Status", style="green")

    for p in periods:
        table.add_row(p, "Ready (Parquet Partitioned)")
    console.print(table)


@app.command("ingest-archive")
def ingest_archive_cmd(
    archive_path: str = typer.Argument(..., help="Path to monthly Eurocontrol zip file (e.g. data/202510.zip)"),
) -> None:
    """Extract, classify, and ingest a monthly Eurocontrol archive into partitioned Parquet."""
    path = Path(archive_path)
    if not path.exists():
        console.print(f"[bold red]Archive not found at: {path}[/bold red]")
        raise typer.Exit(code=1)

    ingestor = EurocontrolArchiveIngestor()
    console.print(f"[bold cyan]Ingesting Eurocontrol Archive:[/bold cyan] {path.name}")
    results = ingestor.ingest_zip(path)
    console.print(f"\n[bold green]Successfully ingested {len(results)} tables into partitioned Parquet![/bold green]")


@app.command("compare-periods")
def compare_periods_cmd(
    p1: str = typer.Option("201611", "--p1", help="Baseline period ID (e.g. 201611)"),
    p2: str = typer.Option("202510", "--p2", help="Comparison period ID (e.g. 202510)"),
) -> None:
    """Deep macro comparison between two Eurocontrol monthly archives (traffic, delay, airlines, fleet)."""
    with console.status(f"[cyan]Comparing aviation periods {p1} vs {p2}..."):
        comp = compare_aviation_periods(period1=p1, period2=p2)

    console.print(f"\n[bold green]Macro Evolution Comparison:[/bold green] [cyan]{comp.period1}[/cyan] ➔ [cyan]{comp.period2}[/cyan]")
    
    # Overview Table
    overview = Table(title="Macro Traffic & Network Delay Overview", header_style="bold magenta")
    overview.add_column("Metric", style="cyan")
    overview.add_column(f"{comp.period1} (Baseline)", style="white")
    overview.add_column(f"{comp.period2} (Comparison)", style="white")
    overview.add_column("Delta / Evolution", style="bold yellow")

    growth_str = f"+{comp.traffic_growth_pct:.1f}%" if comp.traffic_growth_pct >= 0 else f"{comp.traffic_growth_pct:.1f}%"
    delay_str = f"+{comp.delay_change_min:.2f} min" if comp.delay_change_min >= 0 else f"{comp.delay_change_min:.2f} min"

    overview.add_row("Total Flights", f"{comp.total_flights_p1:,}", f"{comp.total_flights_p2:,}", growth_str)
    overview.add_row("Mean Arrival Delay", f"{comp.avg_arr_delay_p1:.2f} min", f"{comp.avg_arr_delay_p2:.2f} min", delay_str)
    console.print(overview)

    # Growing Airlines Table
    if comp.top_growing_airlines:
        air_table = Table(title="Top Busiest & Growing Airlines (Monthly Flights)", header_style="bold cyan")
        air_table.add_column("Airline", style="bold")
        air_table.add_column(f"{comp.period1} Flights", justify="right")
        air_table.add_column(f"{comp.period2} Flights", justify="right")
        air_table.add_column("Absolute Growth", justify="right", style="green")
        air_table.add_column("Growth Rate", justify="right", style="bold green")

        for a in comp.top_growing_airlines[:8]:
            rate_str = f"+{a['growth_pct']:.1f}%" if a.get("growth_pct") is not None and not pd.isna(a['growth_pct']) else "NEW"
            air_table.add_row(
                str(a["airline"]),
                f"{int(a['flights_p1']):,}",
                f"{int(a['flights_p2']):,}",
                f"{int(a['growth_flights']):+,}",
                rate_str,
            )
        console.print(air_table)

    # Fleet Shifts Table
    if comp.fleet_modernization_shifts:
        fleet_table = Table(title="Top Aircraft Types & Fleet Shifts", header_style="bold green")
        fleet_table.add_column("Aircraft Type", style="bold")
        fleet_table.add_column(f"{comp.period1} Flights", justify="right")
        fleet_table.add_column(f"{comp.period2} Flights", justify="right", style="cyan")
        fleet_table.add_column("Delta Flights", justify="right", style="yellow")

        for f in comp.fleet_modernization_shifts[:8]:
            fleet_table.add_row(
                str(f["ac_type"]),
                f"{int(f['flights_p1']):,}",
                f"{int(f['flights_p2']):,}",
                f"{int(f['delta_flights']):+,}",
            )
        console.print(fleet_table)


@app.command("cdo-descent")
def cdo_descent_cmd(
    ectrl_id: int = typer.Argument(..., help="ECTRL Flight ID (e.g. 202007026)"),
    period: str = typer.Option("201611", "--period", "-p", help="Data period ID"),
) -> None:
    """Analyze Continuous Descent Operations (CDO), level-off staircases, and descent fuel waste."""
    ds = FlightDataset(period=period)
    with console.status(f"[cyan]Analyzing CDO descent profile for Flight {ectrl_id}..."):
        report = analyze_cdo_descent(ds, ectrl_id=ectrl_id)

    if not report:
        console.print(f"[bold red]Descent profile not found for flight {ectrl_id}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]Continuous Descent Operations (CDO) Report:[/bold green] ECTRL [cyan]{report.ectrl_id}[/cyan]")
    console.print(f"Route: [cyan]{report.adep}[/cyan] ➔ [cyan]{report.ades}[/cyan] | Aircraft: [white]{report.ac_type}[/white] | Cruise FL: [white]{report.cruise_flight_level:.0f}[/white]")
    
    score_color = "green" if report.cdo_compliance_score >= 80 else "yellow" if report.cdo_compliance_score >= 50 else "red"
    console.print(f"CDO Compliance Score: [{score_color} bold]{report.cdo_compliance_score:.1f}%[/{score_color} bold] | Total Descent Time: {report.total_descent_duration_min:.1f} min")
    console.print(f"Level-Off Inefficiencies: [bold red]{report.inefficiency_events_count} events[/bold red] ({report.total_level_or_climb_min:.1f} min un-optimized level flight)")
    console.print(f"Excess Descent Fuel Burn: [bold yellow]{report.excess_fuel_burned_kg:.1f} kg Jet-A[/bold yellow] | Excess CO₂: [bold]{report.excess_co2_kg:.1f} kg[/bold]\n")

    if report.inefficiency_events:
        table = Table(title="Detected Descent Level-Off & Altitude Reversal Events", header_style="bold magenta")
        table.add_column("Event Type", style="bold red")
        table.add_column("Start Time (UTC)")
        table.add_column("End Time (UTC)")
        table.add_column("Duration", justify="right", style="yellow")
        table.add_column("Flight Level", justify="right", style="cyan")
        table.add_column("Approx Coordinates")

        for ev in report.inefficiency_events:
            table.add_row(
                ev.event_type,
                str(ev.start_time),
                str(ev.end_time),
                f"{ev.duration_min:.1f} min",
                f"FL{ev.flight_level:.0f}",
                f"{ev.latitude:.2f}°, {ev.longitude:.2f}°",
            )
        console.print(table)


@app.command("weather-impact")
def weather_impact_cmd(
    start: str = typer.Option("2016-11-01", "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2016-11-10", "--end", "-e", help="End date (YYYY-MM-DD)"),
    period: str = typer.Option("201611", "--period", "-p", help="Data period ID"),
) -> None:
    """Analyze European macro weather history and direct statistical correlation with air traffic delays."""
    ds = FlightDataset(period=period)
    client = MetarClient()

    with console.status(f"[cyan]Correlating European weather history and flight delay trends ({start} to {end})..."):
        report = analyze_network_weather_impact(ds, client, start_date=start, end_date=end)

    console.print(f"\n[bold green]European Weather Disruption & Air Traffic Correlation Report[/bold green]")
    console.print(f"Time Window: [cyan]{report.start_date} to {report.end_date}[/cyan] | Analyzed Flights: [white]{report.total_flights_analyzed:,}[/white]")
    console.print(f"Monitored Gateways: {', '.join(report.analyzed_hubs)}\n")

    summary_table = Table(title="Statistical Weather Impact Metrics", header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row(
        "Pearson Corr (Arrival Delay vs IFR Hours)",
        f"{report.pearson_corr_delay_vs_ifr:+.3f}",
    )
    summary_table.add_row(
        "Pearson Corr (Severe Delays >30m vs Weather Severity)",
        f"{report.pearson_corr_severe_delays_vs_weather:+.3f}",
    )
    summary_table.add_row(
        "Weather Sensitivity (+Delay min per Severity Index Point)",
        f"+{report.delay_sensitivity_minutes_per_severity_point:.2f} min / pt",
    )
    summary_table.add_row(
        "Clear Sky Baseline Arrival Delay",
        f"{report.clear_sky_baseline_delay_min:.2f} min",
    )
    summary_table.add_row(
        "Peak Adverse Weather Day",
        f"{report.worst_weather_day['date']} (Severity {report.worst_weather_day['severity_index']}/10, {report.worst_weather_day['ifr_pct']}% IFR)",
    )
    console.print(summary_table)

    # Daily breakdown table
    trend_table = Table(title="Daily Traffic Delay & Weather Severity Trends", header_style="bold cyan")
    trend_table.add_column("Date", style="bold")
    trend_table.add_column("Daily Flights", justify="right")
    trend_table.add_column("Avg Arr Delay", justify="right", style="yellow")
    trend_table.add_column("Delayed >30m", justify="right", style="red")
    trend_table.add_column("IFR Hours %", justify="right", style="magenta")
    trend_table.add_column("Severity (0-10)", justify="right", style="bold green")

    for rec in report.daily_time_series:
        trend_table.add_row(
            rec.date_str,
            f"{rec.total_flights:,}",
            f"{rec.avg_arr_delay_min:.2f} min",
            f"{rec.pct_flights_delayed_30m:.1f}%",
            f"{rec.ifr_hours_pct:.1f}%",
            f"{rec.composite_weather_severity_index:.1f}",
        )
    console.print(trend_table)


@app.command("airac-diff")
def airac_diff_cmd(
    c1: str = typer.Option("2509", "--c1", help="First AIRAC cycle ID (e.g. 2509)"),
    c2: str = typer.Option("2510", "--c2", help="Second AIRAC cycle ID (e.g. 2510)"),
) -> None:
    """Compare contiguous AIRAC cycle aeronautical configurations (airway routes & FIR boundaries)."""
    with console.status(f"[cyan]Diffing aeronautical configurations: AIRAC {c1} ➔ AIRAC {c2}..."):
        report = compare_airac_cycles(cycle1=c1, cycle2=c2)

    console.print(f"\n[bold green]AIRAC Cycle Configuration Diff:[/bold green] [cyan]{c1}[/cyan] ➔ [cyan]{c2}[/cyan]\n")

    r_table = Table(title="Airway Route Segment Modifications", header_style="bold magenta")
    r_table.add_column("Metric", style="cyan")
    r_table.add_column(f"AIRAC {c1}", justify="right")
    r_table.add_column(f"AIRAC {c2}", justify="right")
    r_table.add_column("Net Evolution", justify="right", style="bold yellow")

    rc = report.route_changes
    r_table.add_row("Total Airway Segments", f"{rc.total_routes_c1:,}", f"{rc.total_routes_c2:,}", f"{rc.total_routes_c2 - rc.total_routes_c1:+,}")
    r_table.add_row("Newly Commissioned Airways", "-", f"{rc.added_routes_count:,}", f"+{rc.added_routes_count:,}")
    r_table.add_row("Decommissioned Airways", f"{rc.deleted_routes_count:,}", "-", f"-{rc.deleted_routes_count:,}")
    r_table.add_row("Modified Waypoint Sequences", "-", f"{rc.modified_routes_count:,}", f"~{rc.modified_routes_count:,}")
    console.print(r_table)

    if rc.sample_added_routes:
        console.print(f"[green]Sample Newly Commissioned Airways:[/green] {', '.join(rc.sample_added_routes)}")
    if rc.sample_deleted_routes:
        console.print(f"[red]Sample Decommissioned Airways:[/red] {', '.join(rc.sample_deleted_routes)}")

    console.print()
    f_table = Table(title="FIR Airspace Boundary & Stratum Modifications", header_style="bold cyan")
    f_table.add_column("Metric", style="cyan")
    f_table.add_column(f"AIRAC {c1}", justify="right")
    f_table.add_column(f"AIRAC {c2}", justify="right")
    f_table.add_column("Net Evolution", justify="right", style="bold yellow")

    fc = report.fir_changes
    f_table.add_row("Total FIR Airspaces", f"{fc.total_firs_c1:,}", f"{fc.total_firs_c2:,}", f"{fc.total_firs_c2 - fc.total_firs_c1:+,}")
    f_table.add_row("Boundary Re-sectorizations", "-", f"{fc.modified_boundaries_count:,}", f"~{fc.modified_boundaries_count:,}")
    console.print(f_table)

    if fc.sample_modified_firs:
        console.print(f"[yellow]Sectors with Re-profiled Boundary Coordinates:[/yellow] {', '.join(fc.sample_modified_firs)}\n")


@app.command("search-flights")
def search_flights_cmd(
    adep: str = typer.Option(None, "--adep", help="Departure airport ICAO (e.g. EGLL)"),
    ades: str = typer.Option(None, "--ades", help="Destination airport ICAO (e.g. KORD)"),
    operator: str = typer.Option(None, "--operator", "-op", help="Airline operator (e.g. BAW, RYR)"),
    ac_type: str = typer.Option(None, "--ac", help="Aircraft type (e.g. B77W, A320)"),
    tail: str = typer.Option(None, "--tail", help="Tail registration (e.g. G-STBA)"),
    time_of_day: str = typer.Option(None, "--tod", help="Time of day: Morning, Afternoon, Evening, Night"),
    date_from: str = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    min_dist: float = typer.Option(None, "--min-dist", help="Minimum distance flown in nm"),
    max_dist: float = typer.Option(None, "--max-dist", help="Maximum distance flown in nm"),
    min_fl: float = typer.Option(None, "--min-fl", help="Minimum cruising flight level"),
    max_fl: float = typer.Option(None, "--max-fl", help="Maximum cruising flight level"),
    period: str = typer.Option("201611", "--period", "-p", help="Data period ID"),
    limit: int = typer.Option(15, "--limit", "-l", help="Max flights to return"),
) -> None:
    """Search and filter flights with human-readable labels across any historical period."""
    ds = FlightDataset(period=period)
    with console.status("[cyan]Searching flights matching specified criteria..."):
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

    if df.empty:
        console.print("[yellow]No flights found matching the specified filter criteria.[/yellow]")
        return

    table = Table(title=f"Flight Search Results (Period: {period}, {len(df)} matches)", header_style="bold magenta")
    table.add_column("Flight Label & Metadata", style="cyan")
    table.add_column("Distance", justify="right", style="green")
    table.add_column("FL", justify="right", style="white")
    table.add_column("Dep Delay", justify="right")
    table.add_column("Arr Delay", justify="right")

    for _, row in df.iterrows():
        arr_d = float(row.get("arr_delay_min") or 0.0)
        dep_d = float(row.get("dep_delay_min") or 0.0)
        arr_color = "red" if arr_d > 15 else "yellow" if arr_d > 5 else "green"
        dep_color = "red" if dep_d > 15 else "yellow" if dep_d > 5 else "green"

        table.add_row(
            row["label"],
            f"{row.get('actual_distance_nm', 0):.0f} nm",
            f"FL{row.get('requested_fl', 0):.0f}",
            f"[{dep_color}]{dep_d:+.0f}m[/{dep_color}]",
            f"[{arr_color}]{arr_d:+.0f}m[/{arr_color}]",
        )
    console.print(table)


@app.command("recurring-series")
def recurring_series_cmd(
    adep: str = typer.Argument(..., help="Departure airport ICAO (e.g. EGLL)"),
    ades: str = typer.Argument(..., help="Destination airport ICAO (e.g. KORD)"),
    operator: str = typer.Option(None, "--operator", "-op", help="Filter airline operator"),
    ac_type: str = typer.Option(None, "--ac", help="Filter aircraft type"),
    time_of_day: str = typer.Option(None, "--tod", help="Filter time-of-day bank (Morning/Afternoon/Evening/Night)"),
    period: str = typer.Option("201611", "--period", "-p", help="Data period ID"),
    limit: int = typer.Option(200, "--limit", "-l", help="Max flights to inspect"),
) -> None:
    """Analyze recurring flights on a corridor day-by-day with diurnal time-of-day assessment."""
    ds = FlightDataset(period=period)
    with console.status(f"[cyan]Analyzing longitudinal recurring series for {adep} -> {ades}..."):
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
        console.print(f"[yellow]Not enough recurring flights found for corridor {adep} -> {ades}[/yellow]")
        return

    console.print(f"\n[bold green]Recurring Flight Series & Diurnal Analysis:[/bold green] [cyan]{rep.route}[/cyan]")
    console.print(f"Total Occurrences: [white]{rep.total_flights}[/white] | Dates: [cyan]{rep.date_span}[/cyan]")
    console.print(f"Carrier: [yellow]{rep.filter_operator}[/yellow] | Fleet: [yellow]{rep.filter_ac_type}[/yellow]\n")

    # Overview Table
    ov_table = Table(title="Corridor Operational Variability & Dispersion", header_style="bold magenta")
    ov_table.add_column("Operational Metric", style="cyan")
    ov_table.add_column("Baseline Mean", justify="right", style="green")
    ov_table.add_column("Variability / Dispersion Span", justify="right", style="bold yellow")

    ov_table.add_row("Arrival Delay", f"{rep.mean_arr_delay_min:+.2f} min", f"±{rep.std_arr_delay_min:.2f} min (std)")
    ov_table.add_row("Departure Delay", f"{rep.mean_dep_delay_min:+.2f} min", "-")
    ov_table.add_row("Actual Air Time", f"{rep.mean_air_time_min:.1f} min", f"Δ{rep.air_time_span_min:.1f} min ({rep.min_air_time_min:.0f}m to {rep.max_air_time_min:.0f}m)")
    ov_table.add_row("Flown Distance", f"{rep.mean_distance_nm:.1f} nm", f"Δ{rep.distance_span_nm:.1f} nm detour span")
    console.print(ov_table)

    console.print(f"\n[bold green]Best Operational Day:[/bold green] {rep.best_performing_day['date']} ({rep.best_performing_day['flights']} flights, avg delay {rep.best_performing_day['avg_arr_delay_min']:+.1f} min)")
    console.print(f"[bold red]Worst Operational Day:[/bold red] {rep.worst_performing_day['date']} ({rep.worst_performing_day['flights']} flights, avg delay {rep.worst_performing_day['avg_arr_delay_min']:+.1f} min)\n")

    # Diurnal Banks Table
    diurnal_table = Table(title="Diurnal Time-of-Day Departure Bank Congestion Profile", header_style="bold cyan")
    diurnal_table.add_column("Time-of-Day Departure Bank", style="bold")
    diurnal_table.add_column("Flights", justify="right")
    diurnal_table.add_column("Avg Dep Delay", justify="right", style="yellow")
    diurnal_table.add_column("Avg Arr Delay", justify="right", style="bold red")
    diurnal_table.add_column("Avg Air Time", justify="right", style="green")

    for b in rep.diurnal_banks:
        arr_col = "red" if b.avg_arr_delay_min > 10 else "yellow" if b.avg_arr_delay_min > 0 else "green"
        diurnal_table.add_row(
            b.bank_name,
            f"{b.flight_count}",
            f"{b.avg_dep_delay_min:+.1f} min",
            f"[{arr_col}]{b.avg_arr_delay_min:+.1f} min[/{arr_col}]",
            f"{b.avg_air_time_min:.1f} min",
        )
    console.print(diurnal_table)


@app.command("fir-saturation")
def fir_saturation_cmd(
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
    fir_id: str | None = typer.Option(None, "--fir", help="Specific FIR ID to filter (e.g. LTBBFIR)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Filter by specific date (YYYY-MM-DD)"),
    threshold: int = typer.Option(40, "--threshold", "-t", help="Hourly flight threshold for saturation"),
    top_n: int = typer.Option(10, "--limit", "-n", help="Top N saturated FIRs"),
) -> None:
    """Analyze hourly en-route FIR sector saturation and identify European airspace bottlenecks."""
    with console.status(f"[cyan]Calculating en-route FIR sector saturation for period {period}..."):
        ds = FlightDataset(period=period)
        res = calculate_fir_hourly_saturation(
            ds, fir_id=fir_id, date_str=date, threshold_flights_per_hr=threshold, top_n=top_n
        )

    if not res.get("has_data"):
        console.print(f"[yellow]{res.get('message')}[/yellow]")
        return

    table = Table(
        title=f"🌐 European FIR Sector Saturation (Period: {period}, Threshold: ≥{threshold} flt/hr)",
        header_style="bold magenta",
    )
    table.add_column("FIR ID", style="cyan", justify="left")
    table.add_column("Total Traversals", justify="right")
    table.add_column("Peak Flts/Hr", justify="right", style="bold red")
    table.add_column("Avg Flts/Hr", justify="right")
    table.add_column("Avg Dwell (min)", justify="right")
    table.add_column(f"Hours ≥{threshold}", justify="right", style="yellow")
    table.add_column("Saturation %", justify="right", style="bold green")

    for f in res["top_saturated_firs"]:
        sat_pct = f"{f['saturation_pct']:.1f}%"
        color = "red" if f["saturation_pct"] > 80 else "yellow" if f["saturation_pct"] > 50 else "green"
        table.add_row(
            str(f["fir_id"]),
            f"{int(f['total_traversals']):,}",
            str(f["peak_hourly_flights"]),
            f"{f['avg_hourly_flights']:.1f}",
            f"{f['avg_dwell_min']:.1f}",
            str(f["hours_over_capacity"]),
            f"[{color}]{sat_pct}[/{color}]",
        )
    console.print(table)


@app.command("flight-fir-dwell")
def flight_fir_dwell_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID (e.g. 291262109)"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
) -> None:
    """Analyze sector-by-sector en-route dwell efficiency, detecting airborne holding, vectoring, or shortcuts."""
    with console.status(f"[cyan]Evaluating en-route sector traversals for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        res = calculate_flight_fir_dwell_inefficiency(ds, ectrl_id=ectrl_id)

    if not res.get("has_data"):
        console.print(f"[yellow]{res.get('message')}[/yellow]")
        return

    console.print(f"\n[bold green]En-Route Sector Traversal Analysis for Flight {ectrl_id} ({period})[/bold green]")
    console.print(f"Summary: [bold cyan]{res['verdict']}[/bold cyan]")
    console.print(
        f"Total Actual En-Route Dwell: {res['total_actual_dwell_min']} min | "
        f"Planned Filed Dwell: {res['total_filed_dwell_min']} min | "
        f"Net Sector Dwell Delta: [bold]{res['net_dwell_diff_min']:+.1f} min[/bold]\n"
    )

    table = Table(title="Traversed Airspace Sectors", header_style="bold magenta")
    table.add_column("Seq", justify="right", style="cyan")
    table.add_column("FIR ID", style="bold white")
    table.add_column("Actual Entry ➔ Exit", style="dim")
    table.add_column("Act Dwell", justify="right")
    table.add_column("Fld Dwell", justify="right")
    table.add_column("Delta (min)", justify="right")
    table.add_column("Status", justify="left")

    for s in res["sectors"]:
        delta_val = s["dwell_diff_min"]
        delta_str = f"{delta_val:+.1f}" if delta_val is not None else "N/A"
        color = "red" if s["status"] == "HOLDING/VECTORING" else "green" if s["status"] == "DIRECT_SHORTCUT" else "white"

        fld_d = f"{s['filed_dwell_min']:.1f}" if s["filed_dwell_min"] is not None else "N/A"
        entry_exit = f"{s['actual_entry'][11:16]} ➔ {s['actual_exit'][11:16]}"

        table.add_row(
            str(s["seq_num"]),
            s["fir_id"],
            entry_exit,
            f"{s['actual_dwell_min']:.1f}m",
            f"{fld_d}m",
            f"[{color}]{delta_str}[/{color}]",
            f"[{color}]{s['status']}[/{color}]",
        )
    console.print(table)


@app.command("external-data")
def external_data_cmd(
    rebuild: bool = typer.Option(False, "--rebuild", help="Force re-parsing and compiling of all external sources"),
) -> None:
    """Compile and audit all external navigation, OpenAP, ICAO EDB, and CRCO unit rate datasets."""
    from flightml.data.external_sources import compile_all_external_sources

    with console.status("[bold cyan]Compiling & verifying external aeronautical reference datasets..."):
        summary = compile_all_external_sources(force=rebuild)

    table = Table(
        title="🧭 Compiled External Reference Datasets (v2.0 Foundation)",
        header_style="bold magenta",
    )
    table.add_column("DuckDB View", style="bold cyan")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Source Path", style="white")
    table.add_column("Supports v2.0 Sprints", style="yellow")

    for view_name, meta in summary.items():
        table.add_row(
            view_name,
            f"{meta['rows']:,}",
            meta["source"],
            meta["supports_sprints"],
        )
    console.print(table)


@app.command("nearest-waypoints")
def nearest_waypoints_cmd(
    lat: float = typer.Option(..., "--lat", help="Latitude in decimal degrees"),
    lon: float = typer.Option(..., "--lon", help="Longitude in decimal degrees"),
    radius: float = typer.Option(15.0, "--radius", "-r", help="Search radius in nautical miles"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max waypoints/navaids to return"),
) -> None:
    """Find the closest global 5-letter RNAV waypoints (fix.dat) and radio navaids (nav.dat)."""
    ds = FlightDataset()
    df = ds.find_nearest_waypoints(lat=lat, lon=lon, radius_nm=radius, limit=limit)
    if df.empty:
        console.print(f"[yellow]No waypoints or navaids found within {radius} nm of ({lat}, {lon}).[/yellow]")
        return

    table = Table(
        title=f"Nearest Navigation Fixes & Navaids to ({lat:.4f}°, {lon:.4f}°)",
        header_style="bold magenta",
    )
    table.add_column("Ident", style="bold cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Name", style="white")
    table.add_column("Coordinates", justify="right")
    table.add_column("Freq (MHz)", justify="right", style="green")
    table.add_column("Distance", justify="right", style="bold green")

    for _, r in df.iterrows():
        freq_str = f"{r['frequency_mhz']:.2f}" if r["frequency_mhz"] is not None and str(r["frequency_mhz"]) != "nan" else "—"
        table.add_row(
            str(r["ident"]),
            str(r["kind"]),
            str(r["name"]),
            f"{r['latitude']:.4f}°, {r['longitude']:.4f}°",
            freq_str,
            f"{r['distance_nm']:.2f} nm",
        )
    console.print(table)


@app.command("aua-saturation")
def aua_saturation_cmd(
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
    aua_type: str | None = typer.Option(None, "--type", "-k", help="Filter AUA type: TMA, CTA, UTA"),
    aua_id: str | None = typer.Option(None, "--aua", help="Specific AUA ID filter (e.g. EGTTCTA, EYVITMA)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Filter by specific date (YYYY-MM-DD)"),
    threshold: int = typer.Option(30, "--threshold", "-t", help="Hourly flight saturation threshold"),
    top_n: int = typer.Option(15, "--limit", "-n", help="Top N saturated AUAs"),
) -> None:
    """Analyze hourly ATC Unit Airspace (TMA, CTA, UTA) saturation and sector congestion."""
    with console.status(f"[cyan]Calculating ATC Unit Airspace (AUA) saturation for period {period}..."):
        ds = FlightDataset(period=period)
        res = calculate_aua_hourly_saturation(
            ds,
            aua_type=aua_type,
            aua_id=aua_id,
            date_str=date,
            threshold_flights_per_hr=threshold,
            top_n=top_n,
        )

    if not res.get("has_data"):
        console.print(f"[yellow]{res.get('message')}[/yellow]")
        return

    type_label = f" [{aua_type.upper()}]" if aua_type else " [TMA/CTA/UTA]"
    table = Table(
        title=f"📡 European ATC Unit Airspace (AUA) Saturation{type_label} (Period: {period}, ≥{threshold} flt/hr)",
        header_style="bold magenta",
    )
    table.add_column("AUA ID", style="bold cyan", justify="left")
    table.add_column("Type", style="yellow", justify="center")
    table.add_column("Total Traversals", justify="right")
    table.add_column("Peak Flts/Hr", justify="right", style="bold red")
    table.add_column("Avg Flts/Hr", justify="right")
    table.add_column("Avg Dwell (min)", justify="right")
    table.add_column(f"Hours ≥{threshold}", justify="right", style="yellow")
    table.add_column("Saturation %", justify="right", style="bold green")

    for row in res["top_saturated_auas"]:
        sat_pct = f"{row['saturation_pct']:.1f}%"
        color = "red" if row["saturation_pct"] > 75 else "yellow" if row["saturation_pct"] > 40 else "green"
        table.add_row(
            str(row["aua_id"]),
            str(row["aua_type"]),
            f"{int(row['total_traversals']):,}",
            str(row["peak_hourly_flights"]),
            f"{row['avg_hourly_flights']:.1f}",
            f"{row['avg_dwell_min']:.1f}",
            str(row["hours_over_capacity"]),
            f"[{color}]{sat_pct}[/{color}]",
        )
    console.print(table)


@app.command("flight-aua-dwell")
def flight_aua_dwell_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
) -> None:
    """Assess sector-by-sector ATC Unit Airspace (TMA, CTA, UTA) dwell efficiency for a single flight."""
    with console.status(f"[cyan]Evaluating AUA (TMA/CTA/UTA) traversals for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        res = calculate_flight_aua_dwell_inefficiency(ds, ectrl_id=ectrl_id)

    if not res.get("has_data"):
        console.print(f"[yellow]{res.get('message')}[/yellow]")
        return

    console.print(
        f"\n[bold green]ATC Unit Airspace (AUA) Dwell Report:[/bold green] Flight {ectrl_id} "
        f"([cyan]{res.get('adep')}[/cyan] ➔ [cyan]{res.get('ades')}[/cyan], {res.get('ac_type')})"
    )
    console.print(
        f"Total Actual AUA Dwell: {res['total_actual_aua_min']:.1f} min | "
        f"Filed AUA Dwell: {res['total_filed_aua_min']:.1f} min | "
        f"Net Delta: [bold]{res['net_aua_dwell_delta_min']:+.1f} min[/bold]"
    )
    console.print(
        f"TMA (Terminal) Dwell Delta: [yellow]{res['tma_dwell_delta_min']:+.1f} min[/yellow] | "
        f"CTA/UTA (En-Route) Dwell Delta: [cyan]{res['enroute_cta_uta_dwell_delta_min']:+.1f} min[/cyan]\n"
    )

    table = Table(title="Traversed ATC Unit Airspaces (TMA / CTA / UTA)", header_style="bold magenta")
    table.add_column("Seq", justify="right", style="cyan")
    table.add_column("AUA ID", style="bold white")
    table.add_column("Type", style="yellow")
    table.add_column("Actual Entry ➔ Exit", style="dim")
    table.add_column("Act Dwell", justify="right")
    table.add_column("Fld Dwell", justify="right")
    table.add_column("Delta (min)", justify="right")
    table.add_column("Status", justify="left")

    for s in res["sectors"]:
        delta_val = s["dwell_diff_min"]
        delta_str = f"{delta_val:+.1f}" if delta_val is not None else "N/A"
        color = (
            "red"
            if s["status"] == "HOLDING_OR_VECTORING"
            else "green"
            if s["status"] == "DIRECT_SHORTCUT"
            else "white"
        )
        fld_d = f"{s['filed_dwell_min']:.1f}m" if s["filed_dwell_min"] is not None else "N/A"
        entry_exit = f"{s['actual_entry'][11:16]} ➔ {s['actual_exit'][11:16]}"
        table.add_row(
            str(s["seq_num"]),
            s["aua_id"],
            s["aua_type"],
            entry_exit,
            f"{s['actual_dwell_min']:.1f}m",
            fld_d,
            f"[{color}]{delta_str}[/{color}]",
            f"[{color}]{s['status']}[/{color}]",
        )
    console.print(table)


@app.command("taxi-out")
def taxi_out_cmd(
    airport: str | None = typer.Option(None, "--airport", "-a", help="Optional airport ICAO filter (e.g. EGLL, LFPG)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Optional date filter (YYYY-MM-DD)"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
    min_flights: int = typer.Option(50, "--min-flights", help="Minimum departures required to include airport"),
    limit: int = typer.Option(15, "--limit", "-n", help="Top N airports by Additional Taxi-Out Time (ATOT)"),
) -> None:
    """Compute ICAO KPI02 Additional Taxi-Out Time (ATOT), UTOT baselines, and LTO idle emissions across airports."""
    with console.status(f"[cyan]Analyzing surface taxi-out performance & ICAO KPI02 ATOT (Period: {period})..."):
        ds = FlightDataset(period=period)
        res = analyze_airport_taxi_out_performance(
            ds, airport_icao=airport, date_str=date, min_flights=min_flights, top_n=limit
        )

    if not res.get("has_data"):
        console.print(f"[yellow]{res.get('message')}[/yellow]")
        return

    table = Table(
        title=f"🛞 Airport Surface Movement & ICAO KPI02 Additional Taxi-Out Time (ATOT) — Period {period}",
        header_style="bold magenta",
    )
    table.add_column("Airport", style="bold cyan")
    table.add_column("Deps", justify="right")
    table.add_column("UTOT (p20)", justify="right", style="green")
    table.add_column("Actual Taxi", justify="right")
    table.add_column("KPI02 ATOT", justify="right", style="bold red")
    table.add_column("p95 Taxi", justify="right", style="yellow")
    table.add_column("Excess Queue Hrs", justify="right")
    table.add_column("Excess Fuel (t)", justify="right", style="yellow")
    table.add_column("Excess CO₂ (t)", justify="right", style="bold red")
    table.add_column("NOx / CO (kg)", justify="right", style="dim")

    for r in res["airports"]:
        table.add_row(
            str(r["adep"]),
            f"{int(r['departures']):,}",
            f"{r['unimpeded_utot_min']:.1f}m",
            f"{r['avg_actual_taxi_min']:.1f}m",
            f"+{r['avg_additional_atot_min']:.1f}m",
            f"{r['p95_taxi_out_min']:.1f}m",
            f"{r['total_excess_taxi_hours']:,.1f}h",
            f"{r['excess_taxi_fuel_kg'] / 1000.0:,.1f}t",
            f"{r['excess_taxi_co2_kg'] / 1000.0:,.1f}t",
            f"{r['excess_nox_kg']:,.0f} / {r['excess_co_kg']:,.0f}",
        )
    console.print(table)


@app.command("flight-taxi-out")
def flight_taxi_out_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
) -> None:
    """Evaluate single-flight surface taxi-out queue delay (ICAO KPI02 ATOT) and ICAO EDB v32B idle emissions."""
    with console.status(f"[cyan]Evaluating surface taxi-out & LTO idle emissions for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        rep = analyze_flight_taxi_out(ds, ectrl_id=ectrl_id)

    if not rep:
        console.print(f"[yellow]No TAXI_OUT record found for flight {ectrl_id} in period {period}.[/yellow]")
        return

    table = Table(
        title=f"🛞 Single-Flight Surface Taxi-Out & ICAO EDB Idle Emissions — Flight {rep.ectrl_id} ({rep.adep} ➔ {rep.ades})",
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Aircraft & Default Engine", f"{rep.ac_type} (Wake {rep.wake_category}) — {rep.default_engine}")
    table.add_row("Off-Block ➔ Takeoff (UTC)", f"{rep.off_block_time} ➔ {rep.takeoff_time}")
    table.add_row("Actual vs Filed Taxi-Out", f"{rep.actual_taxi_out_min:.1f} min (Filed: {rep.filed_taxi_out_min:.1f} min, Δ {rep.filed_vs_actual_delta_min:+.1f} min)")
    table.add_row("Unimpeded Baseline (UTOT p20)", f"{rep.unimpeded_utot_min:.1f} min at {rep.adep}")
    table.add_row("ICAO KPI02 Additional Taxi-Out (ATOT)", f"[bold red]+{rep.additional_atot_min:.1f} min[/bold red] ({rep.congestion_status})")
    table.add_row("7% Thrust Idle + APU Fuel Rate", f"{rep.idle_fuel_rate_kg_min:.2f} kg/min")
    table.add_row("Total Surface Taxi Fuel Burn", f"{rep.total_taxi_fuel_kg:,.1f} kg")
    table.add_row("Excess Queue Fuel & CO₂ (ATOT)", f"[bold yellow]{rep.excess_atot_fuel_kg:,.1f} kg Jet-A[/bold yellow] ({rep.excess_co2_kg:,.1f} kg CO₂)")
    table.add_row("Excess LTO Gaseous Emissions", f"NOx: {rep.excess_nox_g:,.1f} g | CO: {rep.excess_co_g:,.1f} g | HC: {rep.excess_hc_g:,.1f} g")
    table.add_row("Excess LTO nvPM Soot Mass", f"{rep.excess_nvpm_mg:,.1f} mg")
    table.add_row("Estimated Excess Queue Fuel Cost", f"[bold yellow]€{rep.excess_fuel_cost_eur:,.2f} EUR[/bold yellow] (@ €{rep.fuel_price_eur_per_gallon:.3f}/gal)")

    console.print(table)


@app.command("pru-scorecard")
def pru_scorecard_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 201611, 202510)"),
) -> None:
    """Compute ICAO GANP & Eurocontrol PRU Scorecard (KPI04 KEP, KPI05 KEA, KPI08 ASMA, KPI19 Level-Capping)."""
    with console.status(f"[cyan]Computing ICAO GANP / Eurocontrol PRU Scorecard for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        sc = compute_flight_pru_kpis(ds, ectrl_id=ectrl_id)

    if not sc:
        console.print(f"[red]Flight {ectrl_id} not found in period {period}[/red]")
        raise typer.Exit(code=1)

    table = Table(
        title=f"📏 ICAO GANP & Eurocontrol PRU Performance Scorecard — Flight {sc.ectrl_id} ({sc.adep} ➔ {sc.ades}, {sc.ac_type})",
        header_style="bold magenta",
    )
    table.add_column("PRU / ICAO KPI", style="cyan")
    table.add_column("Measured Value", style="bold green")
    table.add_column("Operational Interpretation", style="yellow")

    table.add_row(
        "Great-Circle O&D vs Achieved En-Route",
        f"{sc.great_circle_od_nm:.1f} nm O&D | {sc.enroute_achieved_dist_nm:.1f} nm (>40nm)",
        "40 nm origin/destination cylinders clipped",
    )
    table.add_row(
        "KPI04 (KEP — Filed En-Route Inefficiency)",
        f"{sc.kpi04_kep_pct:.2f}% ({sc.enroute_filed_dist_nm:.1f} nm)",
        "Flight plan route design vs achieved distance",
    )
    table.add_row(
        "KPI05 (KEA — Actual En-Route Inefficiency)",
        f"{sc.kpi05_kea_pct:.2f}% ({sc.enroute_actual_dist_nm:.1f} nm)",
        sc.enroute_routing_verdict,
    )
    table.add_row(
        "Tactical ATC Shortcut / Vectoring (KEA - KEP)",
        f"{sc.tactical_kea_minus_kep_pct:+.2f}% ({sc.tactical_distance_delta_nm:+.1f} nm)",
        "Direct shortcut (<0) vs tactical vectoring (>0)",
    )
    table.add_row(
        "KPI08 (ASMA 40nm Arrival Transit / Additional)",
        f"{sc.asma_40nm_actual_min:.1f} min (Add: +{sc.kpi08_asma_40nm_additional_min:.1f} min)",
        f"Unimpeded baseline: {sc.asma_40nm_unimpeded_min:.1f}m | 100nm: {sc.asma_100nm_actual_min:.1f}m",
    )
    table.add_row(
        "KPI19 (Cruise Level-Capping: Req vs Max FL)",
        f"FL{sc.requested_fl:.0f} req vs FL{sc.max_actual_fl:.0f} act (Δ -{sc.level_cap_delta_fl:.0f} FL)",
        f"Capped: {sc.is_level_capped} | Excess Fuel: +{sc.level_cap_excess_fuel_kg:.1f} kg",
    )
    table.add_row(
        "Lower FIR vs Upper UIR/UTA Dwell Split",
        f"Lower FIR: {sc.lower_fir_dwell_min:.1f}m | Upper UIR: {sc.upper_uir_dwell_min:.1f}m",
        f"Arrival TMA ({sc.arrival_tma_id or 'N/A'}): {sc.arrival_tma_dwell_min or 0.0:.1f}m",
    )
    console.print(table)


@app.command("cco-climb")
def cco_climb_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 201611, 202510)"),
) -> None:
    """Analyze Continuous Climb Operations (KPI17 CCO), departure SID level-offs, and climb fuel penalty."""
    with console.status(f"[cyan]Analyzing CCO departure climb profile for Flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        rep = analyze_cco_climb(ds, ectrl_id=ectrl_id)

    if not rep:
        console.print(f"[red]Climb profile not found for flight {ectrl_id}[/red]")
        raise typer.Exit(code=1)

    score_color = "green" if rep.cco_compliance_score >= 85 else "yellow" if rep.cco_compliance_score >= 60 else "red"
    console.print(
        f"\n[bold green]Continuous Climb Operations (KPI17 CCO) Report:[/bold green] Flight [cyan]{rep.ectrl_id}[/cyan] "
        f"({rep.adep} ➔ {rep.ades}, {rep.ac_type})"
    )
    console.print(
        f"CCO Compliance Score: [{score_color} bold]{rep.cco_compliance_score:.1f}%[/{score_color} bold] | "
        f"Total Climb Time to FL{rep.cruise_flight_level:.0f}: {rep.total_climb_duration_min:.1f} min"
    )
    console.print(
        f"Climb Level-Offs: [bold red]{rep.inefficiency_events_count} events[/bold red] ({rep.total_level_off_min:.1f} min) | "
        f"Excess Climb Fuel: [bold yellow]{rep.excess_fuel_burned_kg:.1f} kg[/bold yellow] ({rep.excess_co2_kg:.1f} kg CO₂)\n"
    )

    if rep.inefficiency_events:
        table = Table(title="Detected Departure Climb Level-Off Events", header_style="bold magenta")
        table.add_column("Event Type", style="bold red")
        table.add_column("Start Time (UTC)")
        table.add_column("End Time (UTC)")
        table.add_column("Duration", justify="right", style="yellow")
        table.add_column("Flight Level", justify="right", style="cyan")
        table.add_column("Coordinates")
        for ev in rep.inefficiency_events:
            table.add_row(
                ev.event_type,
                ev.start_time,
                ev.end_time,
                f"{ev.duration_min:.1f} min",
                f"FL{ev.flight_level:.0f}",
                f"{ev.latitude:.2f}°, {ev.longitude:.2f}°",
            )
        console.print(table)


@app.command("level-capping")
def level_capping_cmd(
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 201611, 202510)"),
    operator: str | None = typer.Option(None, "--operator", "-op", help="Optional airline ICAO filter"),
    min_dist: float = typer.Option(250.0, "--min-dist", help="Minimum flight distance in nm"),
    limit: int = typer.Option(12, "--limit", "-n", help="Top N corridors by level-capped flights"),
) -> None:
    """Scan the network for KPI19 Cruise Flight-Level Capping (actual peak FL restricted ≥2,000 ft below Requested FL)."""
    with console.status(f"[cyan]Scanning period {period} for KPI19 Cruise Level-Capping..."):
        ds = FlightDataset(period=period)
        res = analyze_network_level_capping(ds, operator=operator, min_distance_nm=min_dist, top_n=limit)

    console.print(
        f"\n[bold green]KPI19 Cruise Flight-Level Capping Summary ({period}):[/bold green] "
        f"Analyzed: [white]{res['analyzed_flights']:,}[/white] | "
        f"Level-Capped (≥20 FL deficit): [bold red]{res['level_capped_flights']:,} ({res['level_capped_pct']:.2f}%)[/bold red] | "
        f"Mean FL Deficit: [yellow]-{res['mean_fl_deficit']:.1f} FL[/yellow]\n"
    )

    table = Table(title=f"Top Level-Capped Corridors (Period {period})", header_style="bold magenta")
    table.add_column("Corridor", style="bold cyan")
    table.add_column("Total Flts", justify="right")
    table.add_column("Capped Flts", justify="right", style="bold red")
    table.add_column("Capped %", justify="right", style="yellow")
    table.add_column("Avg Req FL", justify="right")
    table.add_column("Avg Act Max FL", justify="right", style="green")
    table.add_column("Avg Deficit", justify="right", style="red")

    for r in res["top_capped_corridors"]:
        table.add_row(
            str(r["corridor"]),
            f"{int(r['total_flights']):,}",
            f"{int(r['capped_flights']):,}",
            f"{r['level_capped_pct']:.1f}%",
            f"FL{r['avg_requested_fl']:.0f}",
            f"FL{r['avg_actual_max_fl']:.0f}",
            f"-{r['avg_fl_deficit']:.1f} FL",
        )
    console.print(table)


@app.command("fra-analysis")
def fra_analysis_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 201611, 202510)"),
    tolerance: float = typer.Option(2.5, "--tolerance", "-t", help="Waypoint snap tolerance in nm"),
) -> None:
    """Snap 4D waypoints to ATS airways (raw_routes), RNAV fixes (fix.dat), and navaids (nav.dat) to measure Free Route Airspace (FRA) directness."""
    with console.status(f"[cyan]Snapping 4D waypoints & evaluating Free Route Airspace (FRA) adherence for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        rep = analyze_flight_fra_adherence(ds, ectrl_id=ectrl_id, snap_tolerance_nm=tolerance)

    if not rep:
        console.print(f"[red]Flight {ectrl_id} not found in period {period}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]Free Route Airspace (FRA) & Airway Snapping Report:[/bold green] Flight [cyan]{rep.ectrl_id}[/cyan] "
        f"({rep.adep} ➔ {rep.ades}, {rep.ac_type})"
    )
    console.print(
        f"FRA Directness Index: [bold cyan]{rep.fra_directness_index_pct:.1f}%[/bold cyan] ({rep.fra_direct_distance_nm:.1f} nm DCT) | "
        f"Structured ATS Airway Adherence: [yellow]{rep.ats_airway_adherence_pct:.1f}%[/yellow] ({rep.ats_airway_distance_nm:.1f} nm)"
    )
    console.print(
        f"En-Route Waypoints: {rep.total_enroute_waypoints} "
        f"(Airway: {rep.airway_waypoints_count}, FRA Fix DCT: {rep.fra_fix_dct_waypoints_count}, Radar DCT: {rep.tactical_dct_waypoints_count})"
    )
    if rep.distinct_airways_used:
        console.print(f"[green]Snapped ATS Airways:[/green] {', '.join(rep.distinct_airways_used)}")
    if rep.sample_snapped_fixes:
        console.print(f"[cyan]Snapped 5-Letter Fixes / Navaids:[/cyan] {', '.join(rep.sample_snapped_fixes)}\n")


@app.command("radio-workload")
def radio_workload_cmd(
    airport: str = typer.Option("EGLL", "--airport", "-a", help="Airport ICAO code (e.g. EGLL, LFPG, EHAM)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Optional date filter (YYYY-MM-DD)"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 202510)"),
) -> None:
    """Estimate ATC Controller-Pilot VHF radio frequency workload and voice channel occupancy (%)."""
    with console.status(f"[cyan]Estimating VHF radio frequency workload for {airport.upper()} ({period})..."):
        ds = FlightDataset(period=period)
        res = estimate_airport_and_sector_radio_workload(ds, airport_icao=airport, date_str=date)

    console.print(
        f"\n[bold green]VHF Radio Frequency Workload & Communications Occupancy:[/bold green] [cyan]{res['airport_icao']}[/cyan] (Period {period})"
    )
    console.print(
        f"Peak Hourly Movements: [bold white]{res['peak_hourly_movements']} mvts/hr[/bold white] (Avg: {res['avg_hourly_movements']} mvts/hr) | "
        f"Cataloged VHF Channels: {res['total_vhf_channels_cataloged']} | Terminal ILS/VOR/DME Navaids: {res['associated_ils_vor_dme_count']}"
    )
    console.print(
        f"Primary Communications Bottleneck: [bold red]{res['primary_comms_bottleneck']}[/bold red] "
        f"(Peak Voice Occupancy: {res['peak_voice_occupancy_pct']:.1f}%)\n"
    )

    table = Table(title=f"ATC Position VHF Channel Allocation & Peak Occupancy ({res['airport_icao']})", header_style="bold magenta")
    table.add_column("ATC Position", style="bold cyan")
    table.add_column("Effective VHF Ch", justify="right")
    table.add_column("Sample Frequencies (MHz)", style="dim")
    table.add_column("Peak Mvts / Ch", justify="right", style="yellow")
    table.add_column("Est. Voice Occupancy %", justify="right", style="bold red")
    table.add_column("Status", style="green")

    for p in res["positions"]:
        freqs_str = ", ".join(f"{f:.3f}" for f in p["sample_frequencies_mhz"]) or "Regional TRACON"
        table.add_row(
            p["atc_position"],
            str(p["effective_channels"]),
            freqs_str,
            f"{p['peak_movements_per_channel']:.1f}",
            f"{p['estimated_voice_occupancy_pct']:.1f}%",
            p["workload_status"],
        )
    console.print(table)


@app.command("crco-cost")
def crco_cost_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period (e.g. 201611, 202510)"),
) -> None:
    """Compute Eurocontrol CRCO En-Route Navigation Charges (EUR) and Total Direct Operating Cost (Fuel EUR + Navigation EUR)."""
    with console.status(f"[cyan]Calculating Eurocontrol CRCO route charges & direct operating cost for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        rep = compute_flight_crco_charges(ds, ectrl_id=ectrl_id)

    if not rep:
        console.print(f"[red]Flight {ectrl_id} not found in period {period}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]Eurocontrol CRCO En-Route Charges & Direct Route Economics:[/bold green] "
        f"Flight [cyan]{rep.ectrl_id}[/cyan] ({rep.adep} ➔ {rep.ades}, {rep.ac_type})"
    )
    console.print(
        f"Aircraft MTOW: [white]{rep.mtow_tons:.1f} t[/white] (Weight Factor: [cyan]{rep.crco_weight_factor:.3f}[/cyan]) | "
        f"Distance: [white]{rep.actual_distance_nm:.1f} nm ({rep.actual_distance_km:.1f} km)[/white] | "
        f"Total Service Units: [yellow]{rep.total_service_units:.2f}[/yellow]"
    )
    console.print(
        f"Actual CRCO Route Charge: [bold cyan]€{rep.actual_crco_charge_eur:,.2f} EUR[/bold cyan] (Filed: €{rep.filed_crco_charge_eur:,.2f} EUR, Δ €{rep.crco_charge_delta_eur:+,.2f}) | "
        f"Est. Jet-A Fuel Cost: [bold yellow]€{rep.estimated_total_fuel_cost_eur:,.2f} EUR[/bold yellow]"
    )
    console.print(
        f"Total Direct Route Cost (Fuel + CRCO): [bold green]€{rep.total_direct_route_cost_eur:,.2f} EUR[/bold green] "
        f"(CRCO Share: {rep.crco_share_of_direct_cost_pct:.1f}%)\n"
    )

    table = Table(title="FIR Charging Zone Breakdown (Eurocontrol CRCO Unit Rates in EUR)", header_style="bold magenta")
    table.add_column("FIR ID", style="bold cyan")
    table.add_column("Zone", justify="center", style="yellow")
    table.add_column("State / Charging Zone", style="white")
    table.add_column("Dwell", justify="right")
    table.add_column("Est Dist (km)", justify="right")
    table.add_column("Service Units", justify="right")
    table.add_column("Unit Rate (€)", justify="right", style="cyan")
    table.add_column("Charge (€)", justify="right", style="bold green")

    for b in rep.fir_breakdown:
        table.add_row(
            b.fir_id,
            b.zone_code,
            b.country_name,
            f"{b.dwell_min:.1f}m",
            f"{b.estimated_distance_km:.1f}",
            f"{b.service_units:.2f}",
            f"€{b.unit_rate_eur:.2f}",
            f"€{b.charge_eur:,.2f}",
        )
    console.print(table)


@app.command("route-economics")
def route_economics_cmd(
    adep: str = typer.Option("EGLL", "--adep", help="Origin airport ICAO"),
    ades: str = typer.Option("LTFM", "--ades", help="Destination airport ICAO"),
    ac_type: str | None = typer.Option(None, "--ac", help="Optional aircraft type filter"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period"),
    limit: int = typer.Option(15, "--limit", "-n", help="Max flights to compare"),
) -> None:
    """Compare CRCO route charges vs. Jet-A fuel burn across a corridor (in EUR) to detect Route-Charge Avoidance detours."""
    with console.status(f"[cyan]Analyzing CRCO route charges vs. fuel burn on {adep} ➔ {ades} ({period})..."):
        ds = FlightDataset(period=period)
        res = detect_route_charge_avoidance(ds, adep=adep, ades=ades, ac_type=ac_type, limit=limit)

    console.print(
        f"\n[bold green]Corridor Route Economics (CRCO Navigation Charges vs. Fuel Burn in EUR):[/bold green] "
        f"[cyan]{res['corridor']}[/cyan] (Analyzed: {res['analyzed_flights']} flights, Avoidance Candidates: {res.get('avoidance_candidates_count', 0)})\n"
    )

    table = Table(title=f"Corridor Direct Cost Breakdown in Euros ({res['corridor']})", header_style="bold magenta")
    table.add_column("Flight ID", style="cyan")
    table.add_column("Operator", style="yellow")
    table.add_column("AC", style="white")
    table.add_column("Dist (nm)", justify="right")
    table.add_column("CRCO Fee (€)", justify="right", style="cyan")
    table.add_column("Fuel Cost (€)", justify="right", style="yellow")
    table.add_column("Total Direct (€)", justify="right", style="bold green")
    table.add_column("Top Cost FIR", style="dim")

    for f in res["flights"][:12]:
        table.add_row(
            str(f["ectrl_id"]),
            str(f["ac_operator"]),
            str(f["ac_type"]),
            f"{f['actual_distance_nm']:.1f}",
            f"€{f['crco_charge_eur']:,.2f}",
            f"€{f['fuel_cost_eur']:,.2f}",
            f"€{f['total_direct_cost_eur']:,.2f}",
            str(f["most_expensive_fir"] or "—"),
        )
    console.print(table)


@app.command("conflict-scan")
def conflict_scan_cmd(
    date: str | None = typer.Option(None, "--date", "-d", help="Date YYYY-MM-DD (defaults to first day of period)"),
    hour: int = typer.Option(10, "--hour", "-h", help="UTC hour (0-23)"),
    horiz_nm: float = typer.Option(5.0, "--horiz-nm", help="Horizontal separation threshold in nm"),
    vert_fl: float = typer.Option(10.0, "--vert-fl", help="Vertical separation threshold in Flight Levels (10 FL = 1,000 ft)"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period"),
    limit: int = typer.Option(15, "--limit", "-n", help="Max proximity events to display"),
) -> None:
    """Scan 4D trajectories for co-altitude aircraft proximity events (<5 nm, <=1,000 ft) and compute airspace complexity."""
    with console.status(f"[cyan]Scanning 4D airspace trajectories for co-altitude proximity events ({period}, {hour:02d}:00 UTC)..."):
        ds = FlightDataset(period=period)
        rep = scan_airspace_proximity_events(
            ds, date_str=date, hour=hour, horiz_sep_nm=horiz_nm, vert_sep_fl=vert_fl, limit=limit
        )
        comp = compute_sector_complexity_index(ds, date_str=rep.date_str, hour=hour)

    console.print(
        f"\n[bold green]4D Co-Altitude Airspace Proximity & Complexity Scan:[/bold green] "
        f"[cyan]{rep.date_str} @ {rep.hour_utc:02d}:00 UTC[/cyan] (Threshold: ≤{horiz_nm} nm / ≤{vert_fl*100:.0f} ft)"
    )
    console.print(
        f"Proximity Pairs Detected: [bold yellow]{rep.events_detected}[/bold yellow] | "
        f"Loss-of-Separation Alerts (<3nm/<1000ft): [bold red]{rep.loss_of_sep_alerts}[/bold red]\n"
    )

    if rep.events:
        table = Table(title="Detected Co-Altitude Aircraft Proximity Events", header_style="bold magenta")
        table.add_column("Flight 1 (Op/AC)", style="cyan")
        table.add_column("Flight 2 (Op/AC)", style="cyan")
        table.add_column("Time (UTC)", style="dim")
        table.add_column("Horiz Sep", justify="right", style="bold red")
        table.add_column("Vert Sep", justify="right", style="yellow")
        table.add_column("Flight Levels", justify="center")
        table.add_column("Severity", style="white")

        for e in rep.events[:12]:
            table.add_row(
                f"{e.flight_1_id} ({e.operator_1}/{e.ac_type_1})",
                f"{e.flight_2_id} ({e.operator_2}/{e.ac_type_2})",
                e.timestamp_utc[11:19],
                f"{e.horizontal_sep_nm:.2f} nm",
                f"{e.vertical_sep_ft:.0f} ft",
                f"FL{e.fl_1:.0f}/FL{e.fl_2:.0f}",
                e.severity,
            )
        console.print(table)

    if comp.get("sectors"):
        c_table = Table(title="Top Dynamic Complexity Grid Sectors (5°x5° Cells)", header_style="bold cyan")
        c_table.add_column("Grid Sector", style="bold white")
        c_table.add_column("Active Flights", justify="right")
        c_table.add_column("Vertical Climb/Desc %", justify="right", style="yellow")
        c_table.add_column("FL Dispersion", justify="right")
        c_table.add_column("Complexity Score", justify="right", style="bold green")
        for s in comp["sectors"][:6]:
            c_table.add_row(
                str(s["grid_sector"]),
                str(s["active_flights"]),
                f"{s['vertical_transition_pct']:.1f}%",
                f"±{s['fl_dispersion']:.1f} FL",
                f"{s['dynamic_complexity_score']:.1f}",
            )
        console.print(c_table)


@app.command("runway-safety")
def runway_safety_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period"),
) -> None:
    """Assess arrival runway braking action, density altitude, crosswind limits, and landing distance safety margin."""
    with console.status(f"[cyan]Assessing arrival runway braking & landing distance safety margin for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        rep = assess_flight_runway_safety(ds, metar_client=None, ectrl_id=ectrl_id, fetch_live_metar=False)

    if not rep:
        console.print(f"[red]Flight {ectrl_id} not found in period {period}[/red]")
        raise typer.Exit(code=1)

    table = Table(
        title=f"🛬 Arrival Runway Safety, Density Altitude & Braking Margin — Flight {rep.ectrl_id} (➔ {rep.ades})",
        header_style="bold magenta",
    )
    table.add_column("Safety Parameter", style="cyan")
    table.add_column("Measured / Computed Value", style="bold green")

    table.add_row("Aircraft & Wake Category", f"{rep.ac_type} (Wake {rep.wake_category}) — Operator: {rep.ac_operator}")
    table.add_row("Recommended Arrival Runway", f"RWY {rep.recommended_runway} (Hdg {rep.runway_heading_deg:.0f}°, Surface: {rep.runway_surface_type})")
    table.add_row("ILS Precision Category", rep.ils_precision_category)
    table.add_row("Field Elevation / Pressure / Density Alt", f"Elev: {rep.airport_elevation_ft:.0f} ft | PA: {rep.pressure_altitude_ft:.0f} ft | DA: {rep.density_altitude_ft:.0f} ft (ISA {rep.isa_temp_deviation_c:+.1f}°C)")
    table.add_row("Runway Surface & Braking Code", f"{rep.runway_surface_condition} — ICAO Code: {rep.braking_action_code}")
    table.add_row("Runway Wind Components", f"Headwind: {rep.headwind_kts:.1f} kts | Crosswind: {rep.crosswind_kts:.1f} kts | Tailwind: {rep.tailwind_kts:.1f} kts")
    table.add_row("Available Landing Distance (LDA)", f"{rep.available_lda_m:,.0f} m ({rep.available_lda_ft:,.0f} ft)")
    table.add_row("Required Factored Landing Distance", f"{rep.factored_required_landing_dist_m:,.0f} m (Unfactored: {rep.unfactored_landing_dist_m:,.0f} m)")
    table.add_row("Stopping Safety Margin & LDA Usage", f"+{rep.landing_distance_safety_margin_m:,.0f} m margin ({rep.lda_utilization_pct:.1f}% of LDA)")
    table.add_row("Overall Runway Safety Risk", f"[bold yellow]{rep.safety_risk_level}[/bold yellow]")
    console.print(table)


@app.command("contrail-impact")
def contrail_impact_cmd(
    ectrl_id: int = typer.Argument(..., help="Eurocontrol Flight ID"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period"),
) -> None:
    """Evaluate upper-air wind-optimal routing (TAS vs GS) and persistent contrail ISSR formation & avoidance."""
    with console.status(f"[cyan]Analyzing upper-air winds & Schmidt-Appleman contrail ISSR profile for flight {ectrl_id}..."):
        ds = FlightDataset(period=period)
        rep = analyze_flight_wind_and_contrails(ds, ectrl_id=ectrl_id)

    if not rep:
        console.print(f"[red]Flight {ectrl_id} not found in period {period}[/red]")
        raise typer.Exit(code=1)

    table = Table(
        title=f"🌬️ Upper-Air Winds & Persistent Contrail (ISSR) Climate Analysis — Flight {rep.ectrl_id} ({rep.adep} ➔ {rep.ades})",
        header_style="bold magenta",
    )
    table.add_column("Upper-Air / Climate Metric", style="cyan")
    table.add_column("Value & Tactical Guidance", style="bold green")

    table.add_row("Avg Groundspeed vs True Airspeed", f"GS: {rep.avg_groundspeed_kts:.1f} kts | TAS: {rep.avg_true_airspeed_kts:.1f} kts")
    table.add_row("Net Upper-Air Wind Component", f"{rep.avg_tailwind_component_kts:+.1f} kts ({rep.wind_routing_verdict})")
    table.add_row("Wind Time Savings vs Still Air", f"{rep.wind_time_savings_min:+.1f} min")
    table.add_row("Persistent Contrail (ISSR) Exposure", f"[bold yellow]{rep.persistent_contrail_distance_nm:.1f} nm ({rep.persistent_contrail_pct:.1f}% of cruise)[/bold yellow] ({rep.persistent_contrail_duration_min:.1f} min)")
    table.add_row("Direct CO₂ vs Contrail Cirrus CO₂e", f"Direct CO₂: {rep.direct_co2_tons:.2f} t | Contrail GWP100: +{rep.contrail_co2e_tons:.2f} t CO₂e ({rep.total_climate_co2e_multiplier:.2f}x total)")
    console.print(table)

    if rep.issr_segments:
        s_table = Table(title="Detected Ice Supersaturated Region (ISSR) Contrail Segments & ±2,000 ft Avoidance", header_style="bold cyan")
        s_table.add_column("Seq", style="cyan")
        s_table.add_column("Cruise FL", justify="right")
        s_table.add_column("Dist (nm)", justify="right")
        s_table.add_column("Temp (°C)", justify="right")
        s_table.add_column("RHi (%)", justify="right", style="bold red")
        s_table.add_column("Recommended Tactical Altitude Change", style="bold yellow")
        for seg in rep.issr_segments[:8]:
            s_table.add_row(
                f"{seg.seq_start}➔{seg.seq_end}",
                f"FL{seg.flight_level:.0f}",
                f"{seg.distance_nm:.1f}",
                f"{seg.temperature_c:.1f}°C",
                f"{seg.rh_ice_pct:.1f}%",
                seg.alt_deviation_note,
            )
        console.print(s_table)


@app.command("optimize-tail-swaps")
def optimize_tail_swaps_cmd(
    operator: str = typer.Option("RYR", "--operator", "-op", help="Airline ICAO code (e.g. RYR, BAW, DLH, EZY)"),
    airport: str | None = typer.Option(None, "--airport", "-a", help="Optional hub airport ICAO (or auto-detect top base)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Date YYYY-MM-DD"),
    period: str = typer.Option("202510", "--period", "-p", help="Eurocontrol period"),
) -> None:
    """Prescriptive AI optimizer recommending same-type aircraft tail swaps at hubs to absorb reactionary delays."""
    with console.status(f"[cyan]Solving prescriptive tail-swap recovery schedule for {operator.upper()} ({period})..."):
        ds = FlightDataset(period=period)
        res = optimize_hub_tail_swaps(ds, operator=operator, hub_airport=airport, date_str=date)

    console.print(
        f"\n[bold green]Prescriptive Hub Tail-Swap Recovery Optimizer:[/bold green] "
        f"[cyan]{res.operator}[/cyan] ({res.date_str}, Period {res.period})"
    )
    console.print(
        f"Rotations Analyzed: {res.candidate_rotations_analyzed} | "
        f"Recommended Tail Swaps: [bold cyan]{res.recommended_swaps_count}[/bold cyan] | "
        f"Reactionary Delay Saved: [bold green]{res.total_delay_minutes_saved:.1f} min[/bold green] | "
        f"Est. Cost Saved: [bold yellow]€{res.estimated_cost_saved_eur:,.2f} EUR[/bold yellow]\n"
    )

    if res.swaps:
        table = Table(title="Recommended Same-Type Aircraft Tail Swaps", header_style="bold magenta")
        table.add_column("Hub", style="cyan")
        table.add_column("AC", style="white")
        table.add_column("Delayed Tail ⇄ Donor Tail", style="bold yellow")
        table.add_column("Protected Leg (STD)", style="white")
        table.add_column("Base Delay", justify="right", style="red")
        table.add_column("Post-Swap", justify="right", style="green")
        table.add_column("Net Saved", justify="right", style="bold green")
        table.add_column("Cost Saved (€)", justify="right", style="yellow")

        for s in res.swaps[:12]:
            table.add_row(
                s.hub_airport,
                s.ac_type,
                f"{s.delayed_tail_reg} ⇄ {s.donor_tail_reg}",
                f"➔{s.delayed_outbound_ades} ({s.delayed_outbound_std[11:16]})",
                f"+{s.baseline_reactionary_delay_min:.0f}m",
                f"+{s.post_swap_reactionary_delay_min:.0f}m",
                f"-{s.net_delay_saved_min:.0f}m",
                f"€{s.estimated_cost_saved_eur:,.0f}",
            )
        console.print(table)


@app.command("dashboard")
def dashboard_cmd(
    port: int = typer.Option(8501, "--port", "-p", help="Port for the Streamlit dashboard"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host address"),
) -> None:
    """Launch the interactive FlightML Simulation & Analytics Dashboard."""
    import subprocess
    import sys
    dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    console.print(f"[bold green]🚀 Launching FlightML Simulation Dashboard on http://{host}:{port}[/bold green]")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    subprocess.run(cmd)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Launch the FlightML FastAPI REST service."""
    console.print(f"[bold green]Starting FlightML API at http://{host}:{port}[/bold green]")
    console.print(f"API Docs available at: [cyan]http://{host}:{port}/docs[/cyan]")
    uvicorn.run("flightml.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
