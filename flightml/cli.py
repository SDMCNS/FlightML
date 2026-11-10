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
from flightml.eval.cdo_analyzer import analyze_cdo_descent
from flightml.data.archive_ingest import EurocontrolArchiveIngestor, list_available_periods
from flightml.weather.network_weather_impact import analyze_network_weather_impact
from flightml.geo.airac_diff import compare_airac_cycles
from flightml.eval.recurring_series import analyze_recurring_flight_series
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
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
    price_per_kg: float | None = typer.Option(None, "--price", "-p", help="Fuel price per kg in USD (defaults to official EIA weekly spot price for flight date)"),
) -> None:
    """Calculate excess fuel burn, CO2 emissions, and financial penalty from holdings and detour."""
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
    table.add_row("Weekly Jet Fuel Price", f"${impact.fuel_price_usd_per_gallon:.3f}/gal (${impact.fuel_price_usd_per_kg:.4f}/kg)")
    table.add_row("Price Week & Benchmark", f"{impact.fuel_price_week} [{impact.fuel_price_source}]")
    table.add_row("Estimated Excess Fuel Cost", f"[bold yellow]${impact.estimated_excess_fuel_cost_usd:,.2f} USD[/bold yellow]")

    console.print(table)


@app.command("fuel-prices")
def fuel_prices_cmd(
    period: str | None = typer.Option(None, "--period", "-P", help="Filter by Eurocontrol period (e.g. 201611, 202510)"),
) -> None:
    """Display official EIA / IATA historical weekly jet fuel spot prices across dataset periods."""
    df = get_all_weekly_fuel_prices_df()
    if period:
        df = df[df["period_id"] == period.strip()]

    title = f"⛽ Weekly Aviation Jet Fuel Spot Prices (Source: EIA / IATA Benchmark)"
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Period", style="cyan", justify="center")
    table.add_column("Week Range", style="bold white")
    table.add_column("Price ($/gal)", justify="right", style="bold green")
    table.add_column("Price ($/kg)", justify="right", style="yellow")
    table.add_column("Price ($/metric ton)", justify="right", style="bold yellow")
    table.add_column("Benchmark Source", style="dim")

    for _, row in df.iterrows():
        table.add_row(
            str(row["period_id"]),
            f"{row['year']}-W{int(row['week_number']):02d} ({row['week_start']} ➔ {row['week_end']})",
            f"${row['price_usd_per_gallon']:.3f}",
            f"${row['price_usd_per_kg']:.4f}",
            f"${row['price_usd_per_metric_ton']:,.2f}",
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
    price: float = typer.Option(0.85, "--price", "-p", help="Fuel price per kg in USD"),
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
    table.add_row("Direct Fuel Cost Saved", f"[bold yellow]${res.total_fuel_cost_saved_usd:,.2f} USD[/bold yellow]")

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
    table.add_row("Excess Fuel Cost", "$0.00", f"${comp.excess_fuel_cost_usd:,.2f}", f"${comp.excess_fuel_cost_usd:,.2f} USD")

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
