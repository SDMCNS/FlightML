from flightml.data.dataset import FlightDataset
from flightml.eval.flight_twins import compare_twin_flights
from flightml.models.airline_stress import stress_test_airline
from flightml.eval.cdo_analyzer import analyze_cdo_descent
from flightml.data.archive_ingest import EurocontrolArchiveIngestor
from flightml.weather.metar_client import MetarClient
from flightml.weather.network_weather_impact import analyze_network_weather_impact
from flightml.geo.airac_diff import compare_airac_cycles
from flightml.eval.recurring_series import analyze_recurring_flight_series


def test_compare_twin_flights():
    ds = FlightDataset()
    comp = compare_twin_flights(ds, adep="RKSI", ades="EGLL", ac_type="B77W")
    assert comp is not None
    assert comp.fast_actual_distance_nm < comp.slow_actual_distance_nm
    assert comp.excess_fuel_burned_kg > 0.0
    assert comp.excess_co2_kg > 0.0


def test_stress_test_airline():
    ds = FlightDataset()
    res = stress_test_airline(ds, operator="BAW", shocks=[15.0, 30.0])
    assert res.operator == "BAW"
    assert res.total_rotations > 0
    assert len(res.fragility_curve) == 2
    assert res.resilience_grade != ""


def test_analyze_cdo_descent():
    ds = FlightDataset()
    report = analyze_cdo_descent(ds, ectrl_id=202007026)
    assert report is not None
    assert report.ectrl_id == 202007026
    assert report.adep == "KORD"
    assert report.ades == "EGLL"
    assert report.cdo_compliance_score >= 0.0
    assert report.total_descent_duration_min > 0.0
    assert len(report.inefficiency_events) > 0


def test_archive_ingestor_classifier(tmp_path):
    ingestor = EurocontrolArchiveIngestor()
    # Create mock files matching typical Eurocontrol filenames
    f_flights = tmp_path / "flights_20251001_20251031.csv.gz"
    f_flights.write_text("dummy")
    f_act = tmp_path / "flight_points_actual_202510.csv.gz"
    f_act.write_text("dummy")
    f_fir = tmp_path / "FIR_2510.csv"
    f_fir.write_text("dummy")

    classified = ingestor.classify_files(tmp_path)
    assert "flights" in classified
    assert "flights_actual" in classified
    assert "fir" in classified


def test_compare_airac_cycles():
    report = compare_airac_cycles("2509", "2510")
    assert report is not None
    assert report.cycle1 == "2509"
    assert report.cycle2 == "2510"
    assert report.route_changes.total_routes_c2 > 20000
    assert report.route_changes.added_routes_count > 0
    assert report.fir_changes.total_firs_c2 > 300


def test_analyze_network_weather_impact():
    ds = FlightDataset()
    client = MetarClient()
    report = analyze_network_weather_impact(
        ds, client, hubs=["EGLL", "LFPG"], start_date="2016-11-01", end_date="2016-11-03"
    )
    assert report is not None
    assert len(report.daily_time_series) == 3
    assert report.total_flights_analyzed > 50000
    assert report.worst_weather_day["date"] != ""


def test_search_flights_and_labels():
    ds = FlightDataset()
    df = ds.search_flights(adep="EGLL", ades="KORD", ac_operator="BAW", limit=5)
    assert not df.empty
    assert len(df) == 5
    assert "label" in df.columns
    sample_label = df.iloc[0]["label"]
    assert "EGLL" in sample_label
    assert "KORD" in sample_label
    assert "BAW" in sample_label
    assert "Flight ID:" in sample_label


def test_recurring_series_analysis():
    ds = FlightDataset()
    rep = analyze_recurring_flight_series(ds, adep="EGLL", ades="KORD", operator="BAW")
    assert rep is not None
    assert rep.total_flights > 10
    assert rep.route == "EGLL ➔ KORD"
    assert rep.best_performing_day["date"] != ""
    assert len(rep.diurnal_banks) > 0

