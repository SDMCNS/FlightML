"""
Unit tests for ECAC Station Rationalization & Coverage Architecture.
"""
from flightml.data.dataset import FlightDataset
from flightml.eval.ecac_station_rationalization import (
    load_ecac_stations,
    generate_ecac_grid,
    compute_static_rationalization_views,
    compute_synthetic_gap_fillers,
    get_ecac_rationalization_report,
    list_available_ecac_firs,
    get_fir_geometry,
)
from flightml.viz.map_builder import create_ecac_coverage_map


def test_ecac_station_loading():
    ds = FlightDataset("202510")
    radars, airports = load_ecac_stations(ds, station_type="RADAR")
    assert len(radars) > 100
    assert len(airports) > 50

    dmes, _ = load_ecac_stations(ds, station_type="DME")
    assert len(dmes) > 500


def test_ecac_radar_static_views():
    ds = FlightDataset("202510")
    radars, airports = load_ecac_stations(ds, station_type="RADAR")
    grid = generate_ecac_grid(step_deg=0.50)
    views = compute_static_rationalization_views(radars, grid, airports)

    assert "VIEW_1" in views
    assert "VIEW_2" in views
    assert "VIEW_3" in views
    # View 2 must have at least as many stations as View 1
    assert len(views["VIEW_2"]) >= len(views["VIEW_1"])
    # View 3 must have at least as many stations as View 2
    assert len(views["VIEW_3"]) >= len(views["VIEW_2"])


def test_ecac_radar_report_pipeline():
    ds = FlightDataset("202510")
    rep = get_ecac_rationalization_report(ds, station_type="RADAR", active_view="VIEW_1")
    assert rep["station_type"] == "RADAR"
    assert rep["active_stations_count"] > 0
    assert rep["deactivated_stations_count"] > 0
    assert rep["annual_cost_savings_eur"] > 0
    assert "coverage_by_fl" in rep
    assert "fl50" in rep["coverage_by_fl"]
    assert "fl100" in rep["coverage_by_fl"]
    assert "fl200" in rep["coverage_by_fl"]
    assert "fl300" in rep["coverage_by_fl"]
    assert len(rep["synthetic_gap_fillers"]) > 0

    # Test map generation at all 4 flight levels
    for fl in [50, 100, 200, 300]:
        folium_map = create_ecac_coverage_map(rep, selected_fl=fl)
        assert folium_map is not None


def test_ecac_dme_report_pipeline():
    ds = FlightDataset("202510")
    rep = get_ecac_rationalization_report(ds, station_type="DME", active_view="VIEW_2")
    assert rep["station_type"] == "DME"
    assert rep["active_stations_count"] > 0
    assert rep["deactivated_stations_count"] > 0
    assert len(rep["synthetic_gap_fillers"]) > 0

    # Test map generation at all 4 flight levels
    for fl in [50, 100, 200, 300]:
        folium_map = create_ecac_coverage_map(rep, selected_fl=fl)
        assert folium_map is not None


def test_ecac_user_augmentation():
    ds = FlightDataset("202510")
    base_rep = get_ecac_rationalization_report(ds, station_type="RADAR", active_view="VIEW_1")
    active_ids = [s["id"] for s in base_rep["stations"] if s["is_active"]]
    assert len(active_ids) > 0

    # Force deactivation of 2 active stations
    to_deact = tuple(active_ids[:2])
    aug_rep = get_ecac_rationalization_report(
        ds,
        station_type="RADAR",
        active_view="VIEW_1",
        custom_inactive_ids=to_deact,
    )
    assert aug_rep["active_stations_count"] == base_rep["active_stations_count"] - 2
    assert aug_rep["deactivated_stations_count"] == base_rep["deactivated_stations_count"] + 2


if __name__ == "__main__":
    test_ecac_station_loading()
    test_ecac_radar_static_views()
    test_ecac_radar_report_pipeline()
    test_ecac_dme_report_pipeline()
    test_ecac_user_augmentation()
    print("All ECAC rationalization unit tests PASSED successfully!")


def test_ecac_single_fir_rationalization():
    ds = FlightDataset("202510")
    fir_list = list_available_ecac_firs(ds)
    assert len(fir_list) >= 10
    assert fir_list[0][0] == "ALL"

    # Test London FIR (EGTTFIR) Radar rationalization
    rep_egtt = get_ecac_rationalization_report(
        ds,
        station_type="RADAR",
        active_view="VIEW_1",
        target_fir="EGTTFIR",
    )
    assert rep_egtt["target_fir"] == "EGTTFIR"
    assert rep_egtt["fir_boundary_coords"] is not None
    assert len(rep_egtt["fir_boundary_coords"]) > 10
    assert len(rep_egtt["airports"]) > 0
    # Every airport returned must have latitude within EGTT bounds [48.8, 55.0]
    for apt in rep_egtt["airports"]:
        assert 48.0 <= apt["latitude"] <= 56.0

    # Test map generation for single FIR with boundary
    folium_map = create_ecac_coverage_map(rep_egtt, selected_fl=200)
    assert folium_map is not None

    # Test Paris FIR (LFFFFIR) DME rationalization View 2
    rep_paris = get_ecac_rationalization_report(
        ds,
        station_type="DME",
        active_view="VIEW_2",
        target_fir="LFFFFIR",
    )
    assert rep_paris["target_fir"] == "LFFFFIR"
    assert rep_paris["active_stations_count"] > 0
    assert len(rep_paris["airports"]) > 0
