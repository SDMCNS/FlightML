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
    is_point_on_land,
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

    # User mandate: Primary station inventory for FIR must contain ONLY stations inside the FIR
    egtt_poly, _, _ = get_fir_geometry(ds, "EGTTFIR")
    from shapely.geometry import Point
    for s in rep_egtt["stations"]:
        assert egtt_poly.contains(Point(s["longitude"], s["latitude"])), (
            f"Station {s['id']} at ({s['latitude']}, {s['longitude']}) is outside EGTTFIR!"
        )

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

    # Test Adjacent FIR Infill for EGTTFIR DME (which has North Sea boundary gaps)
    rep_egtt_dme = get_ecac_rationalization_report(
        ds,
        station_type="DME",
        active_view="VIEW_1",
        target_fir="EGTTFIR",
    )
    assert "adjacent_fir_infill_suggestions" in rep_egtt_dme
    assert len(rep_egtt_dme["adjacent_fir_infill_suggestions"]) > 0
    # Verify adjacent stations are outside EGTTFIR and provide cross-border sharing
    for adj_s in rep_egtt_dme["adjacent_fir_infill_suggestions"]:
        assert not egtt_poly.contains(Point(adj_s["longitude"], adj_s["latitude"]))
        assert adj_s["cells_covered_count"] > 0
    assert "methodology" in rep_egtt_dme


def test_synthetic_gap_fillers_land_constraint():
    """
    Ensure all proposed synthetic gap fillers are strictly located on land,
    and that uncoverable oceanic cells that cannot be covered from land
    are never assigned impossible offshore stations.
    """
    # 1. Direct test of deep oceanic cell (Atlantic Ocean far from land)
    deep_ocean_cell = [(45.0, -18.0)]
    ocean_synthetics = compute_synthetic_gap_fillers(deep_ocean_cell, station_type="RADAR")
    # Must NOT propose any station because it cannot be placed on land
    assert len(ocean_synthetics) == 0

    # 2. Direct test with a continental land gap cell (Central Germany)
    land_cell = [(50.5, 9.5)]
    land_synthetics = compute_synthetic_gap_fillers(land_cell, station_type="RADAR")
    assert len(land_synthetics) == 1
    assert is_point_on_land(land_synthetics[0].latitude, land_synthetics[0].longitude)

    # 3. Network-wide report validation: 100% of proposed infill stations MUST be on land
    ds = FlightDataset("202510")
    radar_rep = get_ecac_rationalization_report(ds, station_type="RADAR", active_view="VIEW_1")
    for gap in radar_rep["synthetic_gap_fillers"]:
        assert is_point_on_land(gap["latitude"], gap["longitude"]), (
            f"Proposed RADAR infill {gap['id']} at ({gap['latitude']}, {gap['longitude']}) is not on land!"
        )

    dme_rep = get_ecac_rationalization_report(ds, station_type="DME", active_view="VIEW_1")
    for gap in dme_rep["synthetic_gap_fillers"]:
        assert is_point_on_land(gap["latitude"], gap["longitude"]), (
            f"Proposed DME infill {gap['id']} at ({gap['latitude']}, {gap['longitude']}) is not on land!"
        )


def test_navigation_performance_standards_scenarios():
    """
    Test DME rationalization across Navigation Performance Standards:
    RNAV 5, RNAV 2, RNAV 1, RNP 1 with 10 NM grid performance classification (Green/Orange/Red).
    """
    ds = FlightDataset("202510")

    # RNAV 5 (En-Route Upper Airspace)
    rep_rnav5 = get_ecac_rationalization_report(
        ds,
        station_type="DME",
        active_view="RNAV_5",
        target_fir="EGTTFIR",
    )
    assert rep_rnav5["active_view"] == "RNAV_5"
    fl200_data = rep_rnav5["coverage_by_fl"]["fl200"]
    assert "met_cells_count" in fl200_data
    assert "at_risk_cells_count" in fl200_data
    assert "outage_cells" in fl200_data
    assert fl200_data["met_pct"] > 80.0  # London FIR has high RNAV 5 compliance

    # Verify 10 NM cell attributes
    for c in fl200_data["cells"][:20]:
        assert c["status"] in ("MET", "AT_RISK", "CANNOT_BE_MET", "UNCOVERABLE")
        assert c["cell_size_nm"] == 10.0
        assert "geom_tip" in c

    # RNAV 1 (Terminal P-RNAV SIDs/STARs with N-1 Redundancy)
    rep_rnav1 = get_ecac_rationalization_report(
        ds,
        station_type="DME",
        active_view="RNAV_1",
        target_fir="EGTTFIR",
    )
    assert rep_rnav1["active_view"] == "RNAV_1"
    # RNAV 1 requires higher station density (N-1 redundancy >=3 DMEs) than RNAV 5
    assert rep_rnav1["active_stations_count"] >= rep_rnav5["active_stations_count"]

    # Verify Folium map generates with 10 NM rectangles
    folium_map = create_ecac_coverage_map(rep_rnav5, selected_fl=200)
    assert folium_map is not None


def test_radar_coverage_gap_assumed_stations():
    """
    Test that commercial airports in initial radar coverage gaps are augmented
    with assumed radars:
    - Exactly 1 radar (ABCD_assumed) for commercial airports in coverage gaps.
    - Exactly 2 radars (ABCD_assumed and ABCD_assumed_2) for gap airports with >500 movements/day.
    - No assumed radars for airports already covered in the initial dataset (e.g. EGLL, LFPG, EDDF).
    - In single FIR evaluations (e.g. LRBBFIR), assumed airport radars inside the FIR are scoped to inside_stations.
    """
    ds = FlightDataset("202510")
    radars, airports = load_ecac_stations(ds, station_type="RADAR")

    # 1. High-density gap hubs (>500 movements/day) must have 2 assumed radars
    busy_gap_hubs = ["LIRF", "LGAV", "LIMC", "ENGM", "ESSA"]
    for icao in busy_gap_hubs:
        matched = [s for s in radars if s.id.startswith(f"{icao}_assumed")]
        assert len(matched) == 2, f"Expected 2 assumed radars for {icao}, found {len(matched)}"
        ids = {s.id for s in matched}
        assert f"{icao}_assumed" in ids
        assert f"{icao}_assumed_2" in ids
        for s in matched:
            assert s.type == "RADAR"
            assert s.is_airport_station is True
            assert s.nearest_airport_ident == icao
            assert s.nominal_range_nm == 250.0

    # 2. Medium commercial gap airports (<500 movements/day) must have exactly 1 assumed radar
    medium_gap_airports = ["LROP", "ENVA", "LUKK", "LRIA"]
    for icao in medium_gap_airports:
        matched = [s for s in radars if s.id.startswith(f"{icao}_assumed")]
        assert len(matched) == 1, f"Expected 1 assumed radar for {icao}, found {len(matched)}"
        assert matched[0].id == f"{icao}_assumed"
        assert matched[0].type == "RADAR"
        assert matched[0].is_airport_station is True

    # 3. Already covered hubs must NOT have assumed radars
    covered_hubs = ["EGLL", "LFPG", "EDDF", "EHAM"]
    for icao in covered_hubs:
        matched = [s for s in radars if s.id.startswith(f"{icao}_assumed")]
        assert len(matched) == 0, f"Expected 0 assumed radars for {icao}, found {len(matched)}"

    # 4. Single FIR territorial scoping: LRBBFIR (Bucharest) gets inside assumed radars
    geom = get_fir_geometry(ds, "LRBBFIR")
    assert geom is not None
    poly, _, bbox = geom
    s_in, a_fir, s_adj = load_ecac_stations(
        ds,
        station_type="RADAR",
        fir_poly=poly,
        fir_bbox=bbox,
        include_adjacent=True,
    )
    assert len(s_in) > 0
    in_assumed = [s for s in s_in if s.id.endswith("_assumed") or "_assumed_" in s.id]
    assert len(in_assumed) > 0
    assert any(s.id == "LROP_assumed" for s in s_in)



