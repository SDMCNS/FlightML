from __future__ import annotations

from flightml.data.dataset import FlightDataset
from flightml.data.external_sources import compile_all_external_sources


def test_compile_external_sources() -> None:
    summary = compile_all_external_sources(force=False)
    assert summary["nav_fixes"]["rows"] > 100_000
    assert summary["nav_aids_xplane"]["rows"] > 20_000
    assert summary["openap_aircraft"]["rows"] >= 50
    assert summary["openap_wrap"]["rows"] >= 1000
    assert summary["openap_engines"]["rows"] >= 400
    assert summary["icao_edb_gaseous"]["rows"] >= 900
    assert summary["icao_edb_nvpm"]["rows"] >= 300
    assert summary["crco_unit_rates"]["rows"] >= 40


def test_dataset_external_views_and_lookups() -> None:
    ds = FlightDataset(period="202510")
    ext_counts = ds.get_external_data_summary()
    assert ext_counts["nav_fixes"] > 100_000
    assert ext_counts["nav_aids_xplane"] > 20_000
    assert ext_counts["crco_unit_rates"] == 41

    # 1. Waypoint & Navaid spatial lookup around London Heathrow (EGLL)
    near_df = ds.find_nearest_waypoints(51.4700, -0.4543, radius_nm=10.0, limit=10)
    assert not near_df.empty
    assert "ident" in near_df.columns
    assert "distance_nm" in near_df.columns
    assert near_df["distance_nm"].iloc[0] <= 2.0

    # 2. Aircraft performance & synonym lookup
    b738 = ds.get_aircraft_performance("B738")
    assert b738 is not None
    assert b738["mtow_kg"] == 79000.0
    assert b738["crco_weight_factor"] > 1.2
    assert b738["cd0_clean"] is not None

    # Synonym lookup (B77L -> B77W)
    b77l = ds.get_aircraft_performance("B77L")
    assert b77l is not None
    assert b77l["canonical_type"] == "B77W"
    assert b77l["is_synonym"] is True

    # 3. CRCO unit rate lookup
    eg_rate = ds.get_crco_unit_rate("EGTTFIR")
    assert eg_rate is not None
    assert eg_rate["zone_code"] == "EG"
    assert eg_rate["unit_rate_eur"] == 87.75

    az_rate = ds.get_crco_unit_rate("LPPOFIR")
    assert az_rate is not None
    assert az_rate["zone_code"] == "AZ"
    assert az_rate["unit_rate_eur"] == 8.16

    # 4. Engine emissions lookup (ICAO EDB v32B)
    emis = ds.get_engine_emissions("CFM56")
    assert len(emis["gaseous_matches"]) > 0
    assert len(emis["nvpm_matches"]) > 0

    # 5. AUA traversals in 202510
    if ds.has_aua_traversals():
        aua_df = ds.get_flight_aua_traversals(291262109)
        assert not aua_df.empty
        assert "aua_type" in aua_df.columns
