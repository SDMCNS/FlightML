import sys
import duckdb
from flightml.data.dataset import FlightDataset
from flightml.geo.trajectory import analyze_trajectory
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.cdo_analyzer import analyze_cdo_descent
from flightml.features.fir_saturation import calculate_flight_fir_dwell_inefficiency
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price

print("=== FAST STRESS TESTING ANALYTICAL FUNCTIONS ===", flush=True)

for period in ["201611", "202510"]:
    print(f"\n--- Testing Period: {period} ---", flush=True)
    ds = FlightDataset(period=period)
    
    # Sample 25 flights from diverse operators and aircraft types
    sample_df = ds.query("""
        SELECT ectrl_id, adep, ades, actual_off_block_time, ac_operator, ac_type, ac_reg
        FROM flights 
        WHERE actual_distance_nm > 50 
        LIMIT 25
    """)
    print(f"Sampled {len(sample_df)} flights for period {period}", flush=True)
    
    for idx, row in sample_df.iterrows():
        fid = int(row["ectrl_id"])
        route = f"{row['adep']} -> {row['ades']}"
        label = ds.format_flight_label(row)
        assert str(fid) in label, f"Flight ID {fid} must be in label {label}"
        
        # 1. Trajectory
        trajs = ds.get_trajectory(fid)
        m = analyze_trajectory(trajs["actual"], trajs["planned"])
        assert m.actual_distance_nm >= 0.0
        
        # 2. Fuel & Emissions
        fuel = compute_flight_fuel_and_emissions(ds, fid)
        assert fuel.ectrl_id == fid
        assert fuel.total_excess_fuel_kg >= 0.0
        assert fuel.fuel_price_usd_per_gallon > 0.0
        assert fuel.fuel_price_usd_per_kg > 0.0
        
        # 3. CDO Descent
        cdo = analyze_cdo_descent(ds, fid)
        if cdo:
            assert 0.0 <= cdo.cdo_compliance_score <= 100.0
        
        # 4. FIR Dwell
        fir = calculate_flight_fir_dwell_inefficiency(ds, fid)
        if period == "202510":
            assert fir["has_data"] is True
            assert fir["total_sectors_traversed"] > 0
        
        print(f"  [{idx+1:02d}/25] ECTRL {fid} ({route} {row['ac_operator']}): Dist={m.actual_distance_nm:.0f}nm, FuelCost=${fuel.estimated_excess_fuel_cost_usd:.2f}, CDO={f'{cdo.cdo_compliance_score:.0f}%' if cdo else 'N/A'}", flush=True)

print("\n=== ALL 50 FLIGHTS STRESS TESTED WITH ZERO ERRORS! ===", flush=True)
