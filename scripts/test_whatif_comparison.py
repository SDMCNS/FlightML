import pandas as pd
from flightml.data.dataset import FlightDataset
from flightml.models.airline_stress import stress_test_airline
from flightml.sim.what_if import simulate_turnaround_mitigation

ds = FlightDataset("201611")
ops = ["RYR", "BAW", "DLH", "EZY"]

print("--- Scenario C Comparison ---")
# Fragility curve dataframe: index=shock, columns=operators
shock_vals = [15, 30, 45, 60, 90]
frag_data = {op: [] for op in ops}
summary_rows = []

for op in ops:
    res = stress_test_airline(ds, op)
    summary_rows.append({
        "Operator": op,
        "Total Rotations": res.total_rotations,
        "Mean Turnaround (min)": round(res.mean_scheduled_turnaround_min, 1),
        "Resilience Grade": res.resilience_grade,
        "50% Shock Threshold": f"+{res.resilience_threshold_shock_min:.0f}m",
    })
    curve_map = {c["injected_shock_min"]: c["infection_rate_pct"] for c in res.fragility_curve}
    for s in shock_vals:
        frag_data[op].append(curve_map.get(s, 0.0))

frag_df = pd.DataFrame(frag_data, index=shock_vals)
print("Multi-airline fragility dataframe (index=shock):")
print(frag_df)
print("\nSummary rows:")
print(pd.DataFrame(summary_rows))

print("\n--- Scenario A Comparison ---")
buf_vals = [5, 10, 15, 20, 25]
buf_data = {op: [] for op in ops}

for op in ops:
    for b in buf_vals:
        sim = simulate_turnaround_mitigation(ds, op, extra_buffer_min=float(b))
        buf_data[op].append(sim["delay_reduction_percentage"])

buf_df = pd.DataFrame(buf_data, index=buf_vals)
print("Multi-airline turnaround buffer sensitivity (index=buffer):")
print(buf_df)
