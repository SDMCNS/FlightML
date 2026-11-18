from flightml.data.dataset import FlightDataset
from flightml.eval.ecac_station_rationalization import get_ecac_rationalization_report

ds = FlightDataset("202510")
for fir in ["EGTTFIR", "LFFFFIR", "LECMFIR", "EDGGFIR"]:
    for stype in ["RADAR", "DME"]:
        rep = get_ecac_rationalization_report(ds, station_type=stype, active_view="VIEW_1", target_fir=fir)
        fl200 = rep["coverage_by_fl"]["fl200"]
        print(f"{fir:7s} [{stype:5s}] Active: {rep['active_stations_count']:2d}/{rep['total_stations_count']:2d} | "
              f"Savings: €{rep['annual_cost_savings_eur']:,.0f}/yr | "
              f"Airports: {len(rep['airports']):2d} | "
              f"FL200 1x: {fl200['single_coverage_pct']:.1f}% | 2x: {fl200['dual_coverage_pct']:.1f}%")
