from flightml.data.dataset import FlightDataset
import pandas as pd

ds = FlightDataset()
df = ds.search_flights(adep="EGLL", ades="KORD", limit=200)
print(f"Found {len(df)} flights for EGLL -> KORD")
print("Labels sample:")
for l in df["label"].head(3):
    print(" ", l)

# Check diurnal breakdown
df["dep_hour"] = pd.to_datetime(df["actual_off_block_time"]).dt.hour
diurnal = df.groupby("dep_hour")[["arr_delay_min", "dep_delay_min", "actual_air_time_min"]].mean().reset_index()
print("\nDiurnal Hour Breakdown:")
print(diurnal.head(5))

