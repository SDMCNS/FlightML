from flightml.data.dataset import FlightDataset
from shapely.geometry import Polygon, Point
import numpy as np

ds = FlightDataset("202510")

# Fetch EGTTFIR points
pts = ds.cursor.execute("""
    SELECT Latitude, Longitude
    FROM raw_fir
    WHERE "Airspace ID" = 'EGTTFIR'
    ORDER BY "Sequence Number" ASC
""").df()

poly = Polygon(zip(pts["Longitude"], pts["Latitude"]))
min_lon, min_lat, max_lon, max_lat = poly.bounds
print("EGTTFIR Bounds:", min_lat, max_lat, min_lon, max_lon)

# Generate grid within FIR bounds
step = 0.25
lats = np.arange(min_lat, max_lat + 1e-5, step)
lons = np.arange(min_lon, max_lon + 1e-5, step * 1.5)

fir_cells = []
for lat in lats:
    for lon in lons:
        if poly.contains(Point(lon, lat)):
            fir_cells.append((round(float(lat), 3), round(float(lon), 3)))

print("Cells inside EGTTFIR polygon:", len(fir_cells))

# Fetch airports inside FIR
apts_df = ds.cursor.execute("""
    SELECT ident, name, latitude_deg, longitude_deg, elevation_ft
    FROM raw_airports
    WHERE latitude_deg BETWEEN ? AND ? AND longitude_deg BETWEEN ? AND ?
      AND type IN ('large_airport', 'medium_airport')
""", [min_lat, max_lat, min_lon, max_lon]).df()

fir_apts = []
for _, row in apts_df.iterrows():
    if poly.contains(Point(float(row["longitude_deg"]), float(row["latitude_deg"]))):
        fir_apts.append(row["ident"])

print(f"Airports inside EGTTFIR ({len(fir_apts)}):", fir_apts[:10])
