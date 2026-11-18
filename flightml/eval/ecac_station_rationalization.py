"""
FlightML ECAC Station Rationalization & Coverage Architecture Engine.

Specialized for European Civil Aviation Conference (ECAC) airspace:
- Evaluates Primary/Secondary Radar and Terrestrial DME networks.
- Precomputes 3 progressive static rationalization views:
    * View 1: Minimum Required (>= 1 station in each cell at FL200+ AND >= 1 station at FL10 at airports)
    * View 2: Dual En-Route Redundancy (>= 2 stations in each cell at FL200+ AND >= 1 station at airports)
    * View 3: Low-Level Infill & High-Density Terminal (>= 1 station at FL50+, >= 2 at FL200+, and >= 2 stations at busy airports >= 500 moves/day at FL10)
- Identifies unserviceable baseline gaps as UNCOVERABLE cells (preventing infinite solver loops).
- Computes an ideal synthetic station list (Facility Location problem) to cover gaps with minimum equipment.
- Allows full interactive user augmentation (activating / deactivating individual stations).
- Delivers multi-level coverage maps for FL50, FL100, FL200, and FL300.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Literal
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

from flightml.data.dataset import FlightDataset
import flightml.math as fmath
from flightml.utils.cache import disk_cache

try:
    from global_land_mask import globe
    HAS_GLOBE = True
except ImportError:
    HAS_GLOBE = False


def is_points_on_land(
    latitudes: np.ndarray | list[float],
    longitudes: np.ndarray | list[float],
) -> np.ndarray:
    """
    Vectorized validation that geographic coordinates are located strictly on land.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    if len(lats) == 0:
        return np.zeros(0, dtype=bool)
    if HAS_GLOBE:
        return globe.is_land(lats, lons)
    return np.ones(len(lats), dtype=bool)


def is_point_on_land(latitude: float, longitude: float) -> bool:
    """
    Validation that a single (latitude, longitude) coordinate is located strictly on land.
    """
    if HAS_GLOBE:
        return bool(globe.is_land(latitude, longitude))
    return True

# ==============================================================================
# SURVEILLANCE & NAVIGATION RATIONALIZATION COST & POWER CONSTANTS
# Industry References:
# 1. Eurocontrol Specification for the Rationalisation of Terrestrial Surveillance Infrastructure
# 2. Eurocontrol Navigation Infrastructure Strategy (Conventional Navaids Rationalisation)
# 3. ICAO Doc 9750 (Global Air Navigation Plan - CNS Economic Benchmarks)
# ==============================================================================

HOURS_PER_YEAR = 8760.0

# --- Surveillance Radar Head (PSR/MSSR) ---
# Electrical Power Draw Breakdown:
# - Transmitter racks & high-voltage power supplies: ~5.5 kW
# - Pedestal motor & antenna rotational drive system: ~3.5 kW
# - Signal processing, extractor & UPS charger float losses: ~1.4 kW
# - Equipment shelter HVAC climate control (cooling electronics): ~5.0 kW
# Total continuous average load = 15.41 kW
RADAR_POWER_DRAW_KW = 15.41
RADAR_ANNUAL_POWER_MWH = round((RADAR_POWER_DRAW_KW * HOURS_PER_YEAR) / 1000.0, 1)  # 135.0 MWh/yr

# Annual Operational Expenditure (OPEX) Savings per Decommissioned/Standby Radar Head:
# - Preventative/corrective maintenance & specialized technical labor: €42,000 / yr
# - Periodic airborne flight inspection / calibration sorties (flight check): €22,000 / yr
# - High-reliability leased line telecommunications, site lease & licensing: €13,000 / yr
# - Power utility consumption (~135 MWh @ industrial base rate ~€60/MWh): €8,000 / yr
RADAR_ANNUAL_MAINTENANCE_EUR = 42000.0
RADAR_ANNUAL_FLIGHT_CAL_EUR = 22000.0
RADAR_ANNUAL_COMMS_EUR = 13000.0
RADAR_ANNUAL_ENERGY_EUR = 8000.0
RADAR_ANNUAL_COST_SAVINGS_EUR = (
    RADAR_ANNUAL_MAINTENANCE_EUR + RADAR_ANNUAL_FLIGHT_CAL_EUR + RADAR_ANNUAL_COMMS_EUR + RADAR_ANNUAL_ENERGY_EUR
)  # €85,000 / yr

# --- Distance Measuring Equipment (DME) Ground Transponder ---
# Electrical Power Draw Breakdown:
# - 1 kW peak pulsed solid-state transponder chassis (dual redundant channels): ~1.10 kW
# - Monitor receiver, cavity filters & RF duplexer: ~0.45 kW
# - Battery float charger & inverter standby load: ~0.30 kW
# - Equipment cabinet thermal ventilation & cooling: ~1.00 kW
# Total continuous average load = 2.85 kW
DME_POWER_DRAW_KW = 2.85
DME_ANNUAL_POWER_MWH = round((DME_POWER_DRAW_KW * HOURS_PER_YEAR) / 1000.0, 1)  # 25.0 MWh/yr

# Annual Operational Expenditure (OPEX) Savings per Decommissioned/Standby DME Station:
# - Routine airborne flight calibration & inspection sorties: €18,000 / yr
# - Preventative technician maintenance visits & hardware component spares: €16,000 / yr
# - Telemetry, remote control monitoring lines & mast/site lease: €8,000 / yr
# - Power utility consumption (~25 MWh @ commercial/industrial rate ~€120/MWh): €3,000 / yr
DME_ANNUAL_FLIGHT_CAL_EUR = 18000.0
DME_ANNUAL_MAINTENANCE_EUR = 16000.0
DME_ANNUAL_COMMS_EUR = 8000.0
DME_ANNUAL_ENERGY_EUR = 3000.0
DME_ANNUAL_COST_SAVINGS_EUR = (
    DME_ANNUAL_FLIGHT_CAL_EUR + DME_ANNUAL_MAINTENANCE_EUR + DME_ANNUAL_COMMS_EUR + DME_ANNUAL_ENERGY_EUR
)  # €45,000 / yr

# ECAC Continental Airspace Bounding Box (Lat: 34N to 68N, Lon: -12W to 35E)
ECAC_LAT_MIN = 35.0
ECAC_LAT_MAX = 65.0
ECAC_LON_MIN = -11.0
ECAC_LON_MAX = 32.0

ECAC_COUNTRIES = {
    "AL", "AM", "AT", "AZ", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "GE", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LT", "LU", "MT", "MD", "MC", "ME",
    "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "GB"
}

ECAC_FIR_REGISTRY: dict[str, str] = {
    "ALL": "🌍 All ECAC Airspace (Network-Wide)",
    "EGTTFIR": "🇬🇧 EGTTFIR - London FIR (UK South & England/Wales)",
    "EGPXFIR": "🇬🇧 EGPXFIR - Scottish FIR (UK North & Scotland)",
    "LFFFFIR": "🇫🇷 LFFFFIR - Paris FIR (France North)",
    "LFBBFIR": "🇫🇷 LFBBFIR - Bordeaux FIR (France Southwest)",
    "LFMMFIR": "🇫🇷 LFMMFIR - Marseille FIR (France South/Mediterranean)",
    "LFEEFIR": "🇫🇷 LFEEFIR - Reims FIR (France East)",
    "LFRRFIR": "🇫🇷 LFRRFIR - Brest FIR (France West / Atlantic)",
    "EDGGFIR": "🇩🇪 EDGGFIR - Langen FIR (Germany Central/West)",
    "EDMMFIR": "🇩🇪 EDMMFIR - Munich FIR (Germany South)",
    "EDWWFIR": "🇩🇪 EDWWFIR - Bremen FIR (Germany North)",
    "EDUUUIR": "🇩🇪 EDUUUIR - Rhein UIR (Germany Upper Airspace)",
    "EHAAFIR": "🇳🇱 EHAAFIR - Amsterdam FIR (Netherlands)",
    "EBBUFIR": "🇧🇪 EBBUFIR - Brussels FIR (Belgium / Luxembourg)",
    "LSASFIR": "🇨🇭 LSASFIR - Switzerland FIR (Zurich / Geneva)",
    "LOVVFIR": "🇦🇹 LOVVFIR - Vienna FIR (Austria)",
    "LECMFIR": "🇪🇸 LECMFIR - Madrid FIR (Spain Central / North)",
    "LECBFIR": "🇪🇸 LECBFIR - Barcelona FIR (Spain East / Balearics)",
    "LPPCFIR": "🇵🇹 LPPCFIR - Lisbon FIR (Portugal Continental)",
    "LIRRFIR": "🇮🇹 LIRRFIR - Rome FIR (Italy Central / South)",
    "LIMMFIR": "🇮🇹 LIMMFIR - Milan FIR (Italy North)",
    "LIBBFIR": "🇮🇹 LIBBFIR - Brindisi FIR (Italy Southeast)",
    "EISNFIR": "🇮🇪 EISNFIR - Shannon FIR (Ireland)",
    "EKDKFIR": "🇩🇰 EKDKFIR - Copenhagen FIR (Denmark)",
    "ESAAFIR": "🇸🇪 ESAAFIR - Sweden FIR (Stockholm / Malmo)",
    "ENORFIR": "🇳🇴 ENORFIR - Norway FIR (Oslo / Stavanger)",
    "EFINFIR": "🇫🇮 EFINFIR - Finland FIR (Helsinki)",
    "EPWWFIR": "🇵🇱 EPWWFIR - Warsaw FIR (Poland)",
    "LKAAFIR": "🇨🇿 LKAAFIR - Prague FIR (Czech Republic)",
    "LZBBFIR": "🇸🇰 LZBBFIR - Bratislava FIR (Slovakia)",
    "LHCCFIR": "🇭🇺 LHCCFIR - Budapest FIR (Hungary)",
    "LJLAFIR": "🇸🇮 LJLAFIR - Ljubljana FIR (Slovenia)",
    "LDZOFIR": "🇭🇷 LDZOFIR - Zagreb FIR (Croatia)",
    "LRBBFIR": "🇷🇴 LRBBFIR - Bucharest FIR (Romania)",
    "LBSRFIR": "🇧🇬 LBSRFIR - Sofia FIR (Bulgaria)",
    "LGGGFIR": "🇬🇷 LGGGFIR - Athens FIR (Hellas / Greece)",
    "LCCCFIR": "🇨🇾 LCCCFIR - Nicosia FIR (Cyprus)",
    "LTBBFIR": "🇹🇷 LTBBFIR - Istanbul FIR (Turkey West)",
    "LTAAFIR": "🇹🇷 LTAAFIR - Ankara FIR (Turkey Central/East)",
}


def get_fir_geometry(
    dataset: FlightDataset, fir_id: str
) -> tuple[Polygon, list[list[float]], tuple[float, float, float, float]] | None:
    """
    Retrieve FIR polygon geometry, boundary points, and bounding box from raw_fir.
    """
    pts = dataset.cursor.execute(
        """
        SELECT Latitude, Longitude
        FROM raw_fir
        WHERE "Airspace ID" = ?
        ORDER BY "Sequence Number" ASC
        """,
        [fir_id],
    ).df()

    if pts.empty or len(pts) < 4:
        return None

    coords = [(float(r["Longitude"]), float(r["Latitude"])) for _, r in pts.iterrows()]
    boundary_lat_lon = [[float(r["Latitude"]), float(r["Longitude"])] for _, r in pts.iterrows()]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    min_lon, min_lat, max_lon, max_lat = poly.bounds
    return poly, boundary_lat_lon, (min_lat, max_lat, min_lon, max_lon)


def list_available_ecac_firs(dataset: FlightDataset) -> list[tuple[str, str]]:
    """
    Returns sorted list of (fir_id, display_label) for selectable ECAC FIRs.
    """
    df = dataset.cursor.execute(
        """
        SELECT "Airspace ID" as fir_id, COUNT(*) as pt_count
        FROM raw_fir
        WHERE Latitude BETWEEN ? AND ? AND Longitude BETWEEN ? AND ?
        GROUP BY "Airspace ID"
        HAVING pt_count >= 8 AND fir_id LIKE '%FIR'
        ORDER BY fir_id ASC
        """,
        [ECAC_LAT_MIN - 2.0, ECAC_LAT_MAX + 2.0, ECAC_LON_MIN - 2.0, ECAC_LON_MAX + 2.0],
    ).df()

    firs = [("ALL", ECAC_FIR_REGISTRY.get("ALL", "🌍 All ECAC Airspace (Network-Wide)"))]
    for _, row in df.iterrows():
        fid = str(row["fir_id"])
        if fid in ECAC_FIR_REGISTRY:
            lbl = ECAC_FIR_REGISTRY[fid]
        else:
            lbl = f"📡 {fid}"
        firs.append((fid, lbl))
    return firs


@dataclass
class ECACStation:
    id: str
    name: str
    type: str  # "RADAR" or "DME" / "VOR-DME"
    latitude: float
    longitude: float
    elevation_ft: float
    country: str
    is_airport_station: bool
    nearest_airport_ident: str | None
    nominal_range_nm: float
    annual_cost_savings_eur: float
    annual_power_savings_mwh: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticGapFiller:
    id: str
    name: str
    latitude: float
    longitude: float
    elevation_ft: float
    target_flight_level: int
    service_range_nm: float
    cells_covered_count: int
    region_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ecac_stations(
    dataset: FlightDataset,
    station_type: Literal["RADAR", "DME"] = "RADAR",
    radars_path: str = "data/radars.json",
    fir_poly: Polygon | None = None,
    fir_bbox: tuple[float, float, float, float] | None = None,
    include_adjacent: bool = False,
) -> tuple[list[ECACStation], list[dict[str, Any]]] | tuple[list[ECACStation], list[dict[str, Any]], list[ECACStation]]:
    """
    Load all ECAC stations (Radars or DMEs) along with ECAC commercial airports
    and high-density hubs (>= 500 movements/day).

    FIR Scoping Rule:
    - For FIR-level analysis, primary stations are strictly those inside the FIR polygon.
    - Neighboring stations outside the polygon are segregated as adjacent stations
      for cross-border infill evaluation.
    """
    lat_min = fir_bbox[0] if fir_bbox else ECAC_LAT_MIN
    lat_max = fir_bbox[1] if fir_bbox else ECAC_LAT_MAX
    lon_min = fir_bbox[2] if fir_bbox else ECAC_LON_MIN
    lon_max = fir_bbox[3] if fir_bbox else ECAC_LON_MAX

    # 1. Fetch ECAC airports (filtered to FIR polygon if specified)
    airports_df = dataset.cursor.execute(
        """
        WITH moves AS (
            SELECT adep AS apt FROM flights
            UNION ALL
            SELECT ades AS apt FROM flights
        ),
        counts AS (
            SELECT apt, ROUND(COUNT(*) / 31.0, 1) as daily_moves
            FROM moves
            WHERE apt IS NOT NULL
            GROUP BY apt
        )
        SELECT a.ident, a.name, a.latitude_deg as latitude, a.longitude_deg as longitude,
               COALESCE(a.elevation_ft, 100.0) as elevation_ft, a.iso_country,
               COALESCE(c.daily_moves, 0.0) as daily_movements
        FROM raw_airports a
        LEFT JOIN counts c ON a.ident = c.apt
        WHERE a.latitude_deg BETWEEN ? AND ?
          AND a.longitude_deg BETWEEN ? AND ?
          AND a.type IN ('large_airport', 'medium_airport')
        ORDER BY daily_movements DESC
        """,
        [lat_min - 0.2, lat_max + 0.2, lon_min - 0.5, lon_max + 0.5],
    ).df()

    airports: list[dict[str, Any]] = []
    for _, row in airports_df.iterrows():
        a_lat = float(row["latitude"])
        a_lon = float(row["longitude"])
        if fir_poly and not fir_poly.contains(Point(a_lon, a_lat)):
            continue

        airports.append({
            "ident": str(row["ident"]),
            "name": str(row["name"]),
            "latitude": a_lat,
            "longitude": a_lon,
            "elevation_ft": float(row["elevation_ft"]),
            "country": str(row.get("iso_country") or "EU"),
            "daily_movements": float(row["daily_movements"]),
            "is_high_density": float(row["daily_movements"]) >= 500.0,
        })

    # 2. Load Stations with Line-of-Sight Buffer
    # Buffer station search box: Radar range is 250 NM (~4.2 deg), DME is 160 NM (~2.8 deg)
    buf_lat = 4.2 if station_type == "RADAR" else 2.8
    buf_lon = 6.0 if station_type == "RADAR" else 4.2
    sta_lat_min = max(-80.0, lat_min - buf_lat)
    sta_lat_max = min(80.0, lat_max + buf_lat)
    sta_lon_min = max(-175.0, lon_min - buf_lon)
    sta_lon_max = min(175.0, lon_max + buf_lon)

    stations: list[ECACStation] = []
    seen_ids = set()

    if station_type == "RADAR":
        if os.path.exists(radars_path):
            with open(radars_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for feat in data.get("features", []):
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [0.0, 0.0])
                lon, lat = float(coords[0]), float(coords[1])
                if not (sta_lat_min <= lat <= sta_lat_max and sta_lon_min <= lon <= sta_lon_max):
                    continue

                r_id = str(props.get("id") or props.get("name") or f"RAD_{len(stations)+1}").strip()
                if r_id in seen_ids:
                    r_id = f"{r_id}_{len(stations)+1}"
                seen_ids.add(r_id)

                # Check proximity to nearest airport (<= 5 NM)
                min_dist = 99999.0
                nearest_apt_ident = None
                for a in airports:
                    if abs(lat - a["latitude"]) > 0.15 or abs(lon - a["longitude"]) > 0.25:
                        continue
                    d = fmath.haversine_distance_nm(lat, lon, a["latitude"], a["longitude"])
                    if d < min_dist:
                        min_dist = d
                        nearest_apt_ident = a["ident"]

                is_apt = min_dist <= 5.0
                stations.append(
                    ECACStation(
                        id=r_id,
                        name=str(props.get("name") or r_id),
                        type="RADAR",
                        latitude=lat,
                        longitude=lon,
                        elevation_ft=150.0,
                        country="EU",
                        is_airport_station=is_apt,
                        nearest_airport_ident=nearest_apt_ident if is_apt else None,
                        nominal_range_nm=250.0,  # Max line of sight 250 NM for radar
                        annual_cost_savings_eur=RADAR_ANNUAL_COST_SAVINGS_EUR,
                        annual_power_savings_mwh=RADAR_ANNUAL_POWER_MWH,
                    )
                )
    else:  # DME
        df_dme = dataset.cursor.execute(
            """
            SELECT ident, name, type, latitude_deg, longitude_deg, elevation_ft, iso_country
            FROM raw_navaids
            WHERE (type LIKE '%DME%' OR type = 'VOR-DME' OR type = 'TACAN')
              AND latitude_deg BETWEEN ? AND ?
              AND longitude_deg BETWEEN ? AND ?
            ORDER BY ident ASC
            """,
            [sta_lat_min, sta_lat_max, sta_lon_min, sta_lon_max],
        ).df()

        for _, row in df_dme.iterrows():
            d_id = str(row["ident"]).strip().upper()
            if not d_id or d_id in seen_ids:
                continue
            seen_ids.add(d_id)
            lat = float(row["latitude_deg"])
            lon = float(row["longitude_deg"])
            elev = float(row["elevation_ft"]) if pd.notna(row["elevation_ft"]) else 100.0

            min_dist = 99999.0
            nearest_apt_ident = None
            for a in airports:
                if abs(lat - a["latitude"]) > 0.15 or abs(lon - a["longitude"]) > 0.25:
                    continue
                d = fmath.haversine_distance_nm(lat, lon, a["latitude"], a["longitude"])
                if d < min_dist:
                    min_dist = d
                    nearest_apt_ident = a["ident"]

            is_apt = min_dist <= 5.0
            stations.append(
                ECACStation(
                    id=d_id,
                    name=str(row.get("name") or d_id),
                    type=str(row.get("type") or "DME"),
                    latitude=lat,
                    longitude=lon,
                    elevation_ft=elev,
                    country=str(row.get("iso_country") or "EU"),
                    is_airport_station=is_apt,
                    nearest_airport_ident=nearest_apt_ident if is_apt else None,
                    nominal_range_nm=160.0,  # Max service range 160 NM for DME
                    annual_cost_savings_eur=DME_ANNUAL_COST_SAVINGS_EUR,
                    annual_power_savings_mwh=DME_ANNUAL_POWER_MWH,
                )
            )

    if fir_poly:
        inside_stations = [s for s in stations if fir_poly.contains(Point(s.longitude, s.latitude))]
        adjacent_stations = [s for s in stations if not fir_poly.contains(Point(s.longitude, s.latitude))]
    else:
        inside_stations = stations
        adjacent_stations = []

    if include_adjacent:
        return inside_stations, airports, adjacent_stations
    return inside_stations, airports


def generate_ecac_grid(
    step_deg: float = 0.35,
    fir_poly: Polygon | None = None,
    fir_bbox: tuple[float, float, float, float] | None = None,
) -> list[tuple[float, float]]:
    """
    Generate 2D spatial grid covering ECAC continental airspace or a specific FIR (~10-15 NM cells).
    """
    if fir_poly and fir_bbox:
        lat_min, lat_max, lon_min, lon_max = fir_bbox
        step = 0.20  # High-resolution step inside single FIR (~12 NM cells)
        lats = np.arange(lat_min, lat_max + 1e-5, step)
        lons = np.arange(lon_min, lon_max + 1e-5, step * 1.5)
        cells = []
        for lat in lats:
            for lon in lons:
                if fir_poly.contains(Point(float(lon), float(lat))):
                    cells.append((round(float(lat), 3), round(float(lon), 3)))
        return cells

    lats = np.arange(ECAC_LAT_MIN, ECAC_LAT_MAX + 1e-5, step_deg)
    lons = np.arange(ECAC_LON_MIN, ECAC_LON_MAX + 1e-5, step_deg * 1.5)
    cells = []
    for lat in lats:
        for lon in lons:
            cells.append((round(float(lat), 3), round(float(lon), 3)))
    return cells


def _compute_coverage_matrices(
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    airports: list[dict[str, Any]],
    flight_levels: tuple[int, ...] = (10, 50, 100, 200, 300),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Vectorized computation of boolean visibility matrix:
    cell_matrix[FL]: shape (N_cells, N_stations)
    apt_matrix[FL]:  shape (N_airports, N_stations)
    """
    n_cells = len(grid_cells)
    n_sta = len(stations)
    n_apt = len(airports)

    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])

    s_lats = np.array([s.latitude for s in stations])
    s_lons = np.array([s.longitude for s in stations])
    s_elev = np.array([s.elevation_ft for s in stations])
    s_nom = np.array([s.nominal_range_nm for s in stations])

    a_lats = np.array([a["latitude"] for a in airports])
    a_lons = np.array([a["longitude"] for a in airports])

    # Distance matrices (approx NM)
    # Cell to station
    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)

    # Airport to station
    dlat_a = (a_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_a = np.radians((a_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_a = (a_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_a)
    dist_apt = np.sqrt(dlat_a**2 + dlon_a**2)

    cell_matrices: dict[int, np.ndarray] = {}
    apt_matrices: dict[int, np.ndarray] = {}

    for fl in flight_levels:
        h_ac = fl * 100.0
        # Line of sight for each station
        rlos = 1.23 * (np.sqrt(h_ac) + np.sqrt(np.maximum(0.0, s_elev)))
        max_r = np.minimum(s_nom, rlos)  # shape (N_stations,)

        cell_matrices[fl] = dist_cell <= max_r[None, :]
        apt_matrices[fl] = dist_apt <= max_r[None, :]

    return cell_matrices, apt_matrices


def _compute_dme_navigation_standards_views(
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    airports: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Computes minimum equipment sets for Navigation Performance Standards (ICAO Doc 9613 / FAA AC 90-100A):
    - RNAV_5: En-Route B-RNAV (±5 NM). Min DMEs to provide valid 30°-150° pair at FL200+, >=1 at airports FL10.
    - RNAV_2: En-Route Continental (±2 NM). Valid pair within <=130 NM, busy hubs dual FL10.
    - RNAV_1: Terminal P-RNAV SIDs/STARs (±1 NM). N-1 redundancy (>=3 DMEs with valid pair <=60 NM), dual FL10 at airports.
    - RNP_1: Reversionary Multi-DME/IRU (±1 NM). High-integrity >=3-4 DMEs <=40 NM, dual FL10 at airports.
    """
    n_cells = len(grid_cells)
    n_sta = len(stations)
    sta_ids = [s.id for s in stations]

    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])
    s_lats = np.array([s.latitude for s in stations])
    s_lons = np.array([s.longitude for s in stations])

    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)
    bearings_cell = (np.degrees(np.arctan2(dlon_c, dlat_c)) + 360.0) % 360.0

    cell_mat, apt_mat = _compute_coverage_matrices(stations, grid_cells, airports, flight_levels=(10, 50, 100, 200))

    coverable_a10 = apt_mat[10].sum(axis=1) > 0
    coverable_busy_a10 = apt_mat[10].sum(axis=1) >= 2
    busy_apt_indices = [i for i, a in enumerate(airports) if a["is_high_density"]]

    # 1. RNAV 5: En-route Upper Airspace (±5 NM)
    # Priority 1: Cover all coverable airports at FL10 with >= 1 DME
    rnav5_set: set[int] = set()
    apt_mask = np.zeros(len(airports), dtype=bool)
    while (~apt_mask & coverable_a10).any():
        unc = ~apt_mask & coverable_a10
        gains = np.array([(apt_mat[10][:, s] & unc).sum() if s not in rnav5_set else -1 for s in range(n_sta)])
        best_s = int(np.argmax(gains))
        if gains[best_s] <= 0:
            break
        rnav5_set.add(best_s)
        apt_mask |= apt_mat[10][:, best_s]

    vis_200 = cell_mat[200]
    # Target cells that can form a valid 30°-150° pair at FL200
    rnav5_target = np.zeros(n_cells, dtype=bool)
    for i in range(n_cells):
        v = np.where(vis_200[i])[0]
        if len(v) >= 2:
            b = bearings_cell[i, v]
            diff = np.abs(b[:, None] - b[None, :])
            diff = np.minimum(diff, 360.0 - diff)
            if ((diff >= 30.0) & (diff <= 150.0)).any():
                rnav5_target[i] = True

    # Seed station with highest coverage if set is empty
    if not rnav5_set and n_sta > 0:
        rnav5_set.add(int(np.argmax(vis_200.sum(axis=0))))

    cur_paired = np.zeros(n_cells, dtype=bool)
    if rnav5_set:
        act = list(rnav5_set)
        act_vis = vis_200[:, act]
        act_brg = bearings_cell[:, act]
        diff = np.abs(act_brg[:, :, None] - act_brg[:, None, :])
        diff = np.minimum(diff, 360.0 - diff)
        cur_paired = ((act_vis[:, :, None] & act_vis[:, None, :]) & (diff >= 30.0) & (diff <= 150.0)).any(axis=(1, 2))

    while (~cur_paired & rnav5_target).any():
        rem_target = ~cur_paired & rnav5_target
        act = list(rnav5_set)
        act_vis = vis_200[:, act]
        act_brg = bearings_cell[:, act]
        best_s = -1
        best_gain = 0
        for s in range(n_sta):
            if s in rnav5_set:
                continue
            s_vis = vis_200[:, s]
            s_brg = bearings_cell[:, s, None]
            diff = np.abs(act_brg - s_brg)
            diff = np.minimum(diff, 360.0 - diff)
            new_pairs = (act_vis & s_vis[:, None]) & (diff >= 30.0) & (diff <= 150.0)
            gain = int((new_pairs.any(axis=1) & rem_target).sum())
            if gain > best_gain:
                best_gain = gain
                best_s = s
        if best_gain == 0 or best_s == -1:
            unc_counts = vis_200[rem_target, :].sum(axis=0)
            unc_counts[list(rnav5_set)] = -1
            best_s = int(np.argmax(unc_counts))
            if unc_counts[best_s] <= 0:
                break
        rnav5_set.add(best_s)
        act = list(rnav5_set)
        act_vis = vis_200[:, act]
        act_brg = bearings_cell[:, act]
        diff = np.abs(act_brg[:, :, None] - act_brg[:, None, :])
        diff = np.minimum(diff, 360.0 - diff)
        cur_paired = ((act_vis[:, :, None] & act_vis[:, None, :]) & (diff >= 30.0) & (diff <= 150.0)).any(axis=(1, 2))

    # 2. RNAV 2: Continental / Transition (±2 NM, range <= 130 NM, dual at busy hubs)
    rnav2_set: set[int] = set(rnav5_set)
    for i in busy_apt_indices:
        if not coverable_busy_a10[i]:
            continue
        while apt_mat[10][i, list(rnav2_set)].sum() < 2:
            cand = [s for s in range(n_sta) if s not in rnav2_set and apt_mat[10][i, s]]
            if not cand:
                break
            rnav2_set.add(cand[0])

    # 3. RNAV 1: Terminal SIDs/STARs (±1 NM, N-1 Redundancy >=3 DMEs <=60 NM)
    rnav1_set: set[int] = set(rnav2_set)
    for i in range(len(airports)):
        if not coverable_busy_a10[i]:
            continue
        while apt_mat[10][i, list(rnav1_set)].sum() < 2:
            cand = [s for s in range(n_sta) if s not in rnav1_set and apt_mat[10][i, s]]
            if not cand:
                break
            rnav1_set.add(cand[0])

    vis_100_term = cell_mat[100] & (dist_cell <= 60.0)
    term_counts = vis_100_term[:, list(rnav1_set)].sum(axis=1) if rnav1_set else np.zeros(n_cells)
    term_target = np.minimum(3, vis_100_term.sum(axis=1))
    while (term_counts < term_target).any():
        needed = term_counts < term_target
        gains = np.array([(vis_100_term[:, s] & needed).sum() if s not in rnav1_set else -1 for s in range(n_sta)])
        best_s = int(np.argmax(gains))
        if gains[best_s] <= 0:
            break
        rnav1_set.add(best_s)
        term_counts += vis_100_term[:, best_s]

    # 4. RNP 1: Reversionary Terminal Multi-DME/IRU (High integrity >=4 DMEs <=40 NM)
    rnp1_set: set[int] = set(rnav1_set)
    vis_100_rnp = cell_mat[100] & (dist_cell <= 40.0)
    rnp_counts = vis_100_rnp[:, list(rnp1_set)].sum(axis=1) if rnp1_set else np.zeros(n_cells)
    rnp_target = np.minimum(4, vis_100_rnp.sum(axis=1))
    while (rnp_counts < rnp_target).any():
        needed = rnp_counts < rnp_target
        gains = np.array([(vis_100_rnp[:, s] & needed).sum() if s not in rnp1_set else -1 for s in range(n_sta)])
        best_s = int(np.argmax(gains))
        if gains[best_s] <= 0:
            break
        rnp1_set.add(best_s)
        rnp_counts += vis_100_rnp[:, best_s]

    return {
        "RNAV_5": {sta_ids[idx] for idx in rnav5_set},
        "RNAV_2": {sta_ids[idx] for idx in rnav2_set},
        "RNAV_1": {sta_ids[idx] for idx in rnav1_set},
        "RNP_1": {sta_ids[idx] for idx in rnp1_set},
        "VIEW_1": {sta_ids[idx] for idx in rnav5_set},
        "VIEW_2": {sta_ids[idx] for idx in rnav2_set},
        "VIEW_3": {sta_ids[idx] for idx in rnav1_set},
    }


def compute_static_rationalization_views(
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    airports: list[dict[str, Any]],
    station_type: Literal["RADAR", "DME"] = "RADAR",
) -> dict[str, set[str]]:
    """
    Computes static rationalization views using greedy set-cover:
    - For DME (Navigation): Solves for RNAV_5, RNAV_2, RNAV_1, and RNP_1 performance standards.
    - For RADAR (Surveillance): Solves for View 1 (Minimum Baseline), View 2 (Dual En-Route), View 3 (Low-Level Infill & Busy Hubs).
    """
    if station_type == "DME":
        return _compute_dme_navigation_standards_views(stations, grid_cells, airports)

    cell_mat, apt_mat = _compute_coverage_matrices(stations, grid_cells, airports, flight_levels=(10, 50, 200))
    n_sta = len(stations)
    sta_ids = [s.id for s in stations]

    # Coverability masks (cells that have at least 1 station when ALL stations are active)
    coverable_c_fl200 = cell_mat[200].sum(axis=1) > 0
    coverable_c_fl50 = cell_mat[50].sum(axis=1) > 0
    coverable_a_fl10 = apt_mat[10].sum(axis=1) > 0

    busy_apt_indices = [i for i, a in enumerate(airports) if a["is_high_density"]]
    coverable_busy_fl10 = np.zeros(len(airports), dtype=bool)
    for i in busy_apt_indices:
        if apt_mat[10][i].sum() >= 2:
            coverable_busy_fl10[i] = True

    # ---------------- VIEW 1: Minimum Baseline ----------------
    view_1_set: set[int] = set()

    # Priority 1: Cover all coverable airports at FL10
    apt_cov_mask = np.zeros(len(airports), dtype=bool)
    while (~apt_cov_mask & coverable_a_fl10).any():
        uncovered = ~apt_cov_mask & coverable_a_fl10
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_1_set:
                continue
            gain = (apt_mat[10][:, s] & uncovered).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_1_set.add(best_sta)
        apt_cov_mask |= apt_mat[10][:, best_sta]

    # Priority 2: Cover all coverable FL200 cells with >= 1 station
    cell_cov_count = cell_mat[200][:, list(view_1_set)].sum(axis=1) if view_1_set else np.zeros(len(grid_cells))
    while ((cell_cov_count < 1) & coverable_c_fl200).any():
        uncovered = (cell_cov_count < 1) & coverable_c_fl200
        if not uncovered.any():
            break
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_1_set:
                continue
            gain = (cell_mat[200][:, s] & uncovered).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_1_set.add(best_sta)
        cell_cov_count += cell_mat[200][:, best_sta]

    # ---------------- VIEW 2: Dual En-Route Redundancy ----------------
    view_2_set: set[int] = set(view_1_set)
    max_fl200_potential = cell_mat[200].sum(axis=1)
    target_fl200_cov = np.minimum(2, max_fl200_potential)

    cell_cov_count_v2 = cell_mat[200][:, list(view_2_set)].sum(axis=1) if view_2_set else np.zeros(len(grid_cells))
    while (cell_cov_count_v2 < target_fl200_cov).any():
        needed = cell_cov_count_v2 < target_fl200_cov
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_2_set:
                continue
            gain = (cell_mat[200][:, s] & needed).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_2_set.add(best_sta)
        cell_cov_count_v2 += cell_mat[200][:, best_sta]

    # ---------------- VIEW 3: Low-Level Infill & Busy Hub Dual ----------------
    view_3_set: set[int] = set(view_2_set)

    # 1. Ensure dual FL10 coverage at busy hubs
    for i in busy_apt_indices:
        if not coverable_busy_fl10[i]:
            continue
        cur_apt_cov = apt_mat[10][i, list(view_3_set)].sum() if view_3_set else 0
        while cur_apt_cov < 2:
            best_s = -1
            for s in range(n_sta):
                if s not in view_3_set and apt_mat[10][i, s]:
                    best_s = s
                    break
            if best_s == -1:
                break
            view_3_set.add(best_s)
            cur_apt_cov += 1

    # 2. Infill FL50 cells to >= 1 station
    cell_cov_fl50 = cell_mat[50][:, list(view_3_set)].sum(axis=1) if view_3_set else np.zeros(len(grid_cells))
    while ((cell_cov_fl50 < 1) & coverable_c_fl50).any():
        uncovered = (cell_cov_fl50 < 1) & coverable_c_fl50
        if not uncovered.any():
            break
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_3_set:
                continue
            gain = (cell_mat[50][:, s] & uncovered).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_3_set.add(best_sta)
        cell_cov_fl50 += cell_mat[50][:, best_sta]

    return {
        "VIEW_1": {sta_ids[idx] for idx in view_1_set},
        "VIEW_2": {sta_ids[idx] for idx in view_2_set},
        "VIEW_3": {sta_ids[idx] for idx in view_3_set},
        "SURV_1": {sta_ids[idx] for idx in view_1_set},
        "SURV_2": {sta_ids[idx] for idx in view_2_set},
        "SURV_3": {sta_ids[idx] for idx in view_3_set},
    }


def compute_synthetic_gap_fillers(
    uncoverable_cells: list[tuple[float, float]],
    station_type: Literal["RADAR", "DME"],
    target_fl: int = 200,
    candidate_airports: list[dict[str, Any]] | None = None,
) -> list[SyntheticGapFiller]:
    """
    Greedy Facility Location solver with strict Terrestrial Land Constraint:
    Identifies the minimum number of ideal synthetic station locations on land
    required to cover gap cells.

    Rules:
    - Proposed infill sites MUST be on land (is_point_on_land is True).
    - Water gap cells are covered from coastal/island terrestrial sites where feasible.
    - If a cell cannot be covered by any land site (e.g. deep oceanic airspace),
      do NOT propose an impossible offshore station.
    """
    if not uncoverable_cells:
        return []

    service_r = 160.0 if station_type == "DME" else 220.0
    u_lats = np.array([c[0] for c in uncoverable_cells])
    u_lons = np.array([c[1] for c in uncoverable_cells])
    is_land_u = is_points_on_land(u_lats, u_lons)

    # Candidate land coordinates
    cand_lats: list[float] = list(u_lats[is_land_u])
    cand_lons: list[float] = list(u_lons[is_land_u])

    # If candidate airports are provided, add land airports as potential station sites
    # to provide coastal/island coverage for water gap cells
    if candidate_airports:
        apt_lats = np.array([a["latitude"] for a in candidate_airports])
        apt_lons = np.array([a["longitude"] for a in candidate_airports])
        apt_land = is_points_on_land(apt_lats, apt_lons)
        land_apt_lats = apt_lats[apt_land]
        land_apt_lons = apt_lons[apt_land]

        if len(land_apt_lats) > 0:
            water_indices = np.where(~is_land_u)[0]
            for wi in water_indices:
                w_lat, w_lon = u_lats[wi], u_lons[wi]
                dlat = (land_apt_lats - w_lat) * 60.0
                dlon = (land_apt_lons - w_lon) * 60.0 * np.cos(np.radians(w_lat))
                dists = np.sqrt(dlat**2 + dlon**2)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] <= service_r:
                    cand_lats.append(float(land_apt_lats[min_idx]))
                    cand_lons.append(float(land_apt_lons[min_idx]))

    # Deduplicate and ensure 100% strict land validation
    all_cand_pts = {(round(float(la), 3), round(float(lo), 3)) for la, lo in zip(cand_lats, cand_lons)}
    cand_pts = [p for p in all_cand_pts if is_point_on_land(p[0], p[1])]

    if not cand_pts:
        return []

    c_lats = np.array([p[0] for p in cand_pts])
    c_lons = np.array([p[1] for p in cand_pts])

    rem_indices = set(range(len(uncoverable_cells)))
    synthetics: list[SyntheticGapFiller] = []
    gap_count = 0

    while rem_indices:
        rem_list = list(rem_indices)
        rem_lats = u_lats[rem_list]
        rem_lons = u_lons[rem_list]

        # Vectorized distance from all candidate land points to remaining cells
        dlat = (rem_lats[None, :] - c_lats[:, None]) * 60.0
        mid_lat = np.radians((rem_lats[None, :] + c_lats[:, None]) / 2.0)
        dlon = (rem_lons[None, :] - c_lons[:, None]) * 60.0 * np.cos(mid_lat)
        dists = np.sqrt(dlat**2 + dlon**2)
        in_range = dists <= service_r
        gains = in_range.sum(axis=1)

        best_cand_idx = int(np.argmax(gains))
        best_gain = int(gains[best_cand_idx])
        if best_gain == 0:
            # No remaining uncoverable cells can be covered from any land location!
            # Do NOT propose any synthetic station for them.
            break

        gap_count += 1
        c_lat = float(c_lats[best_cand_idx])
        c_lon = float(c_lons[best_cand_idx])
        covered_indices = {rem_list[k] for k in np.where(in_range[best_cand_idx])[0]}
        rem_indices -= covered_indices

        # Geographic region descriptor
        if c_lat < 40.0 and c_lon > 20.0:
            reg = "Eastern Mediterranean / Aegean Terrestrial Infill"
        elif c_lat < 42.0 and c_lon < 0.0:
            reg = "Iberian Atlantic Terrestrial Infill"
        elif c_lat > 60.0:
            reg = "Nordic / Coastal Terrestrial Infill"
        elif c_lon > 25.0:
            reg = "Eastern European Terrestrial Infill"
        else:
            reg = "Central European Terrestrial Infill"

        synthetics.append(
            SyntheticGapFiller(
                id=f"SYN-{station_type}-{gap_count:02d}",
                name=f"Proposed {station_type} Infill #{gap_count} (Land)",
                latitude=round(c_lat, 3),
                longitude=round(c_lon, 3),
                elevation_ft=1000.0,
                target_flight_level=target_fl,
                service_range_nm=service_r,
                cells_covered_count=len(covered_indices),
                region_note=reg,
            )
        )

    return synthetics


@disk_cache(namespace="ecac_rationalization")
def get_ecac_rationalization_report(
    dataset: FlightDataset,
    station_type: Literal["RADAR", "DME"] = "RADAR",
    active_view: Literal["VIEW_1", "VIEW_2", "VIEW_3"] = "VIEW_1",
    target_fir: str = "ALL",
    custom_inactive_ids: tuple[str, ...] | None = None,
    custom_active_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Main evaluation pipeline with disk caching.
    Computes active/deactivated stations, multi-altitude coverage maps (FL50, FL100, FL200, FL300),
    airport ground integrity, and synthetic gap filler placements.
    Supports either network-wide ECAC evaluation (target_fir='ALL') or any single FIR domain.
    """
    fir_poly = None
    fir_boundary_coords = None
    fir_bbox = None
    fir_name = "All ECAC Airspace (Network-Wide)"
    fir_center = [50.0, 10.0]

    if target_fir and target_fir != "ALL":
        fir_geom = get_fir_geometry(dataset, target_fir)
        if fir_geom:
            fir_poly, fir_boundary_coords, fir_bbox = fir_geom
            fir_name = ECAC_FIR_REGISTRY.get(target_fir, target_fir)
            fir_center = [round((fir_bbox[0] + fir_bbox[1]) / 2.0, 3), round((fir_bbox[2] + fir_bbox[3]) / 2.0, 3)]

    stations, airports, adjacent_stations = load_ecac_stations(
        dataset,
        station_type=station_type,
        fir_poly=fir_poly,
        fir_bbox=fir_bbox,
        include_adjacent=True,
    )
    step_grid = 0.17 if fir_poly else 0.35
    grid_cells = generate_ecac_grid(
        step_deg=step_grid,
        fir_poly=fir_poly,
        fir_bbox=fir_bbox,
    )

    # 1. Compute or retrieve static baseline views
    static_views = compute_static_rationalization_views(
        stations,
        grid_cells,
        airports,
        station_type=station_type,
    )
    base_active_set = set(static_views.get(active_view, set()))

    # 2. Apply user augmentations
    if custom_inactive_ids:
        base_active_set -= set(custom_inactive_ids)
    if custom_active_ids:
        base_active_set |= set(custom_active_ids)

    # 3. Vectorized coverage evaluation across FL50, FL100, FL200, FL300
    levels = (10, 50, 100, 200, 300)
    cell_matrices, apt_matrices = _compute_coverage_matrices(stations, grid_cells, airports, flight_levels=levels)

    sta_id_to_idx = {s.id: i for i, s in enumerate(stations)}
    active_indices = [sta_id_to_idx[sid] for sid in base_active_set if sid in sta_id_to_idx]

    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])
    s_lats = np.array([s.latitude for s in stations])
    s_lons = np.array([s.longitude for s in stations])

    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)
    bearings_cell = (np.degrees(np.arctan2(dlon_c, dlat_c)) + 360.0) % 360.0

    # Full network potential (for identifying uncoverable cells)
    uncoverable_cells_by_fl: dict[int, list[tuple[float, float]]] = {}
    for fl in (50, 100, 200, 300):
        max_potential = cell_matrices[fl].sum(axis=1)
        unc_mask = max_potential == 0
        uncoverable_cells_by_fl[fl] = [grid_cells[idx] for idx in np.where(unc_mask)[0]]

    # Active coverage evaluations per FL
    coverage_by_fl: dict[str, Any] = {}
    for fl in (50, 100, 200, 300):
        max_cov = cell_matrices[fl].sum(axis=1)
        total_cells = len(grid_cells)

        # Assemble compact cell details with Navigation / Surveillance performance standards
        cell_details = []
        for i, (lat, lon) in enumerate(grid_cells):
            m_pot = int(max_cov[i])

            if station_type == "DME":
                act_s = [s for s in active_indices if cell_matrices[fl][i, s]]
                cnt = len(act_s)
                has_valid_pair = False
                best_geom = 0.0
                gdop = 999.0
                d_max_pair = 0.0
                geom_tip = ""

                if cnt >= 2:
                    b = bearings_cell[i, act_s]
                    diffs = np.abs(b[:, None] - b[None, :])
                    delta = np.minimum(diffs, 360.0 - diffs)
                    valid = (delta >= 30.0) & (delta <= 150.0)
                    if valid.any():
                        has_valid_pair = True
                        score = np.abs(delta - 90.0) + (1.0 - valid) * 9999.0
                        idx_pair = np.unravel_index(np.argmin(score), delta.shape)
                        best_geom = float(delta[idx_pair])
                        gdop = round(math.sqrt(2) / max(1e-4, math.sin(math.radians(best_geom))), 2)
                        d_max_pair = float(max(dist_cell[i, act_s[idx_pair[0]]], dist_cell[i, act_s[idx_pair[1]]]))
                        geom_tip = f"Cut Angle: {best_geom:.1f}° | GDOP: {gdop:.2f} | Slant: {d_max_pair:.0f} NM"
                    else:
                        geom_tip = f"Invalid Geometry ({cnt} DMEs visible, all cut angles <30° or >150°)"
                else:
                    geom_tip = f"{cnt} DME(s) visible"

                if m_pot == 0:
                    st = "UNCOVERABLE"
                elif active_view in ("RNAV_5", "VIEW_1"):
                    st = "MET" if has_valid_pair else ("AT_RISK" if cnt >= 1 else "CANNOT_BE_MET")
                elif active_view in ("RNAV_2", "VIEW_2"):
                    if has_valid_pair and d_max_pair <= 130.0:
                        st = "MET"
                    elif has_valid_pair or cnt >= 1:
                        st = "AT_RISK"
                    else:
                        st = "CANNOT_BE_MET"
                elif active_view in ("RNAV_1", "VIEW_3"):
                    if cnt >= 3 and has_valid_pair and d_max_pair <= 60.0:
                        st = "MET"  # N-1 redundant
                    elif has_valid_pair or cnt >= 1:
                        st = "AT_RISK"  # Critical DME single point of failure
                    else:
                        st = "CANNOT_BE_MET"
                elif active_view == "RNP_1":
                    if cnt >= 4 and has_valid_pair and d_max_pair <= 40.0:
                        st = "MET"  # Autonomous multi-DME integrity
                    elif has_valid_pair or cnt >= 1:
                        st = "AT_RISK"  # Reversionary only
                    else:
                        st = "CANNOT_BE_MET"
                else:
                    st = "MET" if has_valid_pair else ("AT_RISK" if cnt >= 1 else "CANNOT_BE_MET")
            else:  # RADAR
                act_s = [s for s in active_indices if cell_matrices[fl][i, s]]
                cnt = len(act_s)
                geom_tip = f"{cnt} Radar Head(s) visible"
                if m_pot == 0:
                    st = "UNCOVERABLE"
                elif cnt >= 2:
                    st = "MET"
                elif cnt == 1:
                    st = "MET" if active_view in ("VIEW_1", "SURV_1") else "AT_RISK"
                else:
                    st = "CANNOT_BE_MET"

            cell_details.append({
                "lat": lat,
                "lon": lon,
                "count": cnt,
                "status": st,
                "cell_size_nm": 10.0,
                "geom_tip": geom_tip,
                "has_valid_pair": has_valid_pair if station_type == "DME" else (cnt >= 2),
                "best_geom_deg": best_geom if station_type == "DME" else 0.0,
                "gdop": gdop if station_type == "DME" and gdop < 900.0 else None,
            })

        met_cov = int(sum(1 for c in cell_details if c["status"] == "MET"))
        at_risk_cov = int(sum(1 for c in cell_details if c["status"] == "AT_RISK"))
        outage_cov = int(sum(1 for c in cell_details if c["status"] == "CANNOT_BE_MET"))
        uncov_cov = int(sum(1 for c in cell_details if c["status"] == "UNCOVERABLE"))
        coverable_total = met_cov + at_risk_cov + outage_cov

        met_pct = round((met_cov / max(1, coverable_total)) * 100.0, 1)
        at_risk_pct = round((at_risk_cov / max(1, coverable_total)) * 100.0, 1)
        outage_pct = round((outage_cov / max(1, coverable_total)) * 100.0, 1)

        coverage_by_fl[f"fl{fl}"] = {
            "flight_level": fl,
            "altitude_ft": fl * 100,
            "total_cells": total_cells,
            "coverable_cells": coverable_total,
            "uncoverable_cells": uncov_cov,
            "single_coverage_cells": met_cov + at_risk_cov,
            "single_coverage_pct": round(((met_cov + at_risk_cov) / max(1, coverable_total)) * 100.0, 1),
            "dual_coverage_cells": met_cov,
            "dual_coverage_pct": met_pct,
            "met_cells_count": met_cov,
            "met_pct": met_pct,
            "at_risk_cells_count": at_risk_cov,
            "at_risk_pct": at_risk_pct,
            "outage_cells": outage_cov,
            "outage_pct": outage_pct,
            "cells": cell_details,
        }

    # 4. Airport Coverage at FL10
    apt_cov_fl10 = apt_matrices[10][:, active_indices].sum(axis=1) if active_indices else np.zeros(len(airports))
    apt_reports = []
    busy_dual_met = 0
    busy_total = 0

    for i, apt in enumerate(airports):
        c_10 = int(apt_cov_fl10[i])
        is_busy = apt["is_high_density"]
        if is_busy:
            busy_total += 1
            if c_10 >= 2:
                busy_dual_met += 1

        target_met = (c_10 >= 2) if is_busy else (c_10 >= 1)
        apt_reports.append({
            "ident": apt["ident"],
            "name": apt["name"],
            "latitude": apt["latitude"],
            "longitude": apt["longitude"],
            "daily_movements": apt["daily_movements"],
            "is_high_density": is_busy,
            "coverage_count_fl10": c_10,
            "target_met": target_met,
        })

    # 5. Station Inventory with Active / Inactive states
    station_list = []
    active_count = 0
    deactivated_count = 0
    tot_cost_saved = 0.0
    tot_power_saved = 0.0

    for s in stations:
        is_act = s.id in base_active_set
        if is_act:
            active_count += 1
        else:
            deactivated_count += 1
            tot_cost_saved += s.annual_cost_savings_eur
            tot_power_saved += s.annual_power_savings_mwh

        station_list.append({
            "id": s.id,
            "name": s.name,
            "type": s.type,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "elevation_ft": s.elevation_ft,
            "country": s.country,
            "is_airport_station": s.is_airport_station,
            "nearest_airport": s.nearest_airport_ident,
            "is_active": is_act,
            "annual_savings_eur": s.annual_cost_savings_eur,
            "annual_power_mwh": s.annual_power_savings_mwh,
        })

    # 6. Hierarchical Gap Infill Architecture
    # User Mandate:
    # 1. For FIR level views, only consider internal stations in primary inventory.
    # 2. Where an unfillable gap exists, suggest using an adjacent FIR station to infill first.
    # 3. If not possible, suggest a synthetic station, but only if it can be sited on land.
    uncovered_fl200 = uncoverable_cells_by_fl.get(200, [])
    suggested_adjacent_stations = []
    remaining_gaps = list(uncovered_fl200)

    if remaining_gaps and adjacent_stations:
        adj_mat, _ = _compute_coverage_matrices(adjacent_stations, remaining_gaps, airports, flight_levels=(200,))
        cov_from_adj = adj_mat[200]  # shape: (N_gaps, N_adj_stations)

        rem_indices = set(range(len(remaining_gaps)))
        while rem_indices:
            best_adj_idx = -1
            best_gain = 0
            best_covered_in_step = set()

            for s_idx in range(len(adjacent_stations)):
                visible = {i for i in rem_indices if cov_from_adj[i, s_idx]}
                if len(visible) > best_gain:
                    best_gain = len(visible)
                    best_adj_idx = s_idx
                    best_covered_in_step = visible

            if best_gain == 0 or best_adj_idx == -1:
                break

            adj_s = adjacent_stations[best_adj_idx]
            suggested_adjacent_stations.append({
                "id": adj_s.id,
                "name": adj_s.name,
                "type": adj_s.type,
                "latitude": round(adj_s.latitude, 3),
                "longitude": round(adj_s.longitude, 3),
                "elevation_ft": round(adj_s.elevation_ft, 0),
                "country": adj_s.country,
                "nominal_range_nm": adj_s.nominal_range_nm,
                "cells_covered_count": len(best_covered_in_step),
                "strategy": "ADJACENT_FIR_CROSS_BORDER",
                "rationale": f"Existing station in {adj_s.country} ({adj_s.name}) provides cross-border coverage bridging {len(best_covered_in_step)} gap cells at €0 capital cost.",
            })
            rem_indices -= best_covered_in_step

        remaining_gaps = [remaining_gaps[i] for i in rem_indices]

    # Step 2: For any gaps that cannot be covered by adjacent stations,
    # compute synthetic infill stations strictly sited on land!
    gap_fillers = compute_synthetic_gap_fillers(
        remaining_gaps,
        station_type=station_type,
        target_fl=200,
        candidate_airports=airports,
    )

    return {
        "station_type": station_type,
        "active_view": active_view,
        "target_fir": target_fir,
        "fir_name": fir_name,
        "fir_boundary_coords": fir_boundary_coords,
        "fir_center": fir_center,
        "fir_bounds": list(fir_bbox) if fir_bbox else [ECAC_LAT_MIN, ECAC_LAT_MAX, ECAC_LON_MIN, ECAC_LON_MAX],
        "total_stations_count": len(stations),
        "active_stations_count": active_count,
        "deactivated_stations_count": deactivated_count,
        "annual_cost_savings_eur": tot_cost_saved,
        "annual_power_savings_mwh": tot_power_saved,
        "high_density_airports_count": busy_total,
        "high_density_dual_target_met_count": busy_dual_met,
        "high_density_dual_pct": round((busy_dual_met / max(1, busy_total)) * 100.0, 1),
        "stations": station_list,
        "airports": apt_reports,
        "coverage_by_fl": coverage_by_fl,
        "adjacent_fir_infill_suggestions": suggested_adjacent_stations,
        "synthetic_gap_fillers": [g.to_dict() for g in gap_fillers],
        "uncoverable_cells_count_fl200": len(uncoverable_cells_by_fl[200]),
        "remaining_uncovered_after_adjacent": len(remaining_gaps),
        "methodology": {
            "radar": {
                "annual_cost_savings_eur": RADAR_ANNUAL_COST_SAVINGS_EUR,
                "maintenance_eur": RADAR_ANNUAL_MAINTENANCE_EUR,
                "flight_calibration_eur": RADAR_ANNUAL_FLIGHT_CAL_EUR,
                "comms_site_eur": RADAR_ANNUAL_COMMS_EUR,
                "energy_eur": RADAR_ANNUAL_ENERGY_EUR,
                "power_draw_kw": RADAR_POWER_DRAW_KW,
                "annual_power_mwh": RADAR_ANNUAL_POWER_MWH,
            },
            "dme": {
                "annual_cost_savings_eur": DME_ANNUAL_COST_SAVINGS_EUR,
                "flight_calibration_eur": DME_ANNUAL_FLIGHT_CAL_EUR,
                "maintenance_eur": DME_ANNUAL_MAINTENANCE_EUR,
                "comms_monitoring_eur": DME_ANNUAL_COMMS_EUR,
                "energy_eur": DME_ANNUAL_ENERGY_EUR,
                "power_draw_kw": DME_POWER_DRAW_KW,
                "annual_power_mwh": DME_ANNUAL_POWER_MWH,
            },
            "sources": [
                "Eurocontrol Specification for the Rationalisation of Terrestrial Surveillance Infrastructure",
                "Eurocontrol Navigation Infrastructure Strategy (Conventional Navaids Rationalisation)",
                "ICAO Doc 9750 (Global Air Navigation Plan - CNS Economic Benchmarks)",
            ],
        },
    }
