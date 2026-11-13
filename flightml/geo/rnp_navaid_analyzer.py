"""
4D Trajectory Active Navaid Selection & Anticipated RNP/RNAV Capability Analyzer.

Evaluates radio navigation infrastructure (`nav_aids_xplane`: DME, VOR, ILS/LOC, NDB) along a
flight's 4D trajectory (`latitude`, `longitude`, `flight_level`, `time_over`) using VHF/UHF
Radio Line-of-Sight (RLOS) horizon physics:
    D_RLOS (nm) = 1.23 * (sqrt(h_aircraft_ft) + sqrt(h_navaid_ft))

Determines:
1. Likely used (auto-tuned / corridor-active) ground navaids along the trajectory, classified by
   type (`DME`, `VOR`, `ILS_LOC`, `NDB`) for rendering as color-coded dots on the trajectory map.
2. Point-by-point anticipated RNP / RNAV navigation capability (in nautical miles: 0.1, 0.3, 1.0,
   2.0, 5.0, 10.0 nm) along the vertical profile, together with the explicit technical reason why
   that RNP/RNAV specification is available (DME/DME triangulation count, VOR/DME backup, ILS
   approach lock, or Oceanic/Remote beyond-RLOS GNSS/IRS navigation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.geodesy import haversine_nm


NAVAID_COLOR_MAP: dict[str, str] = {
    "DME": "#00a8ff",       # Cyan / Dodger Blue — DME slant-range station
    "VOR": "#00b894",       # Emerald Green — VOR / VOR-DME omnidirectional station
    "ILS_LOC": "#8e44ad",   # Purple — ILS Localizer / Precision Approach
    "LOC": "#8e44ad",       # Purple — Localizer
    "DME_ILS": "#9b59b6",   # Violet — Approach DME paired with ILS
    "NDB": "#f39c12",       # Amber Gold — Non-Directional Beacon
}


@dataclass
class ActiveRouteNavaid:
    ident: str
    name: str
    nav_type: str
    frequency_mhz: float | None
    latitude: float
    longitude: float
    elevation_ft: float
    published_range_nm: float
    min_distance_to_track_nm: float
    aircraft_fl_at_closest: float
    closest_track_lat: float
    closest_track_lon: float
    rlos_horizon_nm: float
    fms_navigation_role: str
    marker_color: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryRNPPoint:
    seq_num: int
    time_utc: str
    cumulative_dist_nm: float
    latitude: float
    longitude: float
    flight_level: float
    flight_phase: str
    dme_in_rlos_count: int
    vor_in_rlos_count: int
    ils_in_rlos_count: int
    ndb_in_rlos_count: int
    primary_navaids_tuned: str
    anticipated_rnp_nm: float
    pbn_specification: str
    availability_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlightNavaidRNPReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    total_distance_nm: float
    total_active_navaids: int
    dme_stations_used: int
    vor_stations_used: int
    ils_stations_used: int
    ndb_stations_used: int
    best_rnp_nm: float
    worst_rnp_nm: float
    mean_enroute_rnp_nm: float
    dme_dme_coverage_pct: float
    gnss_only_oceanic_pct: float
    overall_navigation_summary: str
    active_navaids: list[ActiveRouteNavaid]
    profile_points: list[TrajectoryRNPPoint]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_rlos_nm(flight_level: float, station_elev_ft: float) -> float:
    """
    Compute VHF/UHF Radio Line-of-Sight (RLOS) horizon distance in nautical miles:
        D_RLOS = 1.23 * (sqrt(h_ac_ft) + sqrt(h_sta_ft))
    """
    h_ac = max(50.0, float(flight_level) * 100.0)
    h_sta = max(0.0, float(station_elev_ft))
    return 1.23 * (math.sqrt(h_ac) + math.sqrt(h_sta))


def analyze_flight_navaids_and_rnp(
    dataset: FlightDataset,
    ectrl_id: int,
    corridor_half_width_nm: float = 35.0,
    max_navaids_on_map: int = 45,
) -> FlightNavaidRNPReport | None:
    """
    Analyze active radio navigation aids (`DME`, `VOR`, `ILS_LOC`, `NDB`) along a flight's 4D
    trajectory and compute the point-by-point anticipated RNP/RNAV capability and rationale.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    act_df = trajs["actual"]
    if act_df.empty:
        return None

    act_df = act_df.sort_values("seq_num").reset_index(drop=True)

    adep = str(flight["adep"])
    ades = str(flight["ades"])
    ac_op = str(flight["ac_operator"])
    ac_type = str(flight["ac_type"])

    # Bounding box around trajectory (+2.5 deg margin for RLOS navaid search)
    min_lat = float(act_df["latitude"].min()) - 2.5
    max_lat = float(act_df["latitude"].max()) + 2.5
    min_lon = float(act_df["longitude"].min()) - 3.0
    max_lon = float(act_df["longitude"].max()) + 3.0

    nav_sql = """
        SELECT
            ident,
            name,
            nav_type,
            frequency_mhz,
            latitude,
            longitude,
            coalesce(elevation_ft, 0.0) AS elevation_ft,
            coalesce(range_nm, 40.0) AS range_nm
        FROM nav_aids_xplane
        WHERE nav_type IN ('DME', 'VOR', 'ILS_LOC', 'LOC', 'DME_ILS', 'NDB')
          AND latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?;
    """
    nav_df = dataset.cursor.execute(nav_sql, [min_lat, max_lat, min_lon, max_lon]).df()

    # Pre-extract navaid records into a fast list of tuples for spatial evaluation
    # (ident, name, nav_type, freq, lat, lon, elev_ft, range_nm)
    navaids_raw: list[tuple[str, str, str, float | None, float, float, float, float]] = []
    if not nav_df.empty:
        # Deduplicate co-located identical (ident, nav_type) within 0.02 deg
        nav_df["lat_r"] = nav_df["latitude"].round(2)
        nav_df["lon_r"] = nav_df["longitude"].round(2)
        nav_df = nav_df.drop_duplicates(subset=["ident", "nav_type", "lat_r", "lon_r"])
        for r in nav_df.itertuples(index=False):
            freq_val = float(r.frequency_mhz) if pd.notna(r.frequency_mhz) else None
            navaids_raw.append(
                (
                    str(r.ident),
                    str(r.name),
                    str(r.nav_type),
                    freq_val,
                    float(r.latitude),
                    float(r.longitude),
                    float(r.elevation_ft),
                    float(r.range_nm),
                )
            )

    # Compute cumulative distance along trajectory
    n_pts = len(act_df)
    lats = act_df["latitude"].astype(float).tolist()
    lons = act_df["longitude"].astype(float).tolist()
    fls = act_df["flight_level"].astype(float).tolist()
    seqs = act_df["seq_num"].astype(int).tolist()
    times = act_df["time_over"].astype(str).tolist()

    cum_dists: list[float] = [0.0]
    for i in range(1, n_pts):
        seg_d = haversine_nm(lats[i - 1], lons[i - 1], lats[i], lons[i])
        cum_dists.append(cum_dists[-1] + seg_d)
    total_dist_nm = max(1.0, cum_dists[-1])

    # Track closest approach of each navaid to the trajectory for the "Likely Used Navaids" map layer
    # Key: (ident, nav_type, round(lat,2), round(lon,2))
    closest_navaid_hits: dict[tuple[str, str, float, float], dict[str, Any]] = {}

    profile_points: list[TrajectoryRNPPoint] = []

    for i in range(n_pts):
        lat_i = lats[i]
        lon_i = lons[i]
        fl_i = fls[i]
        dist_from_dep = cum_dists[i]
        dist_to_arr = max(0.0, total_dist_nm - dist_from_dep)

        # Determine flight phase
        if dist_from_dep <= 15.0 and fl_i < 80:
            phase = "INITIAL_CLIMB_DEPARTURE"
        elif dist_to_arr <= 15.0 and fl_i < 60:
            phase = "FINAL_APPROACH"
        elif dist_from_dep <= 40.0 or (i < n_pts // 2 and fl_i < 180):
            phase = "TERMINAL_SID_CLIMB"
        elif dist_to_arr <= 40.0 or (i >= n_pts // 2 and fl_i < 180):
            phase = "TERMINAL_STAR_DESCENT"
        else:
            phase = "EN_ROUTE_CRUISE"

        # Fast bounding box filter around waypoint (~2.2 deg lat, ~3.2 deg lon)
        dme_in_rlos: list[tuple[float, str]] = []
        vor_in_rlos: list[tuple[float, str]] = []
        ils_in_rlos: list[tuple[float, str]] = []
        ndb_in_rlos: list[tuple[float, str]] = []

        for ident, name, ntype, freq, nlat, nlon, nelev, nrng in navaids_raw:
            if abs(nlat - lat_i) > 2.2 or abs(nlon - lon_i) > 3.2:
                continue
            d_nm = haversine_nm(lat_i, lon_i, nlat, nlon)
            rlos_nm = compute_rlos_nm(fl_i, nelev)
            # Effective operational range is bounded by RLOS and published service volume
            # (En-route VHF/UHF stations are reliably received up to min(RLOS, max(nrng, 130nm)))
            if ntype in ("ILS_LOC", "LOC", "DME_ILS"):
                eff_range = min(rlos_nm, 25.0)
            elif ntype == "NDB":
                eff_range = min(75.0, max(nrng, 35.0))
            else:
                eff_range = min(rlos_nm, max(nrng, 110.0), 160.0)

            if d_nm <= eff_range:
                if ntype == "DME":
                    dme_in_rlos.append((d_nm, ident))
                elif ntype == "VOR":
                    vor_in_rlos.append((d_nm, ident))
                elif ntype in ("ILS_LOC", "LOC", "DME_ILS"):
                    if dist_from_dep <= 25.0 or dist_to_arr <= 25.0:
                        ils_in_rlos.append((d_nm, ident))
                elif ntype == "NDB":
                    ndb_in_rlos.append((d_nm, ident))

                # Check if this navaid lies within the active route corridor for FMS auto-tuning
                corridor_lim = 18.0 if ntype in ("ILS_LOC", "LOC", "DME_ILS") else (25.0 if ntype == "NDB" else corridor_half_width_nm)
                if d_nm <= corridor_lim:
                    if ntype in ("ILS_LOC", "LOC", "DME_ILS") and not (
                        (dist_from_dep <= 20.0 and fl_i <= 100) or (dist_to_arr <= 20.0 and fl_i <= 100)
                    ):
                        pass
                    else:
                        key = (ident, ntype, round(nlat, 2), round(nlon, 2))
                        prev = closest_navaid_hits.get(key)
                        if prev is None or d_nm < prev["min_distance_to_track_nm"]:
                            if ntype in ("ILS_LOC", "LOC", "DME_ILS"):
                                role = "Terminal Precision Approach (ILS/LOC/DME)"
                            elif ntype == "DME":
                                role = "FMS Multi-DME Slant-Range Triangulation (DME/DME/IRU)"
                            elif ntype == "VOR":
                                role = "VHF Omnidirectional Airway & Rho-Theta Navigation (VOR/DME)"
                            else:
                                role = "Low/Medium Frequency Beacon Reference (NDB)"
                            closest_navaid_hits[key] = {
                                "ident": ident,
                                "name": name,
                                "nav_type": ntype,
                                "frequency_mhz": freq,
                                "latitude": nlat,
                                "longitude": nlon,
                                "elevation_ft": nelev,
                                "published_range_nm": nrng,
                                "min_distance_to_track_nm": round(d_nm, 2),
                                "aircraft_fl_at_closest": round(fl_i, 0),
                                "closest_track_lat": round(lat_i, 4),
                                "closest_track_lon": round(lon_i, 4),
                                "rlos_horizon_nm": round(rlos_nm, 1),
                                "fms_navigation_role": role,
                                "marker_color": NAVAID_COLOR_MAP.get(ntype, "#00a8ff"),
                                "track_seq": i,
                            }

        dme_in_rlos.sort(key=lambda x: x[0])
        vor_in_rlos.sort(key=lambda x: x[0])
        ils_in_rlos.sort(key=lambda x: x[0])
        ndb_in_rlos.sort(key=lambda x: x[0])

        n_dme = len(dme_in_rlos)
        n_vor = len(vor_in_rlos)
        n_ils = len(ils_in_rlos)
        n_ndb = len(ndb_in_rlos)

        # Build list of top primary tuned navaid idents at this waypoint
        tuned_idents: list[str] = []
        for _, idt in (ils_in_rlos[:1] + dme_in_rlos[:2] + vor_in_rlos[:2]):
            if idt not in tuned_idents:
                tuned_idents.append(idt)
        primary_tuned_str = ", ".join(tuned_idents[:4]) if tuned_idents else "GNSS / IRS Only"

        # Determine Anticipated RNP / RNAV Capability (in nm) and Why
        if phase == "FINAL_APPROACH":
            if n_ils >= 1:
                rnp_nm = 0.1
                spec = "xLS / RNP AR APCH 0.1"
                why = f"Final approach (<15nm, FL{fl_i:.0f}) with active ILS/Localizer lock ({ils_in_rlos[0][1]}) + {n_dme} DMEs in RLOS"
            elif (n_dme + n_vor) >= 2:
                rnp_nm = 0.3
                spec = "RNP APCH 0.3 (LNAV/VNAV)"
                why = f"Final approach (<15nm, FL{fl_i:.0f}) supported by GNSS + {n_dme} DME / {n_vor} VOR stations in RLOS"
            else:
                rnp_nm = 0.3
                spec = "RNP APCH 0.3 (GNSS)"
                why = f"Final approach (<15nm, FL{fl_i:.0f}) relying on GNSS approach sensor"
        elif phase in ("INITIAL_CLIMB_DEPARTURE", "TERMINAL_SID_CLIMB", "TERMINAL_STAR_DESCENT"):
            if n_dme >= 2:
                rnp_nm = 1.0
                spec = "RNAV 1 / RNP 1 (TMA DME/DME + GNSS)"
                why = f"Terminal SID/STAR ({phase.replace('_', ' ').title()}) with {n_dme} DMEs & {n_vor} VORs in RLOS ({primary_tuned_str}) providing full DME/DME/IRU reversion"
            elif (n_dme + n_vor) >= 1:
                rnp_nm = 1.0
                spec = "RNAV 1 (GNSS Primary / VOR-DME Backup)"
                why = f"Terminal airspace with limited DME geometry ({n_dme} DME, {n_vor} VOR in RLOS); relies on GNSS primary + IRU"
            else:
                rnp_nm = 2.0
                spec = "RNAV 2 (GNSS / IRU Terminal)"
                why = "Terminal transition outside terrestrial DME/VOR line-of-sight; GNSS + IRU navigation"
        else:
            # En-route cruise
            if n_dme >= 3:
                rnp_nm = 1.0
                spec = "RNAV 1 / RNP 1 (High-Density Multi-DME/IRU + GNSS)"
                why = f"En-route FL{fl_i:.0f} with {n_dme} DMEs & {n_vor} VORs in RLOS ({primary_tuned_str}) enabling +/-1.0 nm multi-DME triangulation & FRA operations"
            elif n_dme == 2 or (n_dme >= 1 and n_vor >= 1):
                rnp_nm = 2.0
                spec = "RNAV 2 / RNP 2 (Continental DME/DME)"
                why = f"En-route FL{fl_i:.0f} with dual terrestrial lock ({n_dme} DME, {n_vor} VOR in RLOS: {primary_tuned_str}) supporting +/-2.0 nm continental RNAV"
            elif (n_dme + n_vor) == 1:
                rnp_nm = 5.0
                spec = "RNAV 5 (B-RNAV Single VOR/DME + IRU)"
                why = f"Sparse terrestrial coverage at FL{fl_i:.0f} (only 1 station in RLOS: {primary_tuned_str}); terrestrial accuracy limited to +/-5.0 nm B-RNAV"
            else:
                rnp_nm = 10.0
                spec = "RNP 4 / RNP 10 (Oceanic / Remote GNSS + IRS)"
                why = f"Beyond VHF/UHF Radio Line-of-Sight (0 VOR/DME within {compute_rlos_nm(fl_i, 0):.0f} nm horizon at FL{fl_i:.0f}); relies solely on satellite GNSS & triple IRS"

        profile_points.append(
            TrajectoryRNPPoint(
                seq_num=seqs[i],
                time_utc=times[i],
                cumulative_dist_nm=round(dist_from_dep, 1),
                latitude=round(lat_i, 4),
                longitude=round(lon_i, 4),
                flight_level=round(fl_i, 0),
                flight_phase=phase,
                dme_in_rlos_count=n_dme,
                vor_in_rlos_count=n_vor,
                ils_in_rlos_count=n_ils,
                ndb_in_rlos_count=n_ndb,
                primary_navaids_tuned=primary_tuned_str,
                anticipated_rnp_nm=rnp_nm,
                pbn_specification=spec,
                availability_reason=why,
            )
        )

    # Sort discovered corridor navaids chronologically along the route (`track_seq`), prioritizing closest & key types
    all_hits = list(closest_navaid_hits.values())
    if len(all_hits) > max_navaids_on_map:
        # Always keep ILS/LOC hits, and pick the closest DME/VOR/NDB distributed along the route
        ils_hits = [h for h in all_hits if h["nav_type"] in ("ILS_LOC", "LOC", "DME_ILS")]
        other_hits = [h for h in all_hits if h["nav_type"] not in ("ILS_LOC", "LOC", "DME_ILS")]
        other_hits.sort(key=lambda h: h["min_distance_to_track_nm"])
        selected_hits = (ils_hits + other_hits)[:max_navaids_on_map]
    else:
        selected_hits = all_hits

    selected_hits.sort(key=lambda h: (h["track_seq"], h["min_distance_to_track_nm"]))

    active_navaids = [
        ActiveRouteNavaid(
            ident=h["ident"],
            name=h["name"],
            nav_type=h["nav_type"],
            frequency_mhz=h["frequency_mhz"],
            latitude=h["latitude"],
            longitude=h["longitude"],
            elevation_ft=h["elevation_ft"],
            published_range_nm=h["published_range_nm"],
            min_distance_to_track_nm=h["min_distance_to_track_nm"],
            aircraft_fl_at_closest=h["aircraft_fl_at_closest"],
            closest_track_lat=h["closest_track_lat"],
            closest_track_lon=h["closest_track_lon"],
            rlos_horizon_nm=h["rlos_horizon_nm"],
            fms_navigation_role=h["fms_navigation_role"],
            marker_color=h["marker_color"],
        )
        for h in selected_hits
    ]

    dme_cnt = sum(1 for a in active_navaids if a.nav_type == "DME")
    vor_cnt = sum(1 for a in active_navaids if a.nav_type == "VOR")
    ils_cnt = sum(1 for a in active_navaids if a.nav_type in ("ILS_LOC", "LOC", "DME_ILS"))
    ndb_cnt = sum(1 for a in active_navaids if a.nav_type == "NDB")

    rnp_vals = [p.anticipated_rnp_nm for p in profile_points]
    enroute_rnps = [p.anticipated_rnp_nm for p in profile_points if p.flight_phase == "EN_ROUTE_CRUISE"] or rnp_vals
    dme_dme_pts = sum(1 for p in profile_points if p.dme_in_rlos_count >= 2)
    oceanic_pts = sum(1 for p in profile_points if (p.dme_in_rlos_count + p.vor_in_rlos_count) == 0)

    dme_cov_pct = (dme_dme_pts / len(profile_points)) * 100.0 if profile_points else 0.0
    oceanic_pct = (oceanic_pts / len(profile_points)) * 100.0 if profile_points else 0.0

    if oceanic_pct >= 20.0:
        summary_verdict = (
            f"Hybrid Continental + Oceanic PBN Profile: {dme_cov_pct:.1f}% multi-DME coverage "
            f"(RNAV 1 / RNP 1) transitioning to {oceanic_pct:.1f}% Oceanic/Remote beyond-RLOS "
            f"(RNP 4 / RNP 10 GNSS+IRS)."
        )
    elif dme_cov_pct >= 85.0:
        summary_verdict = (
            f"Continuous High-Density Multi-DME Coverage ({dme_cov_pct:.1f}% of trajectory has >=2 DMEs in RLOS): "
            f"Supports uninterrupted RNAV 1 / RNP 1 (1.0 nm) en-route & terminal PBN with full terrestrial DME/DME/IRU reversion."
        )
    else:
        summary_verdict = (
            f"Mixed Continental PBN Coverage ({dme_cov_pct:.1f}% dual-DME coverage): "
            f"Alternates between RNAV 1/2 (1.0–2.0 nm) and RNAV 5 (5.0 nm) in sparse navaid sectors."
        )

    return FlightNavaidRNPReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        total_distance_nm=round(total_dist_nm, 1),
        total_active_navaids=len(active_navaids),
        dme_stations_used=dme_cnt,
        vor_stations_used=vor_cnt,
        ils_stations_used=ils_cnt,
        ndb_stations_used=ndb_cnt,
        best_rnp_nm=min(rnp_vals) if rnp_vals else 1.0,
        worst_rnp_nm=max(rnp_vals) if rnp_vals else 1.0,
        mean_enroute_rnp_nm=round(sum(enroute_rnps) / len(enroute_rnps), 2) if enroute_rnps else 1.0,
        dme_dme_coverage_pct=round(dme_cov_pct, 1),
        gnss_only_oceanic_pct=round(oceanic_pct, 1),
        overall_navigation_summary=summary_verdict,
        active_navaids=active_navaids,
        profile_points=profile_points,
    )
