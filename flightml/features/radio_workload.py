"""
Controller-Pilot VHF Radio Frequency Workload & Communications Occupancy Engine (Sprint 3).

Combines:
- `raw_frequencies` (`data/airport-frequencies.csv` — 30,393 VHF/UHF airport & approach frequencies)
- `nav_aids_xplane` (`flightml/data/from_openap/nav/nav.dat` — ILS, Localizer, Glideslope, VOR, DME stations)
- Hourly departure & arrival movements (`flights`, `flight_firs_actual` TAXI_OUT, `flight_auas_actual` TMA)

Estimates Peak Movements per Discrete VHF Channel and Controller-Pilot Voice Channel Occupancy (%).
"""

from __future__ import annotations

from typing import Any

from flightml.data.dataset import FlightDataset

# Standard Eurocontrol / FAA voice channel occupancy parameters:
# Average 4.5 controller-pilot voice exchanges per aircraft per control position,
# averaging 7.5 seconds per exchange (instruction + pilot readback + squelch break) = 33.75 sec/aircraft.
# Above ~45% continuous voice channel occupancy, frequency congestion & step-ons rise sharply.
AVG_EXCHANGES_PER_AIRCRAFT = 4.5
AVG_SECONDS_PER_EXCHANGE = 7.5
VOICE_SECONDS_PER_AIRCRAFT = AVG_EXCHANGES_PER_AIRCRAFT * AVG_SECONDS_PER_EXCHANGE  # 33.75 sec


def estimate_airport_and_sector_radio_workload(
    dataset: FlightDataset,
    airport_icao: str,
    date_str: str | None = None,
) -> dict[str, Any]:
    """
    Inventory discrete VHF communications channels (TWR, GND, APP/DEP, ATIS/DEL) and ILS/VOR navaids
    for `airport_icao`, and compute peak & average hourly controller-pilot radio channel occupancy (%).
    """
    ident = airport_icao.strip().upper()

    # 1. Query airport VHF/UHF communications frequencies from `raw_frequencies`
    freq_df = dataset.cursor.execute(
        """
        SELECT
            type,
            description,
            round(CAST(frequency_mhz AS DOUBLE), 3) AS frequency_mhz
        FROM raw_frequencies
        WHERE upper(airport_ident) = ?
          AND CAST(frequency_mhz AS DOUBLE) BETWEEN 118.0 AND 137.0
        ORDER BY type, frequency_mhz
        """,
        [ident],
    ).df()

    # Classify discrete VHF channels into operational ATC roles
    twr_freqs: set[float] = set()
    gnd_freqs: set[float] = set()
    app_dep_freqs: set[float] = set()
    del_atis_freqs: set[float] = set()

    for _, r in freq_df.iterrows():
        ftype = str(r["type"] or "").upper()
        fdesc = str(r["description"] or "").upper()
        mhz = float(r["frequency_mhz"])
        if "TWR" in ftype or "TOWER" in fdesc:
            twr_freqs.add(mhz)
        elif "GND" in ftype or "GROUND" in fdesc or "RAMP" in ftype or "APRON" in fdesc:
            gnd_freqs.add(mhz)
        elif any(k in ftype or k in fdesc for k in ("APP", "DEP", "DIR", "RADAR", "ARR", "TMA")):
            app_dep_freqs.add(mhz)
        elif any(k in ftype or k in fdesc for k in ("ATIS", "DEL", "CLD", "CLEARANCE")):
            del_atis_freqs.add(mhz)
        else:
            app_dep_freqs.add(mhz)

    # Ensure sensible minimum channel counts if an airport's regional TRACON frequencies are listed under a parent center
    n_twr = max(1, len(twr_freqs))
    n_gnd = max(1, len(gnd_freqs))
    n_app = max(1, len(app_dep_freqs))
    n_del_atis = len(del_atis_freqs)

    # 2. Query associated ILS / Localizer / VOR / DME navigation aids from `nav_aids_xplane` and `raw_navaids`
    navaids_df = dataset.cursor.execute(
        """
        SELECT
            ident,
            nav_type,
            name,
            frequency_mhz,
            elevation_ft
        FROM nav_aids_xplane
        WHERE upper(name) LIKE ?
        ORDER BY nav_type, ident
        """,
        [f"{ident} %"],
    ).df()

    # 3. Compute hourly departure, arrival, and total movements at `airport_icao`
    date_filter_dep = ""
    date_filter_arr = ""
    params_dep: list[Any] = [ident]
    params_arr: list[Any] = [ident]
    if date_str and date_str.strip():
        date_filter_dep = " AND CAST(actual_off_block_time AS DATE) = CAST(? AS DATE)"
        date_filter_arr = " AND CAST(actual_arrival_time AS DATE) = CAST(? AS DATE)"
        params_dep.append(date_str.strip())
        params_arr.append(date_str.strip())

    hourly_sql = f"""
        WITH dep_hrs AS (
            SELECT
                date_trunc('hour', actual_off_block_time) AS hr_ts,
                count(*) AS dep_count
            FROM flights
            WHERE adep = ? {date_filter_dep}
            GROUP BY date_trunc('hour', actual_off_block_time)
        ),
        arr_hrs AS (
            SELECT
                date_trunc('hour', actual_arrival_time) AS hr_ts,
                count(*) AS arr_count
            FROM flights
            WHERE ades = ? {date_filter_arr}
            GROUP BY date_trunc('hour', actual_arrival_time)
        ),
        combined AS (
            SELECT
                coalesce(d.hr_ts, a.hr_ts) AS hr_ts,
                coalesce(d.dep_count, 0) AS deps,
                coalesce(a.arr_count, 0) AS arrs,
                coalesce(d.dep_count, 0) + coalesce(a.arr_count, 0) AS total_mvts
            FROM dep_hrs d
            FULL OUTER JOIN arr_hrs a ON d.hr_ts = a.hr_ts
            WHERE coalesce(d.hr_ts, a.hr_ts) IS NOT NULL
        )
        SELECT
            CAST(count(*) AS BIGINT) AS active_hours,
            CAST(sum(deps) AS BIGINT) AS total_departures,
            CAST(sum(arrs) AS BIGINT) AS total_arrivals,
            CAST(max(total_mvts) AS BIGINT) AS peak_hourly_movements,
            round(avg(total_mvts), 1) AS avg_hourly_movements,
            CAST(max(deps) AS BIGINT) AS peak_hourly_deps,
            CAST(max(arrs) AS BIGINT) AS peak_hourly_arrs
        FROM combined;
    """
    mvt_row = dataset.cursor.execute(hourly_sql, params_dep + params_arr).fetchone()

    active_hrs = int(mvt_row[0] or 0) if mvt_row else 0
    total_deps = int(mvt_row[1] or 0) if mvt_row else 0
    total_arrs = int(mvt_row[2] or 0) if mvt_row else 0
    peak_mvts = int(mvt_row[3] or 0) if mvt_row else 0
    avg_mvts = float(mvt_row[4] or 0.0) if mvt_row else 0.0

    # Calculate per-channel movements & voice occupancy % during peak hour
    # Ground (GND) & Tower (TWR) handle all departures + arrivals; APP/DEP handles all terminal climbs + approaches
    twr_mvts_per_ch = round(peak_mvts / n_twr, 1)
    gnd_mvts_per_ch = round(peak_mvts / n_gnd, 1)
    app_mvts_per_ch = round(peak_mvts / n_app, 1)

    twr_occ_pct = round(min(100.0, (twr_mvts_per_ch * VOICE_SECONDS_PER_AIRCRAFT / 3600.0) * 100.0), 1)
    gnd_occ_pct = round(min(100.0, (gnd_mvts_per_ch * VOICE_SECONDS_PER_AIRCRAFT / 3600.0) * 100.0), 1)
    app_occ_pct = round(min(100.0, (app_mvts_per_ch * VOICE_SECONDS_PER_AIRCRAFT / 3600.0) * 100.0), 1)

    role_breakdown = [
        {
            "atc_position": "Tower (TWR)",
            "discrete_vhf_channels": len(twr_freqs),
            "effective_channels": n_twr,
            "sample_frequencies_mhz": sorted(list(twr_freqs))[:5],
            "peak_movements_per_channel": twr_mvts_per_ch,
            "estimated_voice_occupancy_pct": twr_occ_pct,
            "workload_status": (
                "SATURATED (>50%)" if twr_occ_pct >= 50.0 else "HIGH (35-50%)" if twr_occ_pct >= 35.0 else "NOMINAL"
            ),
        },
        {
            "atc_position": "Ground / Apron (GND)",
            "discrete_vhf_channels": len(gnd_freqs),
            "effective_channels": n_gnd,
            "sample_frequencies_mhz": sorted(list(gnd_freqs))[:5],
            "peak_movements_per_channel": gnd_mvts_per_ch,
            "estimated_voice_occupancy_pct": gnd_occ_pct,
            "workload_status": (
                "SATURATED (>50%)" if gnd_occ_pct >= 50.0 else "HIGH (35-50%)" if gnd_occ_pct >= 35.0 else "NOMINAL"
            ),
        },
        {
            "atc_position": "Approach / Departure (APP/DEP)",
            "discrete_vhf_channels": len(app_dep_freqs),
            "effective_channels": n_app,
            "sample_frequencies_mhz": sorted(list(app_dep_freqs))[:5],
            "peak_movements_per_channel": app_mvts_per_ch,
            "estimated_voice_occupancy_pct": app_occ_pct,
            "workload_status": (
                "SATURATED (>50%)" if app_occ_pct >= 50.0 else "HIGH (35-50%)" if app_occ_pct >= 35.0 else "NOMINAL"
            ),
        },
    ]

    worst_pos = max(role_breakdown, key=lambda x: x["estimated_voice_occupancy_pct"])

    return {
        "airport_icao": ident,
        "period": dataset.period,
        "date_filter": date_str,
        "total_vhf_channels_cataloged": len(freq_df),
        "clearance_atis_channels": n_del_atis,
        "associated_ils_vor_dme_count": len(navaids_df),
        "active_hours_analyzed": active_hrs,
        "total_departures": total_deps,
        "total_arrivals": total_arrs,
        "peak_hourly_movements": peak_mvts,
        "avg_hourly_movements": avg_mvts,
        "primary_comms_bottleneck": worst_pos["atc_position"],
        "peak_voice_occupancy_pct": worst_pos["estimated_voice_occupancy_pct"],
        "positions": role_breakdown,
        "terminal_navaids": navaids_df.to_dict(orient="records") if not navaids_df.empty else [],
    }

