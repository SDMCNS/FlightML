"""
External Aeronautical, Performance, Emissions & Economic Data Compiler.

Parses and normalizes the external reference datasets located under `flightml/data/`:
1. `from_openap/nav/fix.dat` -> 119,720 global 5-letter RNAV waypoints / fixes (`nav_fixes.parquet`)
2. `from_openap/nav/nav.dat` -> 26,771 global NDB, VOR, ILS, LOC, GS, Marker & DME navaids (`nav_aids_xplane.parquet`)
3. `from_openap/aircraft/*.yml` + `dragpolar/*.yml` + `fuel/fuel_models.csv` + `_synonym.csv` -> Unified aircraft MTOW, CRCO weight factors, drag polars & fuel models (`openap_aircraft.parquet`)
4. `from_openap/wrap/*.txt` + `_synonym.csv` -> Kinematic phase performance envelopes (`openap_wrap.parquet`)
5. `from_openap/engine/engines.csv` -> 426 OpenAP turbofan engine specifications (`openap_engines.parquet`)
6. `edb-emissions-databank_v32B__web_.xlsx` -> Official ICAO EDB v32B Gaseous/Smoke (935 rows) and nvPM Soot Mass/Number (316 rows) (`icao_edb_gaseous.parquet`, `icao_edb_nvpm.parquet`)
7. `unit_rates_eu/ur-2026-01.txt` -> 41 European CRCO En-Route Charging Zones & EUR Unit Rates (`crco_unit_rates.parquet`)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from flightml.config import PARQUET_DIR

PACKAGE_DATA_DIR = Path(__file__).resolve().parent
OPENAP_DIR = PACKAGE_DATA_DIR / "from_openap"
UNIT_RATES_DIR = PACKAGE_DATA_DIR / "unit_rates_eu"
NAVIGATION_OUT_DIR = PACKAGE_DATA_DIR / "navigation"
EDB_XLSX_PATH = PACKAGE_DATA_DIR / "edb-emissions-databank_v32B__web_.xlsx"

NAV_ROW_CODE_TYPES: dict[int, str] = {
    2: "NDB",
    3: "VOR",
    4: "ILS_LOC",
    5: "LOC",
    6: "GLIDESLOPE",
    7: "OUTER_MARKER",
    8: "MIDDLE_MARKER",
    9: "INNER_MARKER",
    12: "DME_ILS",
    13: "DME",
}


def parse_xplane_fix_dat(fix_path: Path | None = None) -> pd.DataFrame:
    """
    Parse X-Plane FixXP700 `fix.dat` into a clean DataFrame of global 5-letter RNAV waypoints.
    """
    src = fix_path or (OPENAP_DIR / "nav" / "fix.dat")
    if not src.exists():
        return pd.DataFrame(columns=["fix_ident", "latitude", "longitude"])

    rows: list[tuple[str, float, float]] = []
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                ident = parts[2].strip().upper()
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    rows.append((ident, lat, lon))
            except ValueError:
                continue

    return pd.DataFrame(rows, columns=["fix_ident", "latitude", "longitude"])


def parse_xplane_nav_dat(nav_path: Path | None = None) -> pd.DataFrame:
    """
    Parse X-Plane NavXP810 `nav.dat` into a structured DataFrame of global radio navigation aids
    and instrument landing systems (NDB, VOR, ILS, LOC, Glideslope, Markers, DME).
    """
    src = nav_path or (OPENAP_DIR / "nav" / "nav.dat")
    if not src.exists():
        return pd.DataFrame(
            columns=[
                "row_code",
                "nav_type",
                "ident",
                "name",
                "latitude",
                "longitude",
                "elevation_ft",
                "raw_freq",
                "frequency_mhz",
                "range_nm",
                "bearing_or_var_deg",
            ]
        )

    rows: list[dict[str, Any]] = []
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                code = int(parts[0])
                if code not in NAV_ROW_CODE_TYPES:
                    continue
                lat = float(parts[1])
                lon = float(parts[2])
                elev_ft = float(parts[3])
                raw_freq = int(parts[4])
                rng_nm = float(parts[5])
                bearing_var = float(parts[6])
                ident = parts[7].strip().upper()
                name = " ".join(parts[8:]).strip()

                # Convert X-Plane integer frequency:
                # NDB (code 2) is in kHz (e.g. 396 -> 0.396 MHz); VOR/ILS/DME (3,4,5,6,12,13) are in 100*MHz (10950 -> 109.50 MHz)
                if code == 2:
                    freq_mhz = round(raw_freq / 1000.0, 4) if raw_freq > 0 else None
                elif code in (3, 4, 5, 6, 12, 13) and raw_freq > 0:
                    freq_mhz = round(raw_freq / 100.0, 2)
                else:
                    freq_mhz = None

                rows.append(
                    {
                        "row_code": code,
                        "nav_type": NAV_ROW_CODE_TYPES[code],
                        "ident": ident,
                        "name": name,
                        "latitude": lat,
                        "longitude": lon,
                        "elevation_ft": elev_ft,
                        "raw_freq": raw_freq,
                        "frequency_mhz": freq_mhz,
                        "range_nm": rng_nm,
                        "bearing_or_var_deg": bearing_var,
                    }
                )
            except ValueError:
                continue

    return pd.DataFrame(rows)


def _load_synonym_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = (row.get("orig") or "").strip().upper()
            new = (row.get("new") or "").strip().upper()
            if orig and new:
                mapping[orig] = new
    return mapping


def parse_openap_aircraft_and_dragpolar() -> pd.DataFrame:
    """
    Combine OpenAP `aircraft/*.yml`, `dragpolar/*.yml`, `fuel/fuel_models.csv`, and `_synonym.csv`
    into a unified aircraft performance, aerodynamic drag polar, and CRCO MTOW weight-factor table.
    """
    ac_dir = OPENAP_DIR / "aircraft"
    dp_dir = OPENAP_DIR / "dragpolar"
    fuel_csv = OPENAP_DIR / "fuel" / "fuel_models.csv"

    # 1. Load drag polar YAML files
    dragpolars: dict[str, dict[str, Any]] = {}
    if dp_dir.exists():
        for yml_file in sorted(dp_dir.glob("*.yml")):
            code = yml_file.stem.upper()
            data = yaml.safe_load(yml_file.read_text(encoding="utf-8")) or {}
            clean = data.get("clean") or {}
            flaps = data.get("flaps") or {}
            dragpolars[code] = {
                "cd0_clean": clean.get("cd0"),
                "k_clean": clean.get("k"),
                "e_clean": clean.get("e"),
                "cd_gears": data.get("gears"),
                "flaps_lambda_f": flaps.get("lambda_f"),
                "flaps_cf_c": flaps.get("cf/c"),
                "flaps_sf_s": flaps.get("Sf/S"),
            }

    dp_synonyms = _load_synonym_map(dp_dir / "_synonym.csv")

    # 2. Load fuel models CSV
    fuel_models: dict[str, dict[str, Any]] = {}
    if fuel_csv.exists():
        df_fuel = pd.read_csv(fuel_csv)
        for _, r in df_fuel.iterrows():
            tcode = str(r["typecode"]).strip().upper()
            fuel_models[tcode] = {
                "fuel_model_engine": r["engine_type"],
                "fuel_c1": float(r["c1"]),
                "fuel_c2": float(r["c2"]),
                "fuel_c3": float(r["c3"]),
            }

    default_fuel = fuel_models.get("DEFAULT", {"fuel_model_engine": "default", "fuel_c1": 0.9376, "fuel_c2": 1.9768, "fuel_c3": 1.3955})

    # 3. Load primary aircraft YAML files
    base_aircraft: dict[str, dict[str, Any]] = {}
    if ac_dir.exists():
        for yml_file in sorted(ac_dir.glob("*.yml")):
            code = yml_file.stem.upper()
            data = yaml.safe_load(yml_file.read_text(encoding="utf-8")) or {}
            mtow_kg = float(data.get("mtow") or 0.0)
            mtow_tons = mtow_kg / 1000.0
            crco_wf = round(math.sqrt(mtow_tons / 50.0), 4) if mtow_tons > 0 else None

            # Determine ICAO wake turbulence category from MTOW
            if mtow_kg >= 500_000:
                wake_cat = "J"  # Super (A388)
            elif mtow_kg >= 136_000:
                wake_cat = "H"  # Heavy
            elif mtow_kg >= 7_000:
                wake_cat = "M"  # Medium
            else:
                wake_cat = "L"  # Light

            pax = data.get("pax") or {}
            wing = data.get("wing") or {}
            cruise = data.get("cruise") or {}
            eng = data.get("engine") or {}
            eng_opts = eng.get("options")
            if isinstance(eng_opts, dict):
                eng_list = list(eng_opts.values())
            elif isinstance(eng_opts, list):
                eng_list = eng_opts
            else:
                eng_list = []

            dp_key = code if code in dragpolars else dp_synonyms.get(code, "")
            dp_info = dragpolars.get(dp_key, {})
            f_info = fuel_models.get(code, default_fuel)

            base_aircraft[code] = {
                "ac_type": code,
                "canonical_type": code,
                "is_synonym": False,
                "aircraft_name": data.get("aircraft", code),
                "wake_category": wake_cat,
                "mtow_kg": mtow_kg,
                "mtow_tons": mtow_tons,
                "crco_weight_factor": crco_wf,
                "mlw_kg": data.get("mlw"),
                "oew_kg": data.get("oew"),
                "mfc_kg": data.get("mfc"),
                "vmo_kts": data.get("vmo"),
                "mmo_mach": data.get("mmo"),
                "ceiling_m": data.get("ceiling"),
                "pax_max": pax.get("max"),
                "wing_area_m2": wing.get("area"),
                "wing_span_m": wing.get("span"),
                "wing_mac_m": wing.get("mac"),
                "wing_sweep_deg": wing.get("sweep"),
                "cruise_height_m": cruise.get("height"),
                "cruise_mach": cruise.get("mach"),
                "cruise_range_km": cruise.get("range"),
                "engine_type": eng.get("type"),
                "engine_mount": eng.get("mount"),
                "engine_count": eng.get("number"),
                "default_engine": eng.get("default"),
                "engine_options": ", ".join(str(x) for x in eng_list),
                "cd0_clean": dp_info.get("cd0_clean"),
                "k_clean": dp_info.get("k_clean"),
                "e_clean": dp_info.get("e_clean"),
                "cd_gears": dp_info.get("cd_gears"),
                "fuel_c1": f_info.get("fuel_c1"),
                "fuel_c2": f_info.get("fuel_c2"),
                "fuel_c3": f_info.get("fuel_c3"),
            }

    # 4. Expand aircraft synonyms
    ac_synonyms = _load_synonym_map(ac_dir / "_synonym.csv")
    rows = list(base_aircraft.values())
    for orig_code, target_code in ac_synonyms.items():
        if orig_code not in base_aircraft and target_code in base_aircraft:
            cloned = dict(base_aircraft[target_code])
            cloned["ac_type"] = orig_code
            cloned["canonical_type"] = target_code
            cloned["is_synonym"] = True
            # Check if dragpolar has a more specific mapping for orig_code
            dp_key = orig_code if orig_code in dragpolars else dp_synonyms.get(orig_code, target_code)
            if dp_key in dragpolars:
                for k, v in dragpolars[dp_key].items():
                    if k in cloned:
                        cloned[k] = v
            rows.append(cloned)

    return pd.DataFrame(rows).sort_values("ac_type").reset_index(drop=True)


def parse_openap_wrap_kinematics() -> pd.DataFrame:
    """
    Parse OpenAP kinematic WRAP parameter files (`from_openap/wrap/*.txt` + `_synonym.csv`).
    """
    wrap_dir = OPENAP_DIR / "wrap"
    if not wrap_dir.exists():
        return pd.DataFrame()

    base_rows: dict[str, list[dict[str, Any]]] = {}
    for txt_file in sorted(wrap_dir.glob("*.txt")):
        code = txt_file.stem.upper()
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        file_records: list[dict[str, Any]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            # Fixed-width / multi-space columns: variable, flight_phase, name, opt, min, max, model, parameters
            # Split from left (first 2 tokens) and right (last 5 tokens)
            tokens = line.split()
            if len(tokens) < 8:
                continue
            var_name = tokens[0]
            phase = tokens[1]
            params = tokens[-1]
            model = tokens[-2]
            try:
                max_val = float(tokens[-3])
                min_val = float(tokens[-4])
                opt_val = float(tokens[-5])
                desc = " ".join(tokens[2:-5])
            except ValueError:
                continue

            file_records.append(
                {
                    "ac_type": code,
                    "canonical_wrap_type": code,
                    "variable": var_name,
                    "flight_phase": phase,
                    "description": desc,
                    "opt_val": opt_val,
                    "min_val": min_val,
                    "max_val": max_val,
                    "dist_model": model,
                    "dist_params": params,
                }
            )
        base_rows[code] = file_records

    all_records: list[dict[str, Any]] = []
    for recs in base_rows.values():
        all_records.extend(recs)

    synonyms = _load_synonym_map(wrap_dir / "_synonym.csv")
    for orig, target in synonyms.items():
        if orig not in base_rows and target in base_rows:
            for r in base_rows[target]:
                cloned = dict(r)
                cloned["ac_type"] = orig
                all_records.append(cloned)

    return pd.DataFrame(all_records)


def parse_crco_unit_rates(ur_path: Path | None = None) -> pd.DataFrame:
    """
    Parse Eurocontrol CRCO En-Route Unit Rates (`unit_rates_eu/ur-2026-01.txt`).
    Note: Raw integer values in CRCO `ur-*.txt` files are in Euro-cents (e.g., 8775 -> 87.75 EUR).
    """
    src = ur_path or (UNIT_RATES_DIR / "ur-2026-01.txt")
    if not src.exists():
        return pd.DataFrame()

    # Standard mapping from CRCO 2-letter zone code to primary European FIR prefixes
    rows: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 4:
            continue
        zone_code = parts[0].upper()
        valid_from = parts[1]
        valid_to = parts[2]
        try:
            raw_cents = int(parts[3])
        except ValueError:
            continue
        unit_rate_eur = round(raw_cents / 100.0, 2)
        exch_rate = float(parts[4]) if len(parts) > 4 and parts[4] else 1.0
        currency = parts[5] if len(parts) > 5 and parts[5] else "EUR"
        country_name = parts[-1] if len(parts) >= 5 else zone_code

        rows.append(
            {
                "zone_code": zone_code,
                "country_name": country_name,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "unit_rate_cents": raw_cents,
                "unit_rate_eur": unit_rate_eur,
                "exchange_rate_to_eur": exch_rate,
                "national_currency": currency,
            }
        )

    return pd.DataFrame(rows)


def parse_icao_edb_emissions(xlsx_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the ICAO Aircraft Engine Emissions Databank v32B Excel workbook
    (`edb-emissions-databank_v32B__web_.xlsx`) into Gaseous/Smoke and nvPM DataFrames.
    """
    src = xlsx_path or EDB_XLSX_PATH
    if not src.exists():
        return pd.DataFrame(), pd.DataFrame()

    df_gas_raw = pd.read_excel(src, sheet_name="Gaseous Emissions and Smoke")
    gas_cols = {
        "UID No": "uid",
        "Manufacturer": "manufacturer",
        "Engine Identification": "engine_name",
        "Combustor Description": "combustor",
        "Eng Type": "engine_type",
        "B/P Ratio": "bypass_ratio",
        "Pressure Ratio": "pressure_ratio",
        "Rated Thrust (kN)": "rated_thrust_kn",
        "Current Engine Status": "status",
        "HC EI T/O (g/kg)": "ei_hc_to_g_kg",
        "HC EI C/O (g/kg)": "ei_hc_co_g_kg",
        "HC EI App (g/kg)": "ei_hc_app_g_kg",
        "HC EI Idle (g/kg)": "ei_hc_idle_g_kg",
        "CO EI T/O (g/kg)": "ei_co_to_g_kg",
        "CO EI C/O (g/kg)": "ei_co_co_g_kg",
        "CO EI App (g/kg)": "ei_co_app_g_kg",
        "CO EI Idle (g/kg)": "ei_co_idle_g_kg",
        "NOx EI T/O (g/kg)": "ei_nox_to_g_kg",
        "NOx EI C/O (g/kg)": "ei_nox_co_g_kg",
        "NOx EI App (g/kg)": "ei_nox_app_g_kg",
        "NOx EI Idle (g/kg)": "ei_nox_idle_g_kg",
        "SN T/O": "smoke_num_to",
        "SN C/O": "smoke_num_co",
        "SN App": "smoke_num_app",
        "SN Idle": "smoke_num_idle",
        "SN Max": "smoke_num_max",
        "Fuel Flow T/O (kg/sec)": "ff_to_kg_s",
        "Fuel Flow C/O (kg/sec)": "ff_co_kg_s",
        "Fuel Flow App (kg/sec)": "ff_app_kg_s",
        "Fuel Flow Idle (kg/sec)": "ff_idle_kg_s",
        "Fuel LTO Cycle (kg)  ": "fuel_lto_cycle_kg",
    }
    avail_gas = {k: v for k, v in gas_cols.items() if k in df_gas_raw.columns}
    df_gas = df_gas_raw[list(avail_gas.keys())].rename(columns=avail_gas).dropna(subset=["uid", "engine_name"])
    for col in df_gas.columns:
        if df_gas[col].dtype == object and col not in ("uid", "manufacturer", "engine_name", "combustor", "engine_type", "status"):
            df_gas[col] = pd.to_numeric(df_gas[col], errors="coerce")
        elif col in ("uid", "manufacturer", "engine_name", "combustor", "engine_type", "status"):
            df_gas[col] = df_gas[col].astype(str)

    df_nvpm_raw = pd.read_excel(src, sheet_name="nvPM Emissions")
    nvpm_cols = {
        "UID No": "uid",
        "Manufacturer": "manufacturer",
        "Engine Identification": "engine_name",
        "Combustor Description": "combustor",
        "Eng Type": "engine_type",
        "B/P Ratio": "bypass_ratio",
        "Pressure Ratio": "pressure_ratio",
        "Rated Thrust (kN)": "rated_thrust_kn",
        "Fuel Flow T/O (kg/sec)": "ff_to_kg_s",
        "Fuel Flow C/O (kg/sec)": "ff_co_kg_s",
        "Fuel Flow App (kg/sec)": "ff_app_kg_s",
        "Fuel Flow Idle (kg/sec)": "ff_idle_kg_s",
        "nvPM EImass T/O (mg/kg)": "nvpm_ei_mass_to_mg_kg",
        "nvPM EImass C/O (mg/kg)": "nvpm_ei_mass_co_mg_kg",
        "nvPM EImass App (mg/kg)": "nvpm_ei_mass_app_mg_kg",
        "nvPM EImass Idle (mg/kg)": "nvpm_ei_mass_idle_mg_kg",
        "nvPM EImass Max (mg/kg)": "nvpm_ei_mass_max_mg_kg",
        "nvPM LTO Total Mass (mg)": "nvpm_lto_total_mass_mg",
        "nvPM EInum T/O (#/kg)": "nvpm_ei_num_to_per_kg",
        "nvPM EInum C/O (#/kg)": "nvpm_ei_num_co_per_kg",
        "nvPM EInum App (#/kg)": "nvpm_ei_num_app_per_kg",
        "nvPM EInum Idle (#/kg)": "nvpm_ei_num_idle_per_kg",
        "nvPM EInum Max (#/kg)": "nvpm_ei_num_max_per_kg",
        "nvPM LTO Total Particle Number (#)": "nvpm_lto_total_num",
    }
    avail_nvpm = {k: v for k, v in nvpm_cols.items() if k in df_nvpm_raw.columns}
    df_nvpm = df_nvpm_raw[list(avail_nvpm.keys())].rename(columns=avail_nvpm).dropna(subset=["uid", "engine_name"])
    for col in df_nvpm.columns:
        if df_nvpm[col].dtype == object and col not in ("uid", "manufacturer", "engine_name", "combustor", "engine_type"):
            df_nvpm[col] = pd.to_numeric(df_nvpm[col], errors="coerce")
        elif col in ("uid", "manufacturer", "engine_name", "combustor", "engine_type"):
            df_nvpm[col] = df_nvpm[col].astype(str)

    return df_gas.reset_index(drop=True), df_nvpm.reset_index(drop=True)


def compile_all_external_sources(parquet_dir: Path | None = None, force: bool = False) -> dict[str, dict[str, Any]]:
    """
    Compile all external sources in `flightml/data/` into high-speed Apache Parquet tables
    in `data/parquet/` (and summary CSVs in `flightml/data/navigation/`).
    """
    out_dir = Path(parquet_dir or PARQUET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    NAVIGATION_OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any]] = {}

    # 1. Global Waypoints / Fixes (`fix.dat`)
    fixes_pq = out_dir / "nav_fixes.parquet"
    if force or not fixes_pq.exists():
        df_fixes = parse_xplane_fix_dat()
        if not df_fixes.empty:
            df_fixes.to_parquet(fixes_pq, index=False)
    else:
        df_fixes = pd.read_parquet(fixes_pq)
    summary["nav_fixes"] = {
        "source": "flightml/data/from_openap/nav/fix.dat",
        "parquet": str(fixes_pq),
        "rows": len(df_fixes),
        "description": "Global 5-letter RNAV waypoints & intersection fixes (FixXP700)",
        "supports_sprints": "Sprint 2 (KEP/KEA), Sprint 3 (FRA & Airway Snapping)",
    }

    # 2. Global Radio Navaids & ILS (`nav.dat`)
    nav_pq = out_dir / "nav_aids_xplane.parquet"
    if force or not nav_pq.exists():
        df_nav = parse_xplane_nav_dat()
        if not df_nav.empty:
            df_nav.to_parquet(nav_pq, index=False)
    else:
        df_nav = pd.read_parquet(nav_pq)
    summary["nav_aids_xplane"] = {
        "source": "flightml/data/from_openap/nav/nav.dat",
        "parquet": str(nav_pq),
        "rows": len(df_nav),
        "description": "Global NDB, VOR, ILS Localizer, Glideslope, Marker & DME stations (NavXP810)",
        "supports_sprints": "Sprint 3 (FRA Navaid Snapping), Sprint 5 (ILS/Runway Safety)",
    }

    # 3. OpenAP Aircraft Performance, MTOW & Drag Polars
    ac_pq = out_dir / "openap_aircraft.parquet"
    ac_csv = NAVIGATION_OUT_DIR / "openap_aircraft_compiled.csv"
    if force or not ac_pq.exists():
        df_ac = parse_openap_aircraft_and_dragpolar()
        if not df_ac.empty:
            df_ac.to_parquet(ac_pq, index=False)
            df_ac.to_csv(ac_csv, index=False)
    else:
        df_ac = pd.read_parquet(ac_pq)
    summary["openap_aircraft"] = {
        "source": "flightml/data/from_openap/{aircraft,dragpolar,fuel}/*",
        "parquet": str(ac_pq),
        "rows": len(df_ac),
        "description": "Aircraft MTOW, CRCO weight factors, wake category, drag polars & fuel polynomials",
        "supports_sprints": "Sprint 2 (CCO/CDO), Sprint 4 (CRCO Charges), Sprint 5 (Runway Braking)",
    }

    # 4. OpenAP WRAP Kinematic Envelopes
    wrap_pq = out_dir / "openap_wrap.parquet"
    if force or not wrap_pq.exists():
        df_wrap = parse_openap_wrap_kinematics()
        if not df_wrap.empty:
            df_wrap.to_parquet(wrap_pq, index=False)
    else:
        df_wrap = pd.read_parquet(wrap_pq)
    summary["openap_wrap"] = {
        "source": "flightml/data/from_openap/wrap/*.txt",
        "parquet": str(wrap_pq),
        "rows": len(df_wrap),
        "description": "Kinematic flight-phase speed, climb/descent rate & altitude envelopes (WRAP)",
        "supports_sprints": "Sprint 2 (CCO/CDO/Level-Capping), Sprint 5 (Takeoff/Landing Distance)",
    }

    # 5. OpenAP Engines CSV
    eng_pq = out_dir / "openap_engines.parquet"
    eng_src = OPENAP_DIR / "engine" / "engines.csv"
    if (force or not eng_pq.exists()) and eng_src.exists():
        df_eng = pd.read_csv(eng_src)
        df_eng.to_parquet(eng_pq, index=False)
    else:
        df_eng = pd.read_parquet(eng_pq) if eng_pq.exists() else pd.DataFrame()
    summary["openap_engines"] = {
        "source": "flightml/data/from_openap/engine/engines.csv",
        "parquet": str(eng_pq),
        "rows": len(df_eng),
        "description": "OpenAP turbofan thrust, bypass ratio, cruise SFC & LTO fuel flows",
        "supports_sprints": "Sprint 1 (Taxi Fuel), Sprint 2 (Climb/Cruise Fuel)",
    }

    # 6. ICAO EDB v32B Gaseous & nvPM Soot Emissions
    gas_pq = out_dir / "icao_edb_gaseous.parquet"
    nvpm_pq = out_dir / "icao_edb_nvpm.parquet"
    if force or not gas_pq.exists() or not nvpm_pq.exists():
        df_gas, df_nvpm = parse_icao_edb_emissions()
        if not df_gas.empty:
            df_gas.to_parquet(gas_pq, index=False)
        if not df_nvpm.empty:
            df_nvpm.to_parquet(nvpm_pq, index=False)
    else:
        df_gas = pd.read_parquet(gas_pq)
        df_nvpm = pd.read_parquet(nvpm_pq)
    summary["icao_edb_gaseous"] = {
        "source": "flightml/data/edb-emissions-databank_v32B__web_.xlsx [Gaseous]",
        "parquet": str(gas_pq),
        "rows": len(df_gas),
        "description": "ICAO EDB v32B NOx, CO, HC & Smoke Number across T/O, Climb, Approach, Idle",
        "supports_sprints": "Sprint 1 (TAXI_OUT Emissions), Sprint 2 (LTO Environmental KPIs)",
    }
    summary["icao_edb_nvpm"] = {
        "source": "flightml/data/edb-emissions-databank_v32B__web_.xlsx [nvPM]",
        "parquet": str(nvpm_pq),
        "rows": len(df_nvpm),
        "description": "ICAO EDB v32B non-volatile Particulate Matter (soot mass & particle number)",
        "supports_sprints": "Sprint 1 (Airport Air Quality), Sprint 6 (Contrail Ice Nucleation)",
    }

    # 7. Eurocontrol CRCO En-Route Unit Rates
    crco_pq = out_dir / "crco_unit_rates.parquet"
    crco_csv = NAVIGATION_OUT_DIR / "crco_unit_rates_compiled.csv"
    if force or not crco_pq.exists():
        df_crco = parse_crco_unit_rates()
        if not df_crco.empty:
            df_crco.to_parquet(crco_pq, index=False)
            df_crco.to_csv(crco_csv, index=False)
    else:
        df_crco = pd.read_parquet(crco_pq)
    summary["crco_unit_rates"] = {
        "source": "flightml/data/unit_rates_eu/ur-2026-01.txt",
        "parquet": str(crco_pq),
        "rows": len(df_crco),
        "description": "Eurocontrol CRCO national charging zone unit rates (EUR) & FX rates",
        "supports_sprints": "Sprint 4 (CRCO Route Charge vs. Fuel Burn Optimizer)",
    }

    return summary

