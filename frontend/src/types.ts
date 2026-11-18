export type NavigationTab =
  | 'flight-inspector'
  | 'airport-ops'
  | 'gnss-shield'
  | 'airspace-network'
  | 'dme-rnav'
  | 'longitudinal-trends'
  | 'climate-eco'
  | 'passenger-copilot'
  | 'math-reference';

export interface FlightSummary {
  ectrl_id: number;
  callsign: string;
  ac_operator: string;
  adep: string;
  ades: string;
  ac_type: string;
  ac_reg: string;
  flown_fl?: number;
  actual_distance_nm?: number;
  planned_distance_nm?: number;
  duration_min?: number;
  dep_delay_min?: number;
  arr_delay_min?: number;
  gnss_status?: 'CONFIRMED_SPOOFING' | 'SUSPECTED_SPOOFING' | 'NOMINAL';
  anomaly_score?: number;
}

export interface Waypoint {
  seq_num: number;
  time_over: string;
  latitude: number;
  longitude: number;
  flight_level: number;
  speed_kts?: number;
  vs_fpm?: number;
  is_anomaly?: boolean;
  anomaly_type?: string;
  description?: string;
}

export interface GNSSAnomalyEvent {
  seq_num: number;
  time_utc: string;
  latitude: number;
  longitude: number;
  flight_level: number;
  anomaly_type: string;
  metric_value: number;
  metric_unit: string;
  severity_score: number;
  description: string;
}

export interface GNSSReport {
  ectrl_id: number;
  period: string;
  adep: string;
  ades: string;
  ac_operator: string;
  ac_type: string;
  ac_reg: string;
  gnss_status: string;
  anomaly_score: number;
  max_groundspeed_kts: number;
  max_vertical_speed_fpm: number;
  max_track_jitter_deg: number;
  total_anomaly_events: number;
  hotspots_traversed: string[];
  events: GNSSAnomalyEvent[];
  summary_verdict: string;
}

export interface GNSSNetworkScan {
  period: string;
  total_flights_scanned: number;
  flagged_flights_count: number;
  spoofing_confirmed_count: number;
  spoofing_suspected_count: number;
  top_hotspots: Record<string, number>;
  top_affected_operators: Record<string, number>;
  flagged_flights: GNSSReport[];
}

export interface ItineraryRisk {
  origin: string;
  destination: string;
  hub?: string;
  date: string;
  layover_min?: number;
  missed_connection_prob: number;
  predicted_delay_min: number;
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  eu261_eligible: boolean;
  compensation_eur: number;
  recommendation: string;
  mct_min?: number;
  connection_type?: string;
  slack_buffer_min?: number;
  transfer_note?: string;
  mct_details?: Record<string, any>;
}

export interface ContrailSegment {
  seq_start: number;
  seq_end: number;
  start_time_utc: string;
  end_time_utc: string;
  flight_level: number;
  distance_nm: number;
  duration_min: number;
  temperature_c: number;
  rh_ice_pct: number;
  recommended_alt_fl: number;
  alt_deviation_note: string;
}

export interface ContrailReport {
  ectrl_id: number;
  period: string;
  adep: string;
  ades: string;
  ac_operator: string;
  ac_type: string;
  cruise_distance_nm: number;
  avg_groundspeed_kts: number;
  avg_true_airspeed_kts: number;
  avg_tailwind_component_kts: number;
  wind_time_savings_min: number;
  persistent_contrail_distance_nm: number;
  persistent_contrail_pct: number;
  persistent_contrail_duration_min: number;
  contrail_co2e_tons: number;
  direct_co2_tons: number;
  total_climate_co2e_multiplier: number;
  issr_segments_count: number;
  issr_segments: ContrailSegment[];
  annotated_profile?: Array<{
    seq_num: number;
    flight_level: number;
    recommended_alt_fl: number;
    temperature_c: number;
    rh_ice_pct: number;
    is_issr: boolean;
    forms_contrail: boolean;
  }>;
}

export interface CDOReport {
  ectrl_id: number;
  ac_type: string;
  route: string;
  top_of_descent_time: string;
  touchdown_time: string;
  total_descent_duration_min: number;
  cruise_flight_level: number;
  inefficiency_events_count: number;
  total_level_or_climb_min: number;
  cdo_compliance_score_pct: number;
  excess_fuel_burned_kg: number;
  excess_co2_kg: number;
}

export interface FIRCostBreakdown {
  fir_code: string;
  fir_name: string;
  distance_km: number;
  unit_rate_eur: number;
  charge_eur: number;
}

export interface CRCORouteReport {
  ectrl_id: number;
  period: string;
  adep: string;
  ades: string;
  ac_operator: string;
  ac_type: string;
  mtow_tons: number;
  crco_weight_factor: number;
  actual_distance_km: number;
  total_service_units: number;
  actual_crco_charge_eur: number;
  filed_crco_charge_eur: number;
  crco_charge_delta_eur: number;
  estimated_total_fuel_kg: number;
  estimated_total_fuel_cost_eur: number;
  total_direct_route_cost_eur: number;
  crco_share_of_direct_cost_pct: number;
  fir_breakdown: FIRCostBreakdown[];
}

export interface RunwayAssessment {
  airport_ident: string;
  runway_id: string;
  le_ident: string;
  he_ident: string;
  length_ft: number;
  width_ft: number;
  surface: string;
  le_heading_degT: number;
  he_heading_degT: number;
  le_displaced_threshold_ft: number;
  he_displaced_threshold_ft: number;
  le_lda_ft: number;
  he_lda_ft: number;
  le_crosswind_kts: number;
  le_headwind_kts: number;
  he_crosswind_kts: number;
  he_headwind_kts: number;
  le_is_tailwind: boolean;
  he_is_tailwind: boolean;
  le_exceeds_crosswind_limit: boolean;
  he_exceeds_crosswind_limit: boolean;
  preferred_end: string;
  notes: string;
}

export interface AirportOperationsReport {
  airport_ident: string;
  name: string;
  municipality: string;
  country: string;
  elevation_ft: number;
  period_id: string;
  total_arrivals: number;
  total_departures: number;
  total_movements: number;
  mean_arr_delay_min: number;
  mean_dep_delay_min: number;
  top_airlines: { operator: string; flights: number; share_pct: number }[];
  top_aircraft: { ac_type: string; flights: number; share_pct: number }[];
  hourly_distribution: { hour: number; movements: number }[];
  weather: {
    wind_dir_deg: number;
    wind_speed_kts: number;
    temperature_c: number;
    visibility_km: number;
    qnh_hpa: number;
    source: string;
  };
  runways: RunwayAssessment[];
  recommended_active_config: string;
  operational_advisories: string[];
}

export interface WRAPMetricComparison {
  variable: string;
  phase: string;
  description: string;
  observed_value: number;
  observed_unit: string;
  opt_val: number;
  min_val: number;
  max_val: number;
  dist_model: string;
  dist_params: string;
  z_score: number;
  percentile_pct: number;
  status: 'NORMAL' | 'ABOVE_AVERAGE' | 'BELOW_AVERAGE' | 'TACTICAL_RESTRICTION' | 'ANOMALY';
  interpretation: string;
}

export interface FlightWRAPAnalysisReport {
  ectrl_id: number;
  period: string;
  callsign: string;
  ac_type: string;
  ac_operator: string;
  adep: string;
  ades: string;
  canonical_wrap_type: string;
  total_waypoints: number;
  overall_normalcy_score_pct: number;
  envelope_violations_count: number;
  metrics: WRAPMetricComparison[];
  phase_summaries: Record<string, any>;
  tactical_findings: string[];
}

export interface DMEStation {
  ident: string;
  name: string;
  type: string;
  latitude: number;
  longitude: number;
  elevation_ft: number;
  country: string;
  is_active: boolean;
  service_range_nm: number;
}

export interface GridCellCoverage {
  lat: number;
  lon: number;
  visible_dme_count: number;
  has_valid_dme_dme_pair: boolean;
  best_pair_geometry_deg: number;
  coverage_status: 'REDUNDANT' | 'ADEQUATE' | 'MARGINAL' | 'OUTAGE';
}

export interface DMERNAVCoverageReport {
  flight_level: number;
  altitude_ft: number;
  cell_size_nm: number;
  total_grid_cells: number;
  covered_cells_count: number;
  coverage_percentage: number;
  redundant_cells_percentage: number;
  outage_cells_percentage: number;
  total_dmes_count: number;
  active_dmes_count: number;
  inactive_dmes_count: number;
  suggested_deactivations: {
    ident: string;
    name: string;
    country: string;
    coverage_impact_pct: number;
    remaining_network_coverage_pct: number;
    recommendation: string;
    annual_maintenance_savings_eur: number;
  }[];
  dme_stations: DMEStation[];
  cells_summary: GridCellCoverage[];
}

export interface VerticalCappingReport {
  period: string;
  sample_size: number;
  capped_flights_count: number;
  capping_rate_pct: number;
  mean_deficit_fl: number;
  mean_fuel_penalty_pct: number;
  total_excess_fuel_tonnes: number;
  total_excess_co2_tonnes: number;
  by_market_segment: {
    market_segment: string;
    total_flights: number;
    capped_flights: number;
    capping_rate_pct: number;
    mean_deficit_fl: number;
    estimated_penalty_pct: number;
  }[];
  top_capped_routes: {
    route: string;
    adep: string;
    ades: string;
    flights: number;
    capped_flights: number;
    capping_rate_pct: number;
    mean_deficit_fl: number;
  }[];
}

export interface AirwayFRADividendReport {
  period: string;
  sample_size: number;
  mean_airway_distance_nm: number;
  mean_actual_distance_nm: number;
  mean_gcd_distance_nm: number;
  total_distance_saved_nm: number;
  mean_distance_saved_pct: number;
  total_fuel_saved_tonnes: number;
  total_co2_avoided_tonnes: number;
  top_beneficiary_routes: {
    route: string;
    flights: number;
    mean_saved_nm: number;
    mean_saved_pct: number;
    total_fuel_saved_tonnes: number;
    total_co2_avoided_tonnes: number;
  }[];
}


