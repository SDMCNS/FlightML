export type NavigationTab =
  | 'flight-inspector'
  | 'gnss-shield'
  | 'airspace-network'
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

