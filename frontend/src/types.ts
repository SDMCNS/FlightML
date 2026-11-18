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
}
