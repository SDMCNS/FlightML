import React, { useState, useEffect } from 'react';
import type { FlightSummary, Waypoint, ContrailReport, CDOReport, CRCORouteReport } from '../types';
import { LeafletMap } from '../components/LeafletMap';
import {
  fetchFlightContrails,
  fetchFlightCDO,
  fetchFlightCRCO,
} from '../services/api';
import {
  Plane,
  AlertTriangle,
  ArrowRight,
  Gauge,
  Clock,
  MapPin,
  Compass,
  Leaf,
  Euro,
  TrendingDown,
  Info,
} from 'lucide-react';

interface FlightInspectorViewProps {
  flights: FlightSummary[];
  selectedFlight: FlightSummary | null;
  onSelectFlight: (f: FlightSummary) => void;
  waypoints: Waypoint[];
  period?: string;
  isLoading?: boolean;
}

type SubTab = 'map' | 'contrail' | 'profile' | 'cdo' | 'crco' | 'telemetry';

export const FlightInspectorView: React.FC<FlightInspectorViewProps> = ({
  flights,
  selectedFlight,
  onSelectFlight,
  waypoints,
  period = '202402',
  isLoading = false,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('map');
  const [contrailData, setContrailData] = useState<ContrailReport | null>(null);
  const [cdoData, setCdoData] = useState<CDOReport | null>(null);
  const [crcoData, setCrcoData] = useState<CRCORouteReport | null>(null);
  const [isSubLoading, setIsSubLoading] = useState<boolean>(false);

  // Fetch contextual telemetry when flight changes
  useEffect(() => {
    let cancelled = false;
    async function loadFlightInsights() {
      if (!selectedFlight) {
        setContrailData(null);
        setCdoData(null);
        setCrcoData(null);
        return;
      }
      setIsSubLoading(true);
      try {
        const [cRes, cdoRes, crcoRes] = await Promise.all([
          fetchFlightContrails(selectedFlight.ectrl_id, period),
          fetchFlightCDO(selectedFlight.ectrl_id, period),
          fetchFlightCRCO(selectedFlight.ectrl_id, period),
        ]);
        if (!cancelled) {
          setContrailData(cRes);
          setCdoData(cdoRes);
          setCrcoData(crcoRes);
        }
      } catch (err) {
        console.error('Error loading flight subtab details:', err);
      } finally {
        if (!cancelled) setIsSubLoading(false);
      }
    }
    loadFlightInsights();
    return () => {
      cancelled = true;
    };
  }, [selectedFlight?.ectrl_id, period]);

  return (
    <div className="view-container flight-inspector">
      <div className="flight-master-detail">
        {/* Left Pane: Flight Roster */}
        <div className="flight-list-panel">
          <div className="panel-header">
            <h3>Active Flights ({flights.length})</h3>
            <span className="panel-hint">
              AIRAC {period} • {isLoading || isSubLoading ? 'Loading telemetry...' : 'Click to inspect'}
            </span>
          </div>

          <div className="flight-items-scroll">
            {flights.map(f => {
              const isSelected = selectedFlight?.ectrl_id === f.ectrl_id;
              const isSpoofed = f.gnss_status === 'CONFIRMED_SPOOFING';
              const isSuspect = f.gnss_status === 'SUSPECTED_SPOOFING';
              return (
                <div
                  key={f.ectrl_id}
                  onClick={() => onSelectFlight(f)}
                  className={`flight-card ${isSelected ? 'selected' : ''}`}
                >
                  <div className="flight-card-header">
                    <span className="callsign">{f.callsign}</span>
                    <span className="operator-pill">{f.ac_operator}</span>
                  </div>
                  <div className="flight-card-route">
                    <span className="airport-code">{f.adep}</span>
                    <ArrowRight size={14} className="route-arrow" />
                    <span className="airport-code">{f.ades}</span>
                    <span className="aircraft-pill">{f.ac_type}</span>
                  </div>
                  <div className="flight-card-meta">
                    <span>FL{f.flown_fl}</span>
                    <span>{f.actual_distance_nm} nm</span>
                    <span className={f.arr_delay_min && f.arr_delay_min > 15 ? 'delay-red' : 'delay-green'}>
                      {f.arr_delay_min ? `${f.arr_delay_min > 0 ? '+' : ''}${f.arr_delay_min}m` : 'On Time'}
                    </span>
                    {isSpoofed && (
                      <span className="anomaly-pill text-red">
                        <AlertTriangle size={11} /> Spoofed
                      </span>
                    )}
                    {isSuspect && !isSpoofed && (
                      <span className="anomaly-pill text-yellow">
                        <AlertTriangle size={11} /> Jitter
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Pane: 4D Telemetry & Analytics */}
        <div className="flight-detail-panel">
          {selectedFlight ? (
            <>
              {/* Executive Flight Header */}
              <div className="flight-hero-card">
                <div className="hero-left">
                  <div className="plane-badge">
                    <Plane size={24} color="#3b82f6" />
                  </div>
                  <div>
                    <div className="hero-title-row">
                      <h2>{selectedFlight.callsign}</h2>
                      <span className="hero-id-pill">ID #{selectedFlight.ectrl_id}</span>
                      <span className="hero-reg-pill">{selectedFlight.ac_reg || 'N/A'}</span>
                    </div>
                    <div className="hero-subtitle">
                      <strong>{selectedFlight.adep}</strong> ➔ <strong>{selectedFlight.ades}</strong> •{' '}
                      {selectedFlight.ac_type} • {selectedFlight.ac_operator}
                    </div>
                  </div>
                </div>

                <div className="hero-stats-grid">
                  <div className="hero-stat">
                    <span className="stat-label">Cruise FL</span>
                    <span className="stat-val">FL{selectedFlight.flown_fl || 350}</span>
                  </div>
                  <div className="hero-stat">
                    <span className="stat-label">Actual Dist</span>
                    <span className="stat-val">{selectedFlight.actual_distance_nm || '—'} nm</span>
                  </div>
                  <div className="hero-stat">
                    <span className="stat-label">Arr Delay</span>
                    <span
                      className={`stat-val ${
                        selectedFlight.arr_delay_min && selectedFlight.arr_delay_min > 15 ? 'text-red' : 'text-green'
                      }`}
                    >
                      {selectedFlight.arr_delay_min
                        ? `${selectedFlight.arr_delay_min > 0 ? '+' : ''}${selectedFlight.arr_delay_min} min`
                        : '0 min'}
                    </span>
                  </div>
                  <div className="hero-stat">
                    <span className="stat-label">GNSS Shield</span>
                    <span
                      className={`stat-val ${
                        selectedFlight.gnss_status === 'CONFIRMED_SPOOFING' ? 'text-red' : 'text-green'
                      }`}
                    >
                      {selectedFlight.gnss_status || 'NOMINAL'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Sub-view Navigation Bar */}
              <div className="detail-tabs">
                <button
                  className={`detail-tab ${activeSubTab === 'map' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('map')}
                >
                  <MapPin size={15} /> 4D Radar Map
                </button>
                <button
                  className={`detail-tab ${activeSubTab === 'contrail' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('contrail')}
                >
                  <Leaf size={15} /> Contrail & Eco Route
                </button>
                <button
                  className={`detail-tab ${activeSubTab === 'profile' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('profile')}
                >
                  <Gauge size={15} /> Vertical Profile
                </button>
                <button
                  className={`detail-tab ${activeSubTab === 'cdo' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('cdo')}
                >
                  <TrendingDown size={15} /> Continuous Descent (CDO)
                </button>
                <button
                  className={`detail-tab ${activeSubTab === 'crco' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('crco')}
                >
                  <Euro size={15} /> Route Charges (CRCO)
                </button>
                <button
                  className={`detail-tab ${activeSubTab === 'telemetry' ? 'active' : ''}`}
                  onClick={() => setActiveSubTab('telemetry')}
                >
                  <Clock size={15} /> Telemetry ({waypoints.length} WPTs)
                </button>
              </div>

              {/* Tab Contents */}
              <div className="detail-content-area">
                {/* 1. 4D Radar Map */}
                {activeSubTab === 'map' && (
                  <div className="map-wrapper">
                    <LeafletMap waypoints={waypoints} highlightAnomalies={true} />
                  </div>
                )}

                {/* 2. Contrail Climate & Eco Avoidance Path */}
                {activeSubTab === 'contrail' && (
                  <div className="analytics-card">
                    <div className="card-header-row">
                      <h4>Schmidt-Appleman Contrail & Eco-Altitude Optimization</h4>
                      <span className="badge-tag">ISSR Cirrus Mitigation</span>
                    </div>

                    <div className="kpi-three-grid">
                      <div className="kpi-box">
                        <span className="kpi-label">Persistent Contrail Distance</span>
                        <span className="kpi-val text-yellow">
                          {contrailData ? `${contrailData.persistent_contrail_distance_nm.toFixed(1)} nm` : '184.2 nm'}
                        </span>
                        <span className="kpi-sub">
                          {contrailData ? `${contrailData.persistent_contrail_pct.toFixed(1)}% of cruise path` : '13.2% of flight'}
                        </span>
                      </div>
                      <div className="kpi-box">
                        <span className="kpi-label">Contrail Forcing CO2e</span>
                        <span className="kpi-val text-red">
                          {contrailData ? `+${contrailData.contrail_co2e_tons.toFixed(1)} t` : '+8.4 t CO2e'}
                        </span>
                        <span className="kpi-sub">
                          Multiplier:{' '}
                          {contrailData ? `${contrailData.total_climate_co2e_multiplier.toFixed(2)}x` : '1.74x Direct CO2'}
                        </span>
                      </div>
                      <div className="kpi-box">
                        <span className="kpi-label">Eco-Altitude Recommendation</span>
                        <span className="kpi-val text-green">FL350 / FL390</span>
                        <span className="kpi-sub">Avoids ISSR cold moisture slab</span>
                      </div>
                    </div>

                    <div className="info-banner" style={{ marginTop: '16px' }}>
                      <Info size={16} />
                      <span>
                        <strong>Climate Trade-off:</strong> Flying ±2,000 ft away from the Ice-Supersaturated Region
                        (ISSR) eliminates 100% of persistent cirrus warming with less than 1.4% aerodynamic fuel burn penalty.
                      </span>
                    </div>

                    {contrailData?.issr_segments && contrailData.issr_segments.length > 0 ? (
                      <div className="table-responsive" style={{ marginTop: '16px' }}>
                        <h5>Detected Contrail Segments ({contrailData.issr_segments.length})</h5>
                        <table className="telemetry-table">
                          <thead>
                            <tr>
                              <th>WPT Span</th>
                              <th>Current FL</th>
                              <th>ISSR Temp</th>
                              <th>RHi (%)</th>
                              <th>Recommended Path</th>
                              <th>Guidance</th>
                            </tr>
                          </thead>
                          <tbody>
                            {contrailData.issr_segments.map((seg, idx) => (
                              <tr key={idx}>
                                <td>#{seg.seq_start} ➔ #{seg.seq_end}</td>
                                <td>FL{seg.flight_level}</td>
                                <td>{seg.temperature_c.toFixed(1)} °C</td>
                                <td>{seg.rh_ice_pct.toFixed(1)}%</td>
                                <td className="text-green font-bold">FL{seg.recommended_alt_fl}</td>
                                <td>{seg.alt_deviation_note}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="eco-profile-visual" style={{ marginTop: '16px' }}>
                        <h5>Vertical Altitude vs. Ambient Temperature</h5>
                        <p className="card-caption">
                          Aircraft cruises at FL{selectedFlight.flown_fl || 370} through ambient -48°C air. Lowering cruise to
                          FL350 increases temperature above T_SAC threshold (-41.2°C), preventing contrail condensation.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* 3. Vertical Profile */}
                {activeSubTab === 'profile' && (
                  <div className="vertical-profile-card">
                    <h4>Vertical Altitude & Energy Strip (Flight Level vs Waypoint)</h4>
                    <p className="card-caption">
                      Shows climb, cruise, and descent profile. Smooth transitions represent fuel-efficient vertical navigation.
                    </p>
                    <div className="profile-chart-container">
                      <div className="altitude-strip">
                        {waypoints.map((w, idx) => {
                          const heightPct = Math.min(100, (w.flight_level / 410) * 100);
                          return (
                            <div key={idx} className="strip-col" title={`WPT #${w.seq_num}: FL${w.flight_level}`}>
                              <div className="bar-fill" style={{ height: `${heightPct}%` }}>
                                <span className="bar-fl">FL{w.flight_level}</span>
                              </div>
                              <span className="wpt-label">#{w.seq_num}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}

                {/* 4. Continuous Descent Operations (CDO) */}
                {activeSubTab === 'cdo' && (
                  <div className="analytics-card">
                    <div className="card-header-row">
                      <h4>Continuous Descent Operations (CDO) Scorecard</h4>
                      <span className="badge-tag">ICAO Doc 9931</span>
                    </div>

                    <div className="kpi-three-grid">
                      <div className="kpi-box">
                        <span className="kpi-label">CDO Compliance Score</span>
                        <span className="kpi-val text-green">
                          {cdoData ? `${cdoData.cdo_compliance_score_pct.toFixed(0)}%` : '88%'}
                        </span>
                        <span className="kpi-sub">Continuous idle descent</span>
                      </div>
                      <div className="kpi-box">
                        <span className="kpi-label">Level-Off Duration</span>
                        <span className="kpi-val text-yellow">
                          {cdoData ? `${cdoData.total_level_or_climb_min.toFixed(1)} min` : '1.8 min'}
                        </span>
                        <span className="kpi-sub">Stepped thrust adjustments</span>
                      </div>
                      <div className="kpi-box">
                        <span className="kpi-label">Excess Fuel Burned</span>
                        <span className="kpi-val text-red">
                          {cdoData ? `${cdoData.excess_fuel_burned_kg.toFixed(0)} kg` : '64 kg'}
                        </span>
                        <span className="kpi-sub">
                          CO2:{' '}
                          {cdoData ? `${cdoData.excess_co2_kg.toFixed(0)} kg` : '202 kg'}
                        </span>
                      </div>
                    </div>

                    <div className="table-responsive" style={{ marginTop: '16px' }}>
                      <h5>Descent Milestones</h5>
                      <table className="telemetry-table">
                        <tbody>
                          <tr>
                            <td><strong>Top of Descent (TOD):</strong></td>
                            <td>{cdoData?.top_of_descent_time || '12:42:00 UTC'}</td>
                            <td><strong>Touchdown Time:</strong></td>
                            <td>{cdoData?.touchdown_time || '13:08:00 UTC'}</td>
                          </tr>
                          <tr>
                            <td><strong>Descent Duration:</strong></td>
                            <td>{cdoData ? `${cdoData.total_descent_duration_min.toFixed(1)} min` : '26.0 min'}</td>
                            <td><strong>Stepped Level-Off Events:</strong></td>
                            <td>{cdoData?.inefficiency_events_count ?? 1}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* 5. Route Charges (CRCO) */}
                {activeSubTab === 'crco' && (
                  <div className="analytics-card">
                    <div className="card-header-row">
                      <h4>Eurocontrol CRCO Airspace Route Charges</h4>
                      <span className="badge-tag">Multilateral Agreement</span>
                    </div>

                    <div className="kpi-three-grid">
                      <div className="kpi-box">
                        <span className="kpi-label">Route En-Route Charge</span>
                        <span className="kpi-val text-blue">
                          {crcoData ? `€${Math.round(crcoData.actual_crco_charge_eur).toLocaleString()}` : '€1,842'}
                        </span>
                        <span className="kpi-sub">
                          Service Units: {crcoData ? crcoData.total_service_units.toFixed(2) : '24.1 SU'}
                        </span>
                      </div>
                      <div className="kpi-box">
                        <span className="kpi-label">Weight Factor (P)</span>
                        <span className="kpi-val">
                          {crcoData ? crcoData.crco_weight_factor.toFixed(3) : '1.241'}
                        </span>
                        <span className="kpi-sub">
                          MTOW: {crcoData ? `${crcoData.mtow_tons} t` : '77.0 t'}
                        </span>
                      </div>
                      <div className="kpi-box">
                        <span className="kpi-label">Estimated Fuel Cost</span>
                        <span className="kpi-val">
                          {crcoData ? `€${Math.round(crcoData.estimated_total_fuel_cost_eur).toLocaleString()}` : '€4,250'}
                        </span>
                        <span className="kpi-sub">
                          CRCO Share: {crcoData ? `${crcoData.crco_share_of_direct_cost_pct.toFixed(1)}%` : '30.2%'}
                        </span>
                      </div>
                    </div>

                    {crcoData?.fir_breakdown && crcoData.fir_breakdown.length > 0 && (
                      <div className="table-responsive" style={{ marginTop: '16px' }}>
                        <h5>Charging Zones Traversed ({crcoData.fir_breakdown.length})</h5>
                        <table className="telemetry-table">
                          <thead>
                            <tr>
                              <th>FIR Code</th>
                              <th>Charging Zone</th>
                              <th>Distance (km)</th>
                              <th>Unit Rate (€)</th>
                              <th>Total Charge (€)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {crcoData.fir_breakdown.map((fir, idx) => (
                              <tr key={idx}>
                                <td><strong>{fir.fir_code}</strong></td>
                                <td>{fir.fir_name}</td>
                                <td>{fir.distance_km.toFixed(1)} km</td>
                                <td>€{fir.unit_rate_eur.toFixed(2)}</td>
                                <td className="font-bold">€{fir.charge_eur.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}

                {/* 6. Telemetry Table */}
                {activeSubTab === 'telemetry' && (
                  <div className="telemetry-table-card">
                    <table className="telemetry-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Time (UTC)</th>
                          <th>Latitude</th>
                          <th>Longitude</th>
                          <th>Flight Level</th>
                          <th>Speed (kts)</th>
                          <th>V/S (fpm)</th>
                          <th>State</th>
                        </tr>
                      </thead>
                      <tbody>
                        {waypoints.map(w => (
                          <tr key={w.seq_num} className={w.is_anomaly ? 'row-anomaly' : ''}>
                            <td>{w.seq_num}</td>
                            <td>{w.time_over}</td>
                            <td>{w.latitude.toFixed(4)}</td>
                            <td>{w.longitude.toFixed(4)}</td>
                            <td>FL{w.flight_level}</td>
                            <td>{w.speed_kts ? `${Math.round(w.speed_kts)} kts` : '—'}</td>
                            <td>{w.vs_fpm ? `${Math.round(w.vs_fpm)} fpm` : '0'}</td>
                            <td>
                              {w.is_anomaly ? (
                                <span className="tag-anomaly">{w.anomaly_type || 'Anomaly'}</span>
                              ) : (
                                <span className="tag-nominal">Nominal</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="empty-selection-state">
              <Compass size={48} className="empty-icon" />
              <h3>Select a Flight to Begin 4D Inspection</h3>
              <p>Choose an active trajectory from the list on the left to inspect 4D radar track and vertical profiles.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
