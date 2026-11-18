import React, { useState } from 'react';
import type { FlightSummary, Waypoint } from '../types';
import { LeafletMap } from '../components/LeafletMap';
import { Plane, AlertTriangle, ArrowRight, Gauge, Clock, MapPin, Compass } from 'lucide-react';

interface FlightInspectorViewProps {
  flights: FlightSummary[];
  selectedFlight: FlightSummary | null;
  onSelectFlight: (f: FlightSummary) => void;
  waypoints: Waypoint[];
  isLoading?: boolean;
}

export const FlightInspectorView: React.FC<FlightInspectorViewProps> = ({
  flights,
  selectedFlight,
  onSelectFlight,
  waypoints,
}) => {
  const [activeInspectorSubTab, setActiveInspectorSubTab] = useState<'map' | 'profile' | 'telemetry'>('map');

  return (
    <div className="view-container flight-inspector">
      <div className="flight-master-detail">
        {/* Left: Flight Roster List */}
        <div className="flight-list-panel">
          <div className="panel-header">
            <h3>Active Flights ({flights.length})</h3>
            <span className="panel-hint">Select a flight to inspect 4D telemetry</span>
          </div>

          <div className="flight-items-scroll">
            {flights.map(f => {
              const isSelected = selectedFlight?.ectrl_id === f.ectrl_id;
              const hasAnomaly = f.gnss_status === 'CONFIRMED_SPOOFING';
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
                    {hasAnomaly && (
                      <span className="anomaly-pill">
                        <AlertTriangle size={11} /> Spoofed
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Detailed 4D Inspector */}
        <div className="flight-detail-panel">
          {selectedFlight ? (
            <>
              {/* Executive Flight Header */}
              <div className="detail-hero-card">
                <div className="hero-main">
                  <div className="hero-icon">
                    <Plane size={28} />
                  </div>
                  <div>
                    <div className="hero-title">
                      {selectedFlight.callsign}
                      <span className="hero-id">ID: #{selectedFlight.ectrl_id}</span>
                    </div>
                    <div className="hero-sub">
                      Operator: <b>{selectedFlight.ac_operator}</b> | Aircraft: <b>{selectedFlight.ac_type}</b> ({selectedFlight.ac_reg})
                    </div>
                  </div>
                </div>

                <div className="hero-kpis">
                  <div className="hero-stat">
                    <span className="stat-label">Cruise Level</span>
                    <span className="stat-val">FL{selectedFlight.flown_fl}</span>
                  </div>
                  <div className="hero-stat">
                    <span className="stat-label">Flown Distance</span>
                    <span className="stat-val">{selectedFlight.actual_distance_nm} nm</span>
                  </div>
                  <div className="hero-stat">
                    <span className="stat-label">Arr Delay</span>
                    <span className={`stat-val ${selectedFlight.arr_delay_min && selectedFlight.arr_delay_min > 15 ? 'text-red' : 'text-green'}`}>
                      {selectedFlight.arr_delay_min ? `${selectedFlight.arr_delay_min > 0 ? '+' : ''}${selectedFlight.arr_delay_min} min` : '0 min'}
                    </span>
                  </div>
                  <div className="hero-stat">
                    <span className="stat-label">GNSS State</span>
                    <span className={`stat-val ${selectedFlight.gnss_status === 'CONFIRMED_SPOOFING' ? 'text-red' : 'text-green'}`}>
                      {selectedFlight.gnss_status || 'NOMINAL'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Sub-view switcher */}
              <div className="detail-tabs">
                <button
                  className={`detail-tab ${activeInspectorSubTab === 'map' ? 'active' : ''}`}
                  onClick={() => setActiveInspectorSubTab('map')}
                >
                  <MapPin size={15} /> 4D Trajectory Map
                </button>
                <button
                  className={`detail-tab ${activeInspectorSubTab === 'profile' ? 'active' : ''}`}
                  onClick={() => setActiveInspectorSubTab('profile')}
                >
                  <Gauge size={15} /> Vertical Profile (CDO/CCO)
                </button>
                <button
                  className={`detail-tab ${activeInspectorSubTab === 'telemetry' ? 'active' : ''}`}
                  onClick={() => setActiveInspectorSubTab('telemetry')}
                >
                  <Clock size={15} /> Telemetry Log ({waypoints.length} WPTs)
                </button>
              </div>

              {/* Inspector Content */}
              <div className="detail-content-area">
                {activeInspectorSubTab === 'map' && (
                  <div className="map-wrapper">
                    <LeafletMap waypoints={waypoints} highlightAnomalies={true} />
                  </div>
                )}

                {activeInspectorSubTab === 'profile' && (
                  <div className="vertical-profile-card">
                    <h4>Vertical Altitude & Energy Profile (FL vs Waypoints)</h4>
                    <p className="card-caption">
                      Shows continuous descent operations (CDO) and climb profiles without step level-offs.
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

                {activeInspectorSubTab === 'telemetry' && (
                  <div className="telemetry-table-card">
                    <table className="telemetry-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Time (UTC)</th>
                          <th>Latitude</th>
                          <th>Longitude</th>
                          <th>Flight Level</th>
                          <th>Est. Speed</th>
                          <th>Vertical Rate</th>
                          <th>Status</th>
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
