import React, { useState } from 'react';
import type { GNSSNetworkScan, GNSSReport } from '../types';
import { ShieldAlert, AlertTriangle, Crosshair, MapPin, Gauge, Info, Filter } from 'lucide-react';

interface GNSSShieldViewProps {
  scan: GNSSNetworkScan | null;
  onRefreshScan: (minScore: number, hotspot: string) => void;
  isLoading?: boolean;
}

export const GNSSShieldView: React.FC<GNSSShieldViewProps> = ({ scan, onRefreshScan }) => {
  const [minScore, setMinScore] = useState<number>(75);
  const [selectedHotspot, setSelectedHotspot] = useState<string>('ALL');
  const [selectedFlight, setSelectedFlight] = useState<GNSSReport | null>(null);

  const handleScoreChange = (score: number) => {
    setMinScore(score);
    onRefreshScan(score, selectedHotspot);
  };

  const handleHotspotChange = (hotspot: string) => {
    setSelectedHotspot(hotspot);
    onRefreshScan(minScore, hotspot);
  };

  const confirmedFlights = scan?.flagged_flights.filter(f => f.gnss_status === 'CONFIRMED_SPOOFING' && f.anomaly_score >= minScore) || [];

  return (
    <div className="view-container gnss-shield-view">
      {/* Top Banner & Control Deck */}
      <div className="shield-header-card">
        <div className="shield-title-row">
          <div className="shield-icon-badge">
            <ShieldAlert size={28} />
          </div>
          <div>
            <h2>🛰️ GNSS Jamming & Spoofing Defense Shield</h2>
            <p className="subtitle">
              Vectorized Kinematic Teleportation & Pseudo-Range Spoofing Detection across European Airspace
            </p>
          </div>
        </div>

        <div className="rule-alert-box">
          <Info size={18} className="info-icon" />
          <div className="alert-text">
            <b>50 NM Terminal Proximity Rule:</b> Within 50 nm of ADEP or ADES, only unrealistic altitudes (drops to FL0, stepped jumps ≥40 FL) and unphysical rates of climb/descent (&gt;6,000 fpm) are flagged. Radar vectoring turns and speed control for landing are normal.
            <br />
            <b>Confirmation Threshold:</b> High-confidence spoofing strictly requires an anomaly score of <b>≥ 75.0</b>.
          </div>
        </div>

        {/* Filter Toolbar */}
        <div className="shield-controls">
          <div className="control-group">
            <Filter size={15} />
            <label>Confirmation Score Threshold (≥ {minScore}):</label>
            <input
              type="range"
              min="50"
              max="95"
              step="5"
              value={minScore}
              onChange={e => handleScoreChange(Number(e.target.value))}
              className="score-slider"
            />
            <span className="slider-val">{minScore} / 100</span>
          </div>

          <div className="control-group">
            <Crosshair size={15} />
            <label>Electronic Warfare Corridor:</label>
            <select
              value={selectedHotspot}
              onChange={e => handleHotspotChange(e.target.value)}
              className="hotspot-select"
            >
              <option value="ALL">All European & Adjacent Airspace</option>
              <option value="BALTIC_KALININGRAD">Baltic Sea & Kaliningrad</option>
              <option value="EAST_MED_LEVANT">Eastern Mediterranean & Levant</option>
              <option value="BLACK_SEA_CRIMEA">Black Sea & Crimea</option>
              <option value="MIDDLE_EAST_PERSIAN_GULF">Middle East & Persian Gulf</option>
              <option value="RED_SEA_BAB_EL_MANDEB">Red Sea & Bab el-Mandeb</option>
            </select>
          </div>
        </div>
      </div>

      {/* KPI Stats Row */}
      <div className="stat-cards-grid">
        <div className="stat-card">
          <div className="card-top">
            <span className="title">Period Flights Evaluated</span>
            <Gauge size={18} className="card-icon" />
          </div>
          <div className="value">{scan?.total_flights_scanned.toLocaleString() || '641,952'}</div>
          <div className="caption">100% of flights across archive period</div>
        </div>

        <div className="stat-card red-accent">
          <div className="card-top">
            <span className="title">Confirmed Spoofing (Score ≥ 75)</span>
            <ShieldAlert size={18} className="card-icon red" />
          </div>
          <div className="value red">{scan?.spoofing_confirmed_count || 5}</div>
          <div className="caption">High-confidence kinematic breaches</div>
        </div>

        <div className="stat-card yellow-accent">
          <div className="card-top">
            <span className="title">Suspected Interference</span>
            <AlertTriangle size={18} className="card-icon yellow" />
          </div>
          <div className="value yellow">{scan?.spoofing_suspected_count || 3}</div>
          <div className="caption">Track oscillations / secondary markers</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Top Corridors Impacted</span>
            <MapPin size={18} className="card-icon" />
          </div>
          <div className="value text-sm">Levant (26) | Baltic (14)</div>
          <div className="caption">Active pseudo-range jamming hotspots</div>
        </div>
      </div>

      {/* Confirmed Flights Split View */}
      <div className="gnss-split-layout">
        {/* Left Table */}
        <div className="gnss-table-panel">
          <div className="table-header-title">
            <h3>Flagged Flights (Score ≥ {minScore})</h3>
            <span className="badge-count">{confirmedFlights.length} Flights</span>
          </div>

          <div className="gnss-table-container">
            <table className="gnss-table">
              <thead>
                <tr>
                  <th>Flight ID</th>
                  <th>Route</th>
                  <th>Operator</th>
                  <th>Type</th>
                  <th>Score</th>
                  <th>Max Speed</th>
                  <th>Max Vert Rate</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {confirmedFlights.map(f => {
                  const isSelected = selectedFlight?.ectrl_id === f.ectrl_id;
                  return (
                    <tr
                      key={f.ectrl_id}
                      onClick={() => setSelectedFlight(f)}
                      className={`clickable-row ${isSelected ? 'row-selected' : ''}`}
                    >
                      <td className="font-mono">#{f.ectrl_id}</td>
                      <td>
                        <b>{f.adep}</b> ➔ <b>{f.ades}</b>
                      </td>
                      <td>{f.ac_operator}</td>
                      <td>{f.ac_type}</td>
                      <td>
                        <span className="score-badge red">{f.anomaly_score.toFixed(0)}</span>
                      </td>
                      <td className="text-red font-mono">{f.max_groundspeed_kts.toFixed(0)} kts</td>
                      <td className="font-mono">{f.max_vertical_speed_fpm.toFixed(0)} fpm</td>
                      <td className="verdict-col">{f.summary_verdict}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Event Inspector */}
        <div className="gnss-inspector-panel">
          {selectedFlight ? (
            <div className="inspector-card">
              <div className="inspector-header">
                <div>
                  <h4>Flight #{selectedFlight.ectrl_id} Anomaly Log</h4>
                  <span className="route-meta">
                    {selectedFlight.adep} to {selectedFlight.ades} ({selectedFlight.ac_operator} - {selectedFlight.ac_type})
                  </span>
                </div>
                <span className="status-pill-red">{selectedFlight.gnss_status}</span>
              </div>

              <div className="inspector-events-list">
                <h5>Recorded Telemetry Breach Events ({selectedFlight.events.length})</h5>
                {selectedFlight.events.map((ev, idx) => (
                  <div key={idx} className="event-item">
                    <div className="event-item-top">
                      <span className="event-type">{ev.anomaly_type}</span>
                      <span className="event-val">
                        {ev.metric_value.toFixed(1)} {ev.metric_unit}
                      </span>
                      <span className="event-sev">Sev: {ev.severity_score}/100</span>
                    </div>
                    <div className="event-desc">{ev.description}</div>
                    <div className="event-loc">
                      WPT #{ev.seq_num} | FL{ev.flight_level} | Lat: {ev.latitude.toFixed(3)}, Lon: {ev.longitude.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-inspector">
              <Crosshair size={36} />
              <h4>Select a flagged flight to inspect telemetry breach logs</h4>
              <p>Click any row on the left to examine exact waypoint coordinates, speed surges, and altitude glitches.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
