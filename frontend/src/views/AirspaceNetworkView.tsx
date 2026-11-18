import React from 'react';
import { Globe2, Activity, Clock, TrendingUp, AlertOctagon } from 'lucide-react';

export const AirspaceNetworkView: React.FC = () => {
  const topCongestedFirs = [
    { fir: 'EGTT (London FIR)', capacity: 1420, current_occ: 1380, load_pct: 97.2, avg_delay_min: 18.4, status: 'CRITICAL' },
    { fir: 'EDGG (Frankfurt FIR)', capacity: 1350, current_occ: 1290, load_pct: 95.5, avg_delay_min: 15.2, status: 'HIGH' },
    { fir: 'LFFF (Paris FIR)', capacity: 1510, current_occ: 1395, load_pct: 92.4, avg_delay_min: 12.1, status: 'HIGH' },
    { fir: 'LECM (Madrid FIR)', capacity: 1100, current_occ: 980, load_pct: 89.1, avg_delay_min: 8.6, status: 'MODERATE' },
    { fir: 'LSAS (Zurich FIR)', capacity: 850, current_occ: 730, load_pct: 85.8, avg_delay_min: 6.2, status: 'MODERATE' },
  ];

  return (
    <div className="view-container airspace-network-view">
      <div className="view-header">
        <div className="header-icon-box">
          <Globe2 size={26} />
        </div>
        <div>
          <h2>🌐 Airspace Network Status & Sector Saturation</h2>
          <p className="subtitle">
            Macro Capacity Utilization, En-Route Sector Bottlenecks, and Tactical ATC Inefficiencies
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="stat-cards-grid">
        <div className="stat-card">
          <div className="card-top">
            <span className="title">Network On-Time Rate (OTP 15)</span>
            <Activity size={18} className="card-icon text-green" />
          </div>
          <div className="value text-green">81.4%</div>
          <div className="caption">Target: ≥ 80.0% GANP Standard</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Mean En-Route ATFM Delay</span>
            <Clock size={18} className="card-icon" />
          </div>
          <div className="value">11.8 min</div>
          <div className="caption">+1.4 min vs previous monthly baseline</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Sectors Exceeding Capacity</span>
            <AlertOctagon size={18} className="card-icon text-red" />
          </div>
          <div className="value text-red">14 Sectors</div>
          <div className="caption">London, Frankfurt, Paris TMA peaks</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Free Route Directness (KEA)</span>
            <TrendingUp size={18} className="card-icon" />
          </div>
          <div className="value">2.68%</div>
          <div className="caption">Eurocontrol PRU Target: &lt; 2.60%</div>
        </div>
      </div>

      {/* Sector Congestion Table */}
      <div className="content-card">
        <div className="card-header-row">
          <h3>En-Route Flight Information Region (FIR) Peak Occupancy</h3>
          <span className="badge-pill">Real-Time Load Factors</span>
        </div>
        <div className="table-responsive">
          <table className="network-table">
            <thead>
              <tr>
                <th>Airspace Sector</th>
                <th>Hourly Capacity</th>
                <th>Active Traffic</th>
                <th>Saturation %</th>
                <th>Mean Delay</th>
                <th>Operational Status</th>
              </tr>
            </thead>
            <tbody>
              {topCongestedFirs.map(fir => (
                <tr key={fir.fir}>
                  <td><b>{fir.fir}</b></td>
                  <td>{fir.capacity.toLocaleString()} fl/hr</td>
                  <td>{fir.current_occ.toLocaleString()} fl/hr</td>
                  <td>
                    <div className="progress-bar-wrapper">
                      <div
                        className={`progress-fill ${fir.load_pct > 95 ? 'red' : fir.load_pct > 90 ? 'yellow' : 'green'}`}
                        style={{ width: `${fir.load_pct}%` }}
                      />
                      <span className="progress-text">{fir.load_pct.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td>{fir.avg_delay_min} min</td>
                  <td>
                    <span className={`status-badge-inline ${fir.status.toLowerCase()}`}>{fir.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
