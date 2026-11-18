import React from 'react';
import { Leaf, CloudRain, Wind, CheckCircle, AlertCircle } from 'lucide-react';

export const ClimateEcoView: React.FC = () => {
  return (
    <div className="view-container climate-eco-view">
      <div className="view-header">
        <div className="header-icon-box eco">
          <Leaf size={26} />
        </div>
        <div>
          <h2>🌿 Contrail Microphysics, Upper-Air Winds & Eco Route Optimizer</h2>
          <p className="subtitle">
            Schmidt-Appleman Criterion (T_SAC), Ice Supersaturated Regions (ISSR), and Radiative Forcing Mitigation
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="stat-cards-grid">
        <div className="stat-card">
          <div className="card-top">
            <span className="title">Mean Cruise Temperature</span>
            <Wind size={18} className="card-icon" />
          </div>
          <div className="value">-53.5 °C</div>
          <div className="caption">At FL360 (Standard ISA Lapse Rate)</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Schmidt-Appleman Threshold (T_SAC)</span>
            <CloudRain size={18} className="card-icon text-blue" />
          </div>
          <div className="value text-blue">-43.2 °C</div>
          <div className="caption">Exhaust plume mixes to water saturation</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Persistent Cirrus Formed</span>
            <AlertCircle size={18} className="card-icon text-yellow" />
          </div>
          <div className="value text-yellow">18.4% of Route</div>
          <div className="caption">Ice Supersaturated (RHi &ge; 100%)</div>
        </div>

        <div className="stat-card">
          <div className="card-top">
            <span className="title">Minimal Contrail Altitude Shift</span>
            <CheckCircle size={18} className="card-icon text-green" />
          </div>
          <div className="value text-green">-2,000 ft (FL340)</div>
          <div className="caption">Avoids ISSR with only +0.8% Jet-A penalty</div>
        </div>
      </div>

      {/* Thermodynamic Formulation Cards */}
      <div className="eco-cards-grid">
        <div className="content-card">
          <h3>Schmidt-Appleman Criterion (T_SAC)</h3>
          <p className="card-caption">
            Determines whether the mixing of warm, moist engine exhaust with ambient air produces liquid water droplets that instantly freeze into contrail ice crystals.
          </p>
          <div className="formula-box">
            <code>G = (EI_H2O · C_p · P) / (ε · Q · (1 - η))</code>
            <br />
            <code>T_SAC = 226.69 + 10.54·ln(G - 0.05) + 0.7777·ln²(G - 0.05)</code>
          </div>
          <ul className="eco-list">
            <li><b>Contrail Forms:</b> When ambient temperature T_amb &le; T_SAC.</li>
            <li><b>Persistent Contrail Cirrus:</b> When T_amb &le; T_SAC and relative humidity over ice RHi &ge; 100%.</li>
          </ul>
        </div>

        <div className="content-card">
          <h3>Eco-Trajectory Trade-Off: Fuel Burn vs. Radiative Forcing</h3>
          <p className="card-caption">
            Tactical vertical avoidance of ice-supersaturated layers reduces global warming impact by up to 70% while keeping fuel burn penalties under 1.5%.
          </p>
          <div className="tradeoff-card">
            <div className="tradeoff-row">
              <span className="row-label">Optimal Aerodynamic Cruise:</span>
              <span className="row-val">FL370 (Lowest CO₂ / Jet-A burn)</span>
            </div>
            <div className="tradeoff-row">
              <span className="row-label">ISSR Contrail Band:</span>
              <span className="row-val text-red">FL360 - FL380 (RHi = 118%)</span>
            </div>
            <div className="tradeoff-row highlight">
              <span className="row-label">Recommended Eco Route:</span>
              <span className="row-val text-green">Cruise at FL340 (Avoids Contrail, +22 kg Fuel)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
