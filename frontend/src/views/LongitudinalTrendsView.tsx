import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Gauge,
  Fuel,
  Leaf,
  Layers,
  Route,
  AlertTriangle,
  Award,
} from 'lucide-react';
import { fetchVerticalCapping, fetchAirwayFRADividend } from '../services/api';
import type { VerticalCappingReport, AirwayFRADividendReport } from '../types';

interface LongitudinalTrendsViewProps {
  period?: string;
}

export const LongitudinalTrendsView: React.FC<LongitudinalTrendsViewProps> = ({ period = '202402' }) => {
  const [activeSubTab, setActiveSubTab] = useState<'fl-capping' | 'airway-fra'>('fl-capping');
  const [cappingReport, setCappingReport] = useState<VerticalCappingReport | null>(null);
  const [fraReport, setFRAReport] = useState<AirwayFRADividendReport | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      try {
        const [cap, fra] = await Promise.all([
          fetchVerticalCapping(period, 2500),
          fetchAirwayFRADividend(period, 2500),
        ]);
        setCappingReport(cap);
        setFRAReport(fra);
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, [period]);

  return (
    <div className="longitudinal-trends-view" style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
            <TrendingUp size={24} color="#3b82f6" />
            Longitudinal Airspace Modernization & Multi-Period Benchmarking
          </h2>
          <p style={{ color: 'var(--text-muted)', margin: '4px 0 0 0', fontSize: '13px' }}>
            Vertical flight level capping penalties, market segment stratification, and Free Route Airspace (FRA) environmental dividends.
          </p>
        </div>

        {/* Subtab Switcher */}
        <div style={{ display: 'flex', gap: '8px', background: 'var(--bg-card)', padding: '4px', borderRadius: '6px', border: '1px solid var(--bg-border)' }}>
          <button
            onClick={() => setActiveSubTab('fl-capping')}
            className={`tab-btn ${activeSubTab === 'fl-capping' ? 'active' : ''}`}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              fontSize: '13px',
              fontWeight: 500,
              background: activeSubTab === 'fl-capping' ? 'var(--accent-blue)' : 'transparent',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Gauge size={16} /> Vertical Capping (Use Case 6)
          </button>
          <button
            onClick={() => setActiveSubTab('airway-fra')}
            className={`tab-btn ${activeSubTab === 'airway-fra' ? 'active' : ''}`}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              fontSize: '13px',
              fontWeight: 500,
              background: activeSubTab === 'airway-fra' ? 'var(--accent-blue)' : 'transparent',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Route size={16} /> Airways vs FRA Dividend (Use Case 7)
          </button>
        </div>
      </div>

      {isLoading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Compiling longitudinal intelligence...</div>
      ) : activeSubTab === 'fl-capping' && cappingReport ? (
        <>
          {/* Top KPI Cards for Vertical Capping */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444', marginBottom: '6px' }}>
                <AlertTriangle size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Vertical Capping Rate</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{cappingReport.capping_rate_pct}%</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {cappingReport.capped_flights_count} of {cappingReport.sample_size} flights restricted below requested FL
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', marginBottom: '6px' }}>
                <Gauge size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Mean Altitude Deficit</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>FL{cappingReport.mean_deficit_fl}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                ~{(cappingReport.mean_deficit_fl * 100).toLocaleString()} ft below optimal cruise
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3b82f6', marginBottom: '6px' }}>
                <Fuel size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Fuel Burn Penalty</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>+{cappingReport.mean_fuel_penalty_pct}%</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {cappingReport.total_excess_fuel_tonnes} tonnes excess fuel consumed
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '6px' }}>
                <Leaf size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Avoidable CO₂ Tax</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{cappingReport.total_excess_co2_tonnes} t</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Direct atmospheric carbon penalty
              </div>
            </div>
          </div>

          {/* Market Segment Stratification */}
          <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', marginBottom: '20px' }}>
            <h3 style={{ margin: '0 0 14px 0', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} color="#3b82f6" />
              Vertical Restriction Sensitivity by Market Segment
            </h3>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '8px' }}>Market Segment</th>
                  <th style={{ padding: '8px' }}>Sampled Flights</th>
                  <th style={{ padding: '8px' }}>Capped Flights</th>
                  <th style={{ padding: '8px' }}>Capping Rate</th>
                  <th style={{ padding: '8px' }}>Mean Altitude Deficit</th>
                  <th style={{ padding: '8px' }}>Estimated Extra Fuel</th>
                </tr>
              </thead>
              <tbody>
                {cappingReport.by_market_segment.map(seg => (
                  <tr key={seg.market_segment} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '10px 8px', fontWeight: 'bold' }}>{seg.market_segment}</td>
                    <td style={{ padding: '10px 8px' }}>{seg.total_flights.toLocaleString()}</td>
                    <td style={{ padding: '10px 8px' }}>{seg.capped_flights.toLocaleString()}</td>
                    <td style={{ padding: '10px 8px', color: seg.capping_rate_pct > 15 ? '#ef4444' : '#10b981' }}>
                      {seg.capping_rate_pct}%
                    </td>
                    <td style={{ padding: '10px 8px' }}>FL{seg.mean_deficit_fl} (~{(seg.mean_deficit_fl * 100).toLocaleString()} ft)</td>
                    <td style={{ padding: '10px 8px', color: '#f59e0b' }}>+{seg.estimated_penalty_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Top Capped Routes */}
          <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
            <h3 style={{ margin: '0 0 14px 0', fontSize: '16px' }}>Top Vertically Capped Route Corridors</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
              {cappingReport.top_capped_routes.map(r => (
                <div key={r.route} style={{ padding: '12px', borderRadius: '6px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--bg-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 'bold', fontSize: '14px' }}>{r.route}</div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{r.flights} flights analyzed</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ color: '#ef4444', fontWeight: 'bold', fontSize: '14px' }}>{r.capping_rate_pct}% Capped</div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Avg Deficit: FL{r.mean_deficit_fl}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : activeSubTab === 'airway-fra' && fraReport ? (
        <>
          {/* Top KPI Cards for Airways vs FRA */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '6px' }}>
                <Award size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Distance Reduction</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{fraReport.mean_distance_saved_pct}%</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Actual flown track vs legacy ATS airway route
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3b82f6', marginBottom: '6px' }}>
                <Route size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Track Miles Saved</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{fraReport.total_distance_saved_nm.toLocaleString()} nm</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Across {fraReport.sample_size.toLocaleString()} sampled flights
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', marginBottom: '6px' }}>
                <Fuel size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Fuel Saved</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{fraReport.total_fuel_saved_tonnes} t</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Tonnes Jet A-1 conserved via Direct-To routing
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '6px' }}>
                <Leaf size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>CO₂ Emissions Avoided</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{fraReport.total_co2_avoided_tonnes} t</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Environmental dividend of SESAR / FRA
              </div>
            </div>
          </div>

          {/* Top Route Beneficiaries */}
          <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
            <h3 style={{ margin: '0 0 14px 0', fontSize: '16px' }}>Top Route Beneficiaries of Free Route Airspace</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '8px' }}>Route</th>
                  <th style={{ padding: '8px' }}>Flights</th>
                  <th style={{ padding: '8px' }}>Mean Miles Saved</th>
                  <th style={{ padding: '8px' }}>Distance Saved %</th>
                  <th style={{ padding: '8px' }}>Fuel Saved</th>
                  <th style={{ padding: '8px' }}>CO₂ Avoided</th>
                </tr>
              </thead>
              <tbody>
                {fraReport.top_beneficiary_routes.map(r => (
                  <tr key={r.route} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '10px 8px', fontWeight: 'bold' }}>{r.route}</td>
                    <td style={{ padding: '10px 8px' }}>{r.flights}</td>
                    <td style={{ padding: '10px 8px', color: '#10b981' }}>-{r.mean_saved_nm} nm</td>
                    <td style={{ padding: '10px 8px', color: '#10b981' }}>{r.mean_saved_pct}%</td>
                    <td style={{ padding: '10px 8px' }}>{r.total_fuel_saved_tonnes} t</td>
                    <td style={{ padding: '10px 8px', fontWeight: 'bold', color: '#10b981' }}>{r.total_co2_avoided_tonnes} t</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
};
