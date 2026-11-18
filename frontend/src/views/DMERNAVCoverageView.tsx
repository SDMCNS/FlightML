import React, { useState, useEffect } from 'react';
import {
  Radio,
  Layers,
  ShieldCheck,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  TrendingDown,
} from 'lucide-react';
import { fetchDMERNAVCoverage } from '../services/api';
import type { DMERNAVCoverageReport } from '../types';

export const DMERNAVCoverageView: React.FC = () => {
  const [flightLevel, setFlightLevel] = useState<number>(300);
  const [inactiveStations, setInactiveStations] = useState<string[]>([]);
  const [report, setReport] = useState<DMERNAVCoverageReport | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const loadCoverage = async (fl: number, inactive: string[]) => {
    setIsLoading(true);
    try {
      const data = await fetchDMERNAVCoverage(fl, inactive);
      setReport(data);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadCoverage(flightLevel, inactiveStations);
  }, [flightLevel, inactiveStations]);

  const handleToggleStation = (ident: string) => {
    setInactiveStations(prev =>
      prev.includes(ident) ? prev.filter(x => x !== ident) : [...prev, ident]
    );
  };

  const handleDeactivateAllSuggested = () => {
    if (!report || !report.suggested_deactivations) return;
    const suggestedIdents = report.suggested_deactivations.map(s => s.ident);
    setInactiveStations(prev => Array.from(new Set([...prev, ...suggestedIdents])));
  };

  const handleResetStations = () => {
    setInactiveStations([]);
  };

  return (
    <div className="dme-rnav-view" style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
            <Radio size={24} color="#3b82f6" />
            DME-DME RNAV Backup Coverage & Navaid Rationalization Lab
          </h2>
          <p style={{ color: 'var(--text-muted)', margin: '4px 0 0 0', fontSize: '13px' }}>
            ICAO Doc 9613 line-of-sight and dual-station geometry (30° ≤ Δθ ≤ 150°) evaluation across 10×10 NM cells.
          </p>
        </div>

        {/* Flight Level Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Flight Level:</span>
          {[100, 200, 300, 390].map(fl => (
            <button
              key={fl}
              onClick={() => setFlightLevel(fl)}
              className={`pill-btn ${flightLevel === fl ? 'active' : ''}`}
              style={{
                padding: '6px 12px',
                borderRadius: '4px',
                fontSize: '12px',
                background: flightLevel === fl ? 'var(--accent-blue)' : 'var(--bg-card)',
                color: '#fff',
                border: '1px solid var(--bg-border)',
                cursor: 'pointer',
              }}
            >
              FL{fl}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Evaluating terrestrial radio horizon & DME pairs...</div>
      ) : report ? (
        <>
          {/* Top KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '6px' }}>
                <ShieldCheck size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>DME-DME Coverage</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{report.coverage_percentage}%</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {report.covered_cells_count} of {report.total_grid_cells} grid cells covered
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3b82f6', marginBottom: '6px' }}>
                <Layers size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Redundant Availability</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{report.redundant_cells_percentage}%</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                ≥ 3 DMEs satisfying 30°–150° angle
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', marginBottom: '6px' }}>
                <Radio size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Active Terrestrial DMEs</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{report.active_dmes_count} / {report.total_dmes_count}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {report.inactive_dmes_count} stations currently disabled
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#8b5cf6', marginBottom: '6px' }}>
                <TrendingDown size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Deactivation Potential</span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{report.suggested_deactivations.length} Stations</div>
              <div style={{ fontSize: '11px', color: '#a78bfa', marginTop: '4px' }}>
                Zero impact on 98%+ coverage
              </div>
            </div>
          </div>

          {/* Rationalization & Decommissioning Recommender Panel */}
          <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <h3 style={{ margin: 0, fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles size={18} color="#a855f7" />
                  Greedy Navaid Rationalization Optimizer (Decommissioning Recommender)
                </h3>
                <p style={{ margin: '4px 0 0 0', color: 'var(--text-muted)', fontSize: '12px' }}>
                  Identifies redundant terrestrial DMEs whose removal preserves at least 98.0% PBN RNAV 1 navigation coverage.
                </p>
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={handleDeactivateAllSuggested}
                  className="pill-btn"
                  style={{ background: '#7c3aed', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
                >
                  Deactivate All Suggested ({report.suggested_deactivations.length})
                </button>
                {inactiveStations.length > 0 && (
                  <button
                    onClick={handleResetStations}
                    className="pill-btn"
                    style={{ background: '#334155', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
                  >
                    Reset All to Active
                  </button>
                )}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
              {report.suggested_deactivations.slice(0, 9).map(sug => {
                const isDeactivated = inactiveStations.includes(sug.ident);
                return (
                  <div
                    key={sug.ident}
                    style={{
                      padding: '12px',
                      borderRadius: '6px',
                      background: isDeactivated ? 'rgba(239, 68, 68, 0.1)' : 'rgba(255, 255, 255, 0.03)',
                      border: isDeactivated ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid var(--bg-border)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 'bold', fontSize: '14px' }}>
                        {sug.ident} ({sug.country})
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{sug.name}</div>
                      <div style={{ fontSize: '11px', color: sug.coverage_impact_pct === 0 ? '#10b981' : '#f59e0b', marginTop: '2px' }}>
                        Impact: {sug.coverage_impact_pct}% • Savings: €{sug.annual_maintenance_savings_eur.toLocaleString()}/yr
                      </div>
                    </div>

                    <button
                      onClick={() => handleToggleStation(sug.ident)}
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: isDeactivated ? '#ef4444' : '#10b981' }}
                    >
                      {isDeactivated ? <ToggleLeft size={24} /> : <ToggleRight size={24} />}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Interactive Stations & Cells Grid Status */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            {/* 10x10 NM Grid Cells Breakdown */}
            <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <h4 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>10×10 NM Spatial Grid Status (FL{report.flight_level})</h4>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '14px' }}>
                Cells classified by line-of-sight DME availability and angular geometric intersection (30°–150°).
              </p>

              <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
                <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#10b981' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#10b981', display: 'inline-block' }}></span>
                  Redundant (≥3 pairs): {report.redundant_cells_percentage}%
                </span>
                <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#3b82f6' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#3b82f6', display: 'inline-block' }}></span>
                  Adequate (2 DMEs)
                </span>
                <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#ef4444' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#ef4444', display: 'inline-block' }}></span>
                  Outage / Blindspot: {report.outage_cells_percentage}%
                </span>
              </div>

              {/* Grid cell matrix visualization snippet */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '4px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
                {report.cells_summary.slice(0, 128).map((c, idx) => {
                  const bg =
                    c.coverage_status === 'REDUNDANT'
                      ? '#10b981'
                      : c.coverage_status === 'ADEQUATE'
                      ? '#3b82f6'
                      : c.coverage_status === 'MARGINAL'
                      ? '#f59e0b'
                      : '#ef4444';
                  return (
                    <div
                      key={idx}
                      title={`Cell (${c.lat}°N, ${c.lon}°E): ${c.coverage_status} (${c.visible_dme_count} DMEs visible, Best angle: ${c.best_pair_geometry_deg}°)`}
                      style={{
                        height: '14px',
                        borderRadius: '2px',
                        background: bg,
                        opacity: 0.85,
                      }}
                    />
                  );
                })}
              </div>
            </div>

            {/* Terrestrial DME Station List with Manual Toggles */}
            <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', maxHeight: '420px', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h4 style={{ margin: 0, fontSize: '14px' }}>Terrestrial DME Stations Inventory</h4>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Click toggle to simulate outage</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {report.dme_stations.slice(0, 40).map(sta => {
                  const isDeactivated = inactiveStations.includes(sta.ident);
                  return (
                    <div
                      key={sta.ident}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '6px 10px',
                        borderRadius: '4px',
                        background: 'rgba(255, 255, 255, 0.02)',
                        border: '1px solid var(--bg-border)',
                        fontSize: '12px',
                      }}
                    >
                      <div>
                        <strong>{sta.ident}</strong> - {sta.name} ({sta.country}) • {sta.elevation_ft} ft MSL
                      </div>
                      <button
                        onClick={() => handleToggleStation(sta.ident)}
                        style={{
                          background: isDeactivated ? '#ef4444' : '#10b981',
                          color: '#fff',
                          border: 'none',
                          padding: '2px 8px',
                          borderRadius: '3px',
                          fontSize: '10px',
                          cursor: 'pointer',
                        }}
                      >
                        {isDeactivated ? 'INACTIVE' : 'ACTIVE'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
