import React, { useState, useEffect } from 'react';
import {
  Radio,
  Layers,
  ShieldCheck,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  TrendingDown,
  Radar as RadarIcon,
  Building2,
  Search,
} from 'lucide-react';
import { fetchDMERNAVCoverage, fetchRadarCoverageReport } from '../services/api';
import type { DMERNAVCoverageReport, RadarCoverageReport } from '../types';

export const DMERNAVCoverageView: React.FC = () => {
  // Navigation mode: 'dme' or 'radar'
  const [subtab, setSubtab] = useState<'dme' | 'radar'>('dme');

  // Shared flight level with 1,000 ft increments (FL010 to FL450)
  const [flightLevel, setFlightLevel] = useState<number>(300);

  // DME state
  const [inactiveStations, setInactiveStations] = useState<string[]>([]);
  const [dmeReport, setDmeReport] = useState<DMERNAVCoverageReport | null>(null);
  const [isDmeLoading, setIsDmeLoading] = useState<boolean>(false);

  // Radar state
  const [inactiveRadars, setInactiveRadars] = useState<string[]>([]);
  const [radarReport, setRadarReport] = useState<RadarCoverageReport | null>(null);
  const [isRadarLoading, setIsRadarLoading] = useState<boolean>(false);
  const [radarSearchQuery, setRadarSearchQuery] = useState<string>('');
  const [radarFilter, setRadarFilter] = useState<'all' | 'airport' | 'enroute'>('all');

  // Load DME coverage
  const loadDmeCoverage = async (fl: number, inactive: string[]) => {
    setIsDmeLoading(true);
    try {
      const data = await fetchDMERNAVCoverage(fl, inactive);
      setDmeReport(data);
    } finally {
      setIsDmeLoading(false);
    }
  };

  // Load Radar coverage
  const loadRadarCoverage = async (fl: number, inactive: string[]) => {
    setIsRadarLoading(true);
    try {
      const data = await fetchRadarCoverageReport(fl, inactive);
      setRadarReport(data);
    } finally {
      setIsRadarLoading(false);
    }
  };

  useEffect(() => {
    if (subtab === 'dme') {
      loadDmeCoverage(flightLevel, inactiveStations);
    } else {
      loadRadarCoverage(flightLevel, inactiveRadars);
    }
  }, [subtab, flightLevel, inactiveStations, inactiveRadars]);

  // DME actions
  const handleToggleDME = (ident: string) => {
    setInactiveStations(prev =>
      prev.includes(ident) ? prev.filter(x => x !== ident) : [...prev, ident]
    );
  };

  const handleDeactivateAllSuggestedDME = () => {
    if (!dmeReport || !dmeReport.suggested_deactivations) return;
    const suggestedIdents = dmeReport.suggested_deactivations.map(s => s.ident);
    setInactiveStations(prev => Array.from(new Set([...prev, ...suggestedIdents])));
  };

  const handleResetDME = () => {
    setInactiveStations([]);
  };

  // Radar actions
  const handleToggleRadar = (id: string) => {
    setInactiveRadars(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const handleDeactivateAllSuggestedRadar = () => {
    if (!radarReport || !radarReport.suggested_deactivations) return;
    const suggestedIds = radarReport.suggested_deactivations.map(s => s.radar_id);
    setInactiveRadars(prev => Array.from(new Set([...prev, ...suggestedIds])));
  };

  const handleResetRadars = () => {
    setInactiveRadars([]);
  };

  const filteredRadars = (radarReport?.radars || []).filter(r => {
    const matchesSearch =
      r.id.toLowerCase().includes(radarSearchQuery.toLowerCase()) ||
      r.name.toLowerCase().includes(radarSearchQuery.toLowerCase()) ||
      (r.nearest_airport_ident && r.nearest_airport_ident.toLowerCase().includes(radarSearchQuery.toLowerCase()));

    if (!matchesSearch) return false;
    if (radarFilter === 'airport') return r.is_airport_radar;
    if (radarFilter === 'enroute') return !r.is_airport_radar;
    return true;
  });

  return (
    <div className="dme-rnav-view" style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Subtab Navigation Bar */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--bg-border)', paddingBottom: '14px', marginBottom: '20px' }}>
        <button
          onClick={() => setSubtab('dme')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 18px',
            borderRadius: '6px',
            background: subtab === 'dme' ? 'var(--accent-blue)' : 'var(--bg-card)',
            color: '#fff',
            border: '1px solid var(--bg-border)',
            fontWeight: subtab === 'dme' ? 'bold' : 'normal',
            cursor: 'pointer',
            fontSize: '13px',
          }}
        >
          <Radio size={16} />
          DME-DME RNAV Backup & Rationalization Lab
        </button>

        <button
          onClick={() => setSubtab('radar')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 18px',
            borderRadius: '6px',
            background: subtab === 'radar' ? '#8b5cf6' : 'var(--bg-card)',
            color: '#fff',
            border: '1px solid var(--bg-border)',
            fontWeight: subtab === 'radar' ? 'bold' : 'normal',
            cursor: 'pointer',
            fontSize: '13px',
          }}
        >
          <RadarIcon size={16} />
          Primary & Secondary Radar Surveillance Lab (SSR / Mode S)
        </button>
      </div>

      {/* Flight Level Control Header (1,000 ft increments) */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '15px' }}>
        <div>
          {subtab === 'dme' ? (
            <>
              <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
                <Radio size={24} color="#3b82f6" />
                DME-DME RNAV Backup Coverage & Navaid Rationalization Lab
              </h2>
              <p style={{ color: 'var(--text-muted)', margin: '4px 0 0 0', fontSize: '13px' }}>
                ICAO Doc 9613 line-of-sight and dual-station geometry (30° ≤ Δθ ≤ 150°) evaluation across 10×10 NM cells.
              </p>
            </>
          ) : (
            <>
              <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
                <RadarIcon size={24} color="#8b5cf6" />
                Primary / Secondary Radar Surveillance & Multi-Tier Optimization
              </h2>
              <p style={{ color: 'var(--text-muted)', margin: '4px 0 0 0', fontSize: '13px' }}>
                Dual-head coverage (≥2) above FL100 • Single-head (≥1) below FL100 • Dual ground coverage at hubs (&gt;500 moves/day) • Max range 250 NM.
              </p>
            </>
          )}
        </div>

        {/* Flight Level Selector with 1,000 ft Stepping */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--bg-card)', padding: '8px 14px', borderRadius: '8px', border: '1px solid var(--bg-border)' }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Flight Level (1,000 ft steps):</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#fff' }}>FL{flightLevel}</span>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>({(flightLevel * 100).toLocaleString()} ft)</span>
            </div>
          </div>

          <input
            type="range"
            min={10}
            max={450}
            step={10}
            value={flightLevel}
            onChange={e => setFlightLevel(Number(e.target.value))}
            style={{ width: '130px', cursor: 'pointer' }}
          />

          <div style={{ display: 'flex', gap: '4px' }}>
            {[50, 100, 200, 300, 390].map(fl => (
              <button
                key={fl}
                onClick={() => setFlightLevel(fl)}
                className={`pill-btn ${flightLevel === fl ? 'active' : ''}`}
                style={{
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  background: flightLevel === fl ? (subtab === 'dme' ? 'var(--accent-blue)' : '#8b5cf6') : 'rgba(255, 255, 255, 0.05)',
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
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB 1: DME-DME RNAV LAB                                     */}
      {/* ------------------------------------------------------------- */}
      {subtab === 'dme' && (
        <>
          {isDmeLoading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Evaluating terrestrial radio horizon & DME pairs...</div>
          ) : dmeReport ? (
            <>
              {/* Top KPI Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
                <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '6px' }}>
                    <ShieldCheck size={18} />
                    <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>DME-DME Coverage</span>
                  </div>
                  <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{dmeReport.coverage_percentage}%</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                    {dmeReport.covered_cells_count} of {dmeReport.total_grid_cells} grid cells covered
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3b82f6', marginBottom: '6px' }}>
                    <Layers size={18} />
                    <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Redundant Availability</span>
                  </div>
                  <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{dmeReport.redundant_cells_percentage}%</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                    ≥ 3 DMEs satisfying 30°–150° angle
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', marginBottom: '6px' }}>
                    <Radio size={18} />
                    <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Active Terrestrial DMEs</span>
                  </div>
                  <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{dmeReport.active_dmes_count} / {dmeReport.total_dmes_count}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                    {dmeReport.inactive_dmes_count} stations currently disabled
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#8b5cf6', marginBottom: '6px' }}>
                    <TrendingDown size={18} />
                    <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Deactivation Potential</span>
                  </div>
                  <div style={{ fontSize: '26px', fontWeight: 'bold' }}>{dmeReport.suggested_deactivations.length} Stations</div>
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
                      onClick={handleDeactivateAllSuggestedDME}
                      className="pill-btn"
                      style={{ background: '#7c3aed', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
                    >
                      Deactivate All Suggested ({dmeReport.suggested_deactivations.length})
                    </button>
                    {inactiveStations.length > 0 && (
                      <button
                        onClick={handleResetDME}
                        className="pill-btn"
                        style={{ background: '#334155', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
                      >
                        Reset All to Active
                      </button>
                    )}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  {dmeReport.suggested_deactivations.slice(0, 9).map(sug => {
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
                          onClick={() => handleToggleDME(sug.ident)}
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
                  <h4 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>10×10 NM Spatial Grid Status (FL{dmeReport.flight_level})</h4>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '14px' }}>
                    Cells classified by line-of-sight DME availability and angular geometric intersection (30°–150°).
                  </p>

                  <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#10b981' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#10b981', display: 'inline-block' }}></span>
                      Redundant (≥3 pairs): {dmeReport.redundant_cells_percentage}%
                    </span>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#3b82f6' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#3b82f6', display: 'inline-block' }}></span>
                      Adequate (2 DMEs)
                    </span>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#ef4444' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#ef4444', display: 'inline-block' }}></span>
                      Outage: {dmeReport.outage_cells_percentage}%
                    </span>
                  </div>

                  {/* Grid cell matrix visualization snippet */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '4px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
                    {dmeReport.cells_summary.slice(0, 128).map((c, idx) => {
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
                    {dmeReport.dme_stations.slice(0, 40).map(sta => {
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
                            onClick={() => handleToggleDME(sta.ident)}
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
        </>
      )}

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB 2: RADAR SURVEILLANCE LAB (SSR / MODE S)               */}
      {/* ------------------------------------------------------------- */}
      {subtab === 'radar' && (
        <>
          {isRadarLoading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Evaluating radar horizons, airport ground lines-of-sight & surveillance cells...</div>
          ) : radarReport ? (
            <>
              {/* Top KPI Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div className="kpi-box" style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10b981', marginBottom: '4px' }}>
                    <ShieldCheck size={16} />
                    <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Target Met %</span>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{radarReport.target_met_percentage}%</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {radarReport.flight_level >= 100 ? 'Dual-head (≥2) target' : 'Single-head (≥1) target'}
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#3b82f6', marginBottom: '4px' }}>
                    <RadarIcon size={16} />
                    <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Total Coverage</span>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{radarReport.coverage_percentage}%</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {radarReport.covered_cells_count} of {radarReport.total_grid_cells} cells
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#06b6d4', marginBottom: '4px' }}>
                    <Building2 size={16} />
                    <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Hubs Dual Ground</span>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{radarReport.high_density_dual_ground_pct}%</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    ≥2 heads to ground at &gt;500 mv/d hubs
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#f59e0b', marginBottom: '4px' }}>
                    <Layers size={16} />
                    <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Active Radars</span>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{radarReport.active_radars_count} / {radarReport.total_radars_count}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {radarReport.airport_radars_count} co-located with airports (≤5 NM)
                  </div>
                </div>

                <div className="kpi-box" style={{ padding: '14px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#a855f7', marginBottom: '4px' }}>
                    <TrendingDown size={16} />
                    <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Standby Potential</span>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{radarReport.suggested_deactivations.length} Heads</div>
                  <div style={{ fontSize: '11px', color: '#c084fc', marginTop: '2px' }}>
                    €{(radarReport.suggested_deactivations.length * 85000).toLocaleString()}/yr savings
                  </div>
                </div>
              </div>

              {/* High-Density Hubs Ground Surveillance Panel */}
              <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Building2 size={18} color="#06b6d4" />
                      High-Density Aerodromes Ground Surveillance Integrity (&gt;500 Movements / Day)
                    </h3>
                    <p style={{ margin: '3px 0 0 0', color: 'var(--text-muted)', fontSize: '12px' }}>
                      Mandatory dual radar coverage to aerodrome surface elevation (h<sub>ac</sub> = h<sub>airport</sub>), prioritizing radars within 5 NM.
                    </p>
                  </div>

                  <span style={{ fontSize: '12px', color: '#10b981', fontWeight: 'bold', background: 'rgba(16, 185, 129, 0.1)', padding: '4px 10px', borderRadius: '4px' }}>
                    {radarReport.high_density_airports.filter(a => a.meets_dual_radar_target).length} / {radarReport.high_density_airports.length} Hubs Satisfy Dual Ground Target
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                  {radarReport.high_density_airports.map(apt => (
                    <div
                      key={apt.ident}
                      style={{
                        padding: '12px',
                        borderRadius: '6px',
                        background: apt.meets_dual_radar_target ? 'rgba(16, 185, 129, 0.05)' : 'rgba(239, 68, 68, 0.08)',
                        border: apt.meets_dual_radar_target ? '1px solid rgba(16, 185, 129, 0.25)' : '1px solid rgba(239, 68, 68, 0.3)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontWeight: 'bold', fontSize: '14px' }}>{apt.ident}</span>
                        {apt.meets_dual_radar_target ? (
                          <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '3px', background: '#10b981', color: '#fff', fontWeight: 'bold' }}>
                            DUAL GROUND OK
                          </span>
                        ) : (
                          <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '3px', background: '#ef4444', color: '#fff', fontWeight: 'bold' }}>
                            {apt.visible_radar_count === 1 ? 'SINGLE HEAD' : 'NO COVERAGE'}
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {apt.name}
                      </div>
                      <div style={{ fontSize: '11px', marginTop: '6px', display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
                        <span>{Math.round(apt.daily_movements)} mv/day</span>
                        <span>{apt.elevation_ft} ft MSL</span>
                        <span style={{ color: apt.visible_radar_count >= 2 ? '#10b981' : '#f59e0b', fontWeight: 'bold' }}>
                          {apt.visible_radar_count} heads visible
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Multi-Tier Network Rationalization Optimizer Panel */}
              <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Sparkles size={18} color="#a855f7" />
                      Multi-Tier Radar Network Rationalization Optimizer (Learning Engine)
                    </h3>
                    <p style={{ margin: '3px 0 0 0', color: 'var(--text-muted)', fontSize: '12px' }}>
                      Identifies redundant radar heads for standby while strictly maintaining dual-head above FL100, single-head below FL100, and dual ground coverage at major hubs.
                    </p>
                  </div>

                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      onClick={handleDeactivateAllSuggestedRadar}
                      className="pill-btn"
                      style={{ background: '#7c3aed', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
                    >
                      Deactivate All Suggested ({radarReport.suggested_deactivations.length})
                    </button>
                    {inactiveRadars.length > 0 && (
                      <button
                        onClick={handleResetRadars}
                        className="pill-btn"
                        style={{ background: '#334155', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
                      >
                        Reset All to Active
                      </button>
                    )}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  {radarReport.suggested_deactivations.slice(0, 9).map(sug => {
                    const isDeactivated = inactiveRadars.includes(sug.radar_id);
                    return (
                      <div
                        key={sug.radar_id}
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
                          <div style={{ fontWeight: 'bold', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            {sug.radar_id}
                            {sug.is_airport_radar ? (
                              <span style={{ fontSize: '10px', background: '#3b82f6', color: '#fff', padding: '1px 5px', borderRadius: '3px' }}>
                                Airport ({sug.nearest_airport})
                              </span>
                            ) : (
                              <span style={{ fontSize: '10px', background: '#475569', color: '#cbd5e1', padding: '1px 5px', borderRadius: '3px' }}>
                                En-Route
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{sug.name}</div>
                          <div style={{ fontSize: '11px', color: sug.coverage_loss_pct === 0 ? '#10b981' : '#f59e0b', marginTop: '3px' }}>
                            Impact: {sug.coverage_loss_pct}% • Savings: €{sug.annual_cost_savings_eur.toLocaleString()}/yr
                          </div>
                          <div style={{ fontSize: '10px', color: '#94a3b8' }}>
                            Power: {sug.annual_power_savings_kwh.toLocaleString()} kWh/yr
                          </div>
                        </div>

                        <button
                          onClick={() => handleToggleRadar(sug.radar_id)}
                          style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: isDeactivated ? '#ef4444' : '#10b981' }}
                        >
                          {isDeactivated ? <ToggleLeft size={24} /> : <ToggleRight size={24} />}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Interactive Radar Spatial Grid & Station Inventory */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* 10x10 NM Grid Cells Breakdown */}
                <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
                  <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>
                    10×10 NM Spatial Surveillance Grid (FL{radarReport.flight_level} - {radarReport.altitude_ft.toLocaleString()} ft)
                  </h4>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '14px' }}>
                    Line-of-sight evaluated up to 250 NM max range per radar head.
                  </p>

                  <div style={{ display: 'flex', gap: '10px', marginBottom: '16px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#10b981' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#10b981', display: 'inline-block' }}></span>
                      High Redundancy (≥3 heads)
                    </span>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#3b82f6' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#3b82f6', display: 'inline-block' }}></span>
                      Dual-Head (2 heads)
                    </span>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#f59e0b' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#f59e0b', display: 'inline-block' }}></span>
                      Single-Head (1 head)
                    </span>
                    <span style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px', color: '#ef4444' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#ef4444', display: 'inline-block' }}></span>
                      Outage (0 heads)
                    </span>
                  </div>

                  {/* Grid cell matrix visualization snippet */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '4px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
                    {radarReport.cells_summary.slice(0, 128).map((c, idx) => {
                      const bg =
                        c.coverage_status === 'HIGH_REDUNDANCY'
                          ? '#10b981'
                          : c.coverage_status === 'DUAL_HEAD'
                          ? '#3b82f6'
                          : c.coverage_status === 'SINGLE_HEAD'
                          ? '#f59e0b'
                          : '#ef4444';
                      return (
                        <div
                          key={idx}
                          title={`Cell (${c.lat}°N, ${c.lon}°E): ${c.coverage_status} (${c.visible_radars_count} heads visible)`}
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

                {/* Radar Stations Roster with Manual Toggles & Search */}
                <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', maxHeight: '450px', overflowY: 'auto' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                    <h4 style={{ margin: 0, fontSize: '14px' }}>Surveillance Radar Heads Roster</h4>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Click toggle to simulate radar head outage</span>
                  </div>

                  {/* Filters and search */}
                  <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
                    <div style={{ position: 'relative', flex: 1 }}>
                      <Search size={14} style={{ position: 'absolute', left: '8px', top: '8px', color: 'var(--text-muted)' }} />
                      <input
                        type="text"
                        placeholder="Search radar ID, name, or airport..."
                        value={radarSearchQuery}
                        onChange={e => setRadarSearchQuery(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '6px 8px 6px 28px',
                          borderRadius: '4px',
                          border: '1px solid var(--bg-border)',
                          background: 'rgba(0,0,0,0.2)',
                          color: '#fff',
                          fontSize: '12px',
                        }}
                      />
                    </div>

                    <div style={{ display: 'flex', gap: '4px' }}>
                      <button
                        onClick={() => setRadarFilter('all')}
                        style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '11px',
                          background: radarFilter === 'all' ? '#8b5cf6' : 'rgba(255,255,255,0.05)',
                          color: '#fff',
                          border: '1px solid var(--bg-border)',
                          cursor: 'pointer',
                        }}
                      >
                        All
                      </button>
                      <button
                        onClick={() => setRadarFilter('airport')}
                        style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '11px',
                          background: radarFilter === 'airport' ? '#3b82f6' : 'rgba(255,255,255,0.05)',
                          color: '#fff',
                          border: '1px solid var(--bg-border)',
                          cursor: 'pointer',
                        }}
                      >
                        &le;5 NM Airport
                      </button>
                      <button
                        onClick={() => setRadarFilter('enroute')}
                        style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '11px',
                          background: radarFilter === 'enroute' ? '#64748b' : 'rgba(255,255,255,0.05)',
                          color: '#fff',
                          border: '1px solid var(--bg-border)',
                          cursor: 'pointer',
                        }}
                      >
                        En-Route
                      </button>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {filteredRadars.slice(0, 50).map(r => {
                      const isDeactivated = inactiveRadars.includes(r.id);
                      return (
                        <div
                          key={r.id}
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
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <strong>{r.id}</strong> - {r.name}
                              {r.is_airport_radar && (
                                <span style={{ fontSize: '10px', background: '#3b82f6', color: '#fff', padding: '1px 4px', borderRadius: '3px' }}>
                                  {r.nearest_airport_ident} ({r.nearest_airport_dist_nm} NM)
                                </span>
                              )}
                              {r.mode_s && (
                                <span style={{ fontSize: '10px', background: '#10b981', color: '#fff', padding: '1px 4px', borderRadius: '3px' }}>
                                  Mode S
                                </span>
                              )}
                            </div>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                              Type: {r.type} • Elev: {r.elevation_ft} ft • Range: {r.max_range_nm} NM
                            </div>
                          </div>

                          <button
                            onClick={() => handleToggleRadar(r.id)}
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
        </>
      )}
    </div>
  );
};
