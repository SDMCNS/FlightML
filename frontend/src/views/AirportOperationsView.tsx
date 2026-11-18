import React, { useState, useEffect } from 'react';
import {
  Building2,
  Wind,
  Compass,
  AlertTriangle,
  PlaneLanding,
  Clock,
  Gauge,
  Layers,
  ShieldCheck,
} from 'lucide-react';
import { fetchAirportOperations } from '../services/api';
import type { AirportOperationsReport } from '../types';

interface AirportOperationsViewProps {
  period?: string;
}

export const AirportOperationsView: React.FC<AirportOperationsViewProps> = ({ period = '202402' }) => {
  const [airportIdent, setAirportIdent] = useState<string>('EGLL');
  const [report, setReport] = useState<AirportOperationsReport | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Weather simulator controls
  const [windDir, setWindDir] = useState<number>(250);
  const [windSpeed, setWindSpeed] = useState<number>(16);
  const [tempC, setTempC] = useState<number>(12);

  const quickAirports = [
    { code: 'EGLL', name: 'London Heathrow' },
    { code: 'EHAM', name: 'Amsterdam Schiphol' },
    { code: 'LFPG', name: 'Paris CDG' },
    { code: 'EDDF', name: 'Frankfurt' },
    { code: 'EBBR', name: 'Brussels' },
    { code: 'EINN', name: 'Shannon' },
    { code: 'KJFK', name: 'New York JFK' },
  ];

  const loadData = async (ident: string, wDir: number, wSpd: number, tC: number) => {
    setIsLoading(true);
    try {
      const data = await fetchAirportOperations(ident, period, {
        wind_dir: wDir,
        wind_speed: wSpd,
        temp_c: tC,
      });
      setReport(data);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData(airportIdent, windDir, windSpeed, tempC);
  }, [airportIdent, period]);

  const handleApplyWeather = (e: React.FormEvent) => {
    e.preventDefault();
    loadData(airportIdent, windDir, windSpeed, tempC);
  };

  return (
    <div className="airport-ops-view" style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header bar */}
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
            <Building2 size={24} color="#3b82f6" />
            Airport Operations & Extrapolated Runway Intelligence
          </h2>
          <p style={{ color: 'var(--text-muted)', margin: '4px 0 0 0', fontSize: '13px' }}>
            Daily arrival/departure volume, hourly demand peaks, and aerodynamic crosswind/active configuration modeling.
          </p>
        </div>

        {/* Airport Search */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <input
            type="text"
            value={airportIdent}
            onChange={e => setAirportIdent(e.target.value.toUpperCase())}
            placeholder="ICAO (e.g. EGLL)"
            className="airport-input"
            style={{ width: '130px', textAlign: 'center', fontWeight: 'bold' }}
            maxLength={4}
          />
          <button
            onClick={() => loadData(airportIdent, windDir, windSpeed, tempC)}
            className="pill-btn"
            style={{ background: 'var(--accent-blue)', color: '#fff', padding: '8px 16px', borderRadius: '4px' }}
          >
            Load Airport
          </button>
        </div>
      </div>

      {/* Quick Airport Selector Pills */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '20px', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Quick Select:</span>
        {quickAirports.map(apt => (
          <button
            key={apt.code}
            onClick={() => {
              setAirportIdent(apt.code);
            }}
            className={`pill-btn ${airportIdent === apt.code ? 'active' : ''}`}
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              fontSize: '12px',
              background: airportIdent === apt.code ? 'var(--accent-blue)' : 'var(--bg-card)',
              color: '#fff',
              border: '1px solid var(--bg-border)',
              cursor: 'pointer',
            }}
          >
            {apt.code} - {apt.name}
          </button>
        ))}
      </div>

      {/* Weather Simulator Control Bar */}
      <div className="analytics-card" style={{ padding: '16px', marginBottom: '20px', background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--bg-border)' }}>
        <form onSubmit={handleApplyWeather} style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Wind size={18} color="#06b6d4" />
            <span style={{ fontWeight: 600, fontSize: '13px' }}>METAR Simulator:</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Wind Direction (°):</label>
            <input
              type="number"
              min="0"
              max="360"
              value={windDir}
              onChange={e => setWindDir(Number(e.target.value))}
              style={{ width: '70px', padding: '4px 8px', borderRadius: '4px', background: 'var(--bg-input)', color: '#fff', border: '1px solid var(--bg-border)' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Wind Speed (kts):</label>
            <input
              type="number"
              min="0"
              max="60"
              value={windSpeed}
              onChange={e => setWindSpeed(Number(e.target.value))}
              style={{ width: '70px', padding: '4px 8px', borderRadius: '4px', background: 'var(--bg-input)', color: '#fff', border: '1px solid var(--bg-border)' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Temperature (°C):</label>
            <input
              type="number"
              min="-30"
              max="50"
              value={tempC}
              onChange={e => setTempC(Number(e.target.value))}
              style={{ width: '70px', padding: '4px 8px', borderRadius: '4px', background: 'var(--bg-input)', color: '#fff', border: '1px solid var(--bg-border)' }}
            />
          </div>

          <button
            type="submit"
            className="pill-btn"
            style={{ background: '#0284c7', color: '#fff', padding: '6px 14px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px' }}
          >
            Update Runway Crosswinds
          </button>
        </form>
      </div>

      {isLoading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading airport operations...</div>
      ) : report ? (
        <>
          {/* Top KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#3b82f6', marginBottom: '6px' }}>
                <PlaneLanding size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Daily Movements</span>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{report.total_movements.toLocaleString()}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {report.total_arrivals.toLocaleString()} Arr / {report.total_departures.toLocaleString()} Dep
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', marginBottom: '6px' }}>
                <Clock size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Average Delay</span>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{report.mean_arr_delay_min} min</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Departure Delay: {report.mean_dep_delay_min} min
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f59e0b', marginBottom: '6px' }}>
                <Gauge size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Field Elevation</span>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{report.elevation_ft} ft</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {report.name} ({report.country})
              </div>
            </div>

            <div className="kpi-box" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#8b5cf6', marginBottom: '6px' }}>
                <Compass size={18} />
                <span style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Active Flow</span>
              </div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#a78bfa', marginTop: '4px' }}>
                {report.recommended_active_config.split(':')[1] || report.recommended_active_config}
              </div>
            </div>
          </div>

          {/* Operational Advisories Banner */}
          {report.operational_advisories.length > 0 && (
            <div style={{ padding: '12px 16px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertTriangle size={18} color="#ef4444" />
              <div style={{ fontSize: '13px', color: '#fca5a5' }}>
                <strong>Operational Alerts:</strong> {report.operational_advisories.join(' • ')}
              </div>
            </div>
          )}

          {/* Extrapolated Runway Intelligence Table */}
          <div className="analytics-card" style={{ padding: '20px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)', marginBottom: '20px' }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} color="#3b82f6" />
              Extrapolated Runway Performance & Aerodynamic Crosswind Components
            </h3>

            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-border)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '8px' }}>Runway</th>
                  <th style={{ padding: '8px' }}>Heading (°T)</th>
                  <th style={{ padding: '8px' }}>Dimensions</th>
                  <th style={{ padding: '8px' }}>LDA Available</th>
                  <th style={{ padding: '8px' }}>Crosswind</th>
                  <th style={{ padding: '8px' }}>Headwind / Tailwind</th>
                  <th style={{ padding: '8px' }}>Preferred Status</th>
                </tr>
              </thead>
              <tbody>
                {report.runways.map(rwy => (
                  <React.Fragment key={rwy.runway_id}>
                    {/* Low End */}
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: rwy.preferred_end === rwy.le_ident ? 'rgba(59, 130, 246, 0.08)' : 'transparent' }}>
                      <td style={{ padding: '10px 8px', fontWeight: 'bold' }}>
                        RWY {rwy.le_ident}
                        {rwy.preferred_end === rwy.le_ident && (
                          <span style={{ marginLeft: '6px', fontSize: '10px', background: '#2563eb', color: '#fff', padding: '2px 6px', borderRadius: '3px' }}>
                            ACTIVE
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '10px 8px' }}>{rwy.le_heading_degT}°T</td>
                      <td style={{ padding: '10px 8px' }}>{rwy.length_ft.toLocaleString()} × {rwy.width_ft} ft ({rwy.surface})</td>
                      <td style={{ padding: '10px 8px' }}>{rwy.le_lda_ft.toLocaleString()} ft</td>
                      <td style={{ padding: '10px 8px', color: rwy.le_exceeds_crosswind_limit ? '#ef4444' : '#fff' }}>
                        {rwy.le_crosswind_kts} kts {rwy.le_exceeds_crosswind_limit && '⚠️'}
                      </td>
                      <td style={{ padding: '10px 8px', color: rwy.le_headwind_kts >= 0 ? '#10b981' : '#f87171' }}>
                        {rwy.le_headwind_kts >= 0 ? `+${rwy.le_headwind_kts} kts (Headwind)` : `${rwy.le_headwind_kts} kts (Tailwind)`}
                      </td>
                      <td style={{ padding: '10px 8px' }}>
                        {rwy.preferred_end === rwy.le_ident ? (
                          <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <ShieldCheck size={14} /> Preferred for Landing/Dep
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>Unfavorable Wind</span>
                        )}
                      </td>
                    </tr>

                    {/* High End */}
                    <tr style={{ borderBottom: '1px solid var(--bg-border)', background: rwy.preferred_end === rwy.he_ident ? 'rgba(59, 130, 246, 0.08)' : 'transparent' }}>
                      <td style={{ padding: '10px 8px', fontWeight: 'bold' }}>
                        RWY {rwy.he_ident}
                        {rwy.preferred_end === rwy.he_ident && (
                          <span style={{ marginLeft: '6px', fontSize: '10px', background: '#2563eb', color: '#fff', padding: '2px 6px', borderRadius: '3px' }}>
                            ACTIVE
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '10px 8px' }}>{rwy.he_heading_degT}°T</td>
                      <td style={{ padding: '10px 8px' }}>{rwy.length_ft.toLocaleString()} × {rwy.width_ft} ft ({rwy.surface})</td>
                      <td style={{ padding: '10px 8px' }}>{rwy.he_lda_ft.toLocaleString()} ft</td>
                      <td style={{ padding: '10px 8px', color: rwy.he_exceeds_crosswind_limit ? '#ef4444' : '#fff' }}>
                        {rwy.he_crosswind_kts} kts {rwy.he_exceeds_crosswind_limit && '⚠️'}
                      </td>
                      <td style={{ padding: '10px 8px', color: rwy.he_headwind_kts >= 0 ? '#10b981' : '#f87171' }}>
                        {rwy.he_headwind_kts >= 0 ? `+${rwy.he_headwind_kts} kts (Headwind)` : `${rwy.he_headwind_kts} kts (Tailwind)`}
                      </td>
                      <td style={{ padding: '10px 8px' }}>
                        {rwy.preferred_end === rwy.he_ident ? (
                          <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <ShieldCheck size={14} /> Preferred for Landing/Dep
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>Unfavorable Wind</span>
                        )}
                      </td>
                    </tr>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>

          {/* Traffic Breakdown (Airlines, Fleet Mix, Hourly Distribution) */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            {/* Top Operating Airlines */}
            <div className="analytics-card" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <h4 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>Top Operating Carriers</h4>
              {report.top_airlines.map(op => (
                <div key={op.operator} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '13px' }}>
                  <span style={{ fontWeight: 600 }}>{op.operator}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{op.flights.toLocaleString()} flights ({op.share_pct}%)</span>
                </div>
              ))}
            </div>

            {/* Fleet Mix */}
            <div className="analytics-card" style={{ padding: '16px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--bg-border)' }}>
              <h4 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>Top Aircraft Types</h4>
              {report.top_aircraft.map(ac => (
                <div key={ac.ac_type} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '13px' }}>
                  <span style={{ fontWeight: 600 }}>{ac.ac_type}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{ac.flights.toLocaleString()} flights ({ac.share_pct}%)</span>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
