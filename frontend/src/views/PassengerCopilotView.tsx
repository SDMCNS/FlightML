import React, { useState } from 'react';
import {
  Luggage,
  Search,
  AlertTriangle,
  Scale,
  Clock,
  ArrowRight,
  Building2,
  Info,
} from 'lucide-react';
import { assessPassengerItinerary } from '../services/api';
import { getHubMCTProfile } from '../math/mct';
import type { ItineraryRisk } from '../types';

export const PassengerCopilotView: React.FC = () => {
  const [origin, setOrigin] = useState('EGLL');
  const [destination, setDestination] = useState('KLAX');
  const [hub, setHub] = useState('KJFK');
  const [layoverMin, setLayoverMin] = useState(90);
  const [result, setResult] = useState<ItineraryRisk | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Quick select popular hubs
  const popularHubs = [
    { code: 'KJFK', label: 'JFK' },
    { code: 'EGLL', label: 'LHR' },
    { code: 'LFPG', label: 'CDG' },
    { code: 'EHAM', label: 'AMS' },
    { code: 'EDDF', label: 'FRA' },
    { code: 'OMDB', label: 'DXB' },
    { code: 'KORD', label: 'ORD' },
    { code: 'KATL', label: 'ATL' },
    { code: 'KLAX', label: 'LAX' },
    { code: 'KSFO', label: 'SFO' },
    { code: 'WSSS', label: 'SIN' },
    { code: 'RJTT', label: 'HND' },
    { code: 'EDDM', label: 'MUC' },
  ];

  const hubProfile = hub ? getHubMCTProfile(hub) : null;

  const handleEvaluate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsEvaluating(true);
    try {
      const res = await assessPassengerItinerary(origin, destination, hub || undefined, layoverMin);
      setResult(res);
    } finally {
      setIsEvaluating(false);
    }
  };

  // Determine active connection type for highlighting
  const activeConnType = result?.connection_type || 'I_to_D';

  return (
    <div className="view-container passenger-copilot-view">
      <div className="view-header">
        <div className="header-icon-box passenger">
          <Luggage size={26} />
        </div>
        <div>
          <h2>🧳 Passenger Travel Copilot & Multi-Leg Journey Predictor</h2>
          <p className="subtitle">
            Empirical Delay Distributions, Transfer Hub Minimum Connection Time (MCT) Intelligence, and EU261 / UK261 Compensation Rights
          </p>
        </div>
      </div>

      {/* Form Deck */}
      <div className="itinerary-card">
        <form onSubmit={handleEvaluate} className="itinerary-form">
          <div className="form-col">
            <label>Origin Airport (ADEP):</label>
            <input
              type="text"
              value={origin}
              onChange={e => setOrigin(e.target.value.toUpperCase())}
              placeholder="e.g. EGLL / LHR"
              className="airport-input"
              required
            />
          </div>

          <div className="form-col">
            <label>Transfer Hub:</label>
            <input
              type="text"
              value={hub}
              onChange={e => setHub(e.target.value.toUpperCase())}
              placeholder="e.g. KJFK / JFK (Optional)"
              className="airport-input"
            />
          </div>

          <div className="form-col">
            <label>Destination Airport (ADES):</label>
            <input
              type="text"
              value={destination}
              onChange={e => setDestination(e.target.value.toUpperCase())}
              placeholder="e.g. KLAX / LAX"
              className="airport-input"
              required
            />
          </div>

          <div className="form-col">
            <label>Scheduled Layover (min):</label>
            <input
              type="number"
              value={layoverMin}
              onChange={e => setLayoverMin(Number(e.target.value))}
              className="airport-input"
              min="20"
              max="480"
              required
            />
          </div>

          <div className="form-col btn-col">
            <button type="submit" disabled={isEvaluating} className="predict-btn">
              <Search size={16} /> {isEvaluating ? 'Evaluating...' : 'Predict Itinerary Risk'}
            </button>
          </div>
        </form>

        {/* Quick Hub Selector Pills */}
        <div className="hub-pills-row" style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Quick Hubs:</span>
          {popularHubs.map(h => (
            <button
              key={h.code}
              type="button"
              className={`pill-btn ${hub === h.code ? 'active' : ''}`}
              style={{
                fontSize: '11px',
                padding: '4px 8px',
                borderRadius: '4px',
                border: '1px solid var(--bg-border)',
                background: hub === h.code ? 'var(--accent-blue)' : 'var(--bg-input)',
                color: '#fff',
                cursor: 'pointer',
              }}
              onClick={() => setHub(h.code)}
            >
              {h.label}
            </button>
          ))}
        </div>
      </div>

      {/* Hub MCT Intelligence Matrix Card */}
      {hubProfile && (
        <div className="analytics-card" style={{ marginTop: '16px' }}>
          <div className="card-header-row">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Building2 size={18} color="#3b82f6" />
              <h4>
                {hubProfile.airport} ({hubProfile.iata} / {hubProfile.icao})
              </h4>
              {hubProfile.is_placeholder ? (
                <span
                  style={{
                    background: 'rgba(148, 163, 184, 0.15)',
                    color: '#94a3b8',
                    border: '1px solid #475569',
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontWeight: 500,
                  }}
                >
                  Industry Baseline (Placeholder)
                </span>
              ) : (
                <span
                  style={{
                    background: 'rgba(59, 130, 246, 0.2)',
                    color: '#60a5fa',
                    border: '1px solid #2563eb',
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontWeight: 500,
                  }}
                >
                  Verified Hub Profile
                </span>
              )}
            </div>
            <span className="badge-tag">{hubProfile.city}, {hubProfile.country}</span>
          </div>

          <div className="mct-matrix-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginTop: '12px' }}>
            {/* D to D */}
            <div
              className={`kpi-box ${activeConnType === 'D_to_D' ? 'active-highlight' : ''}`}
              style={{
                padding: '10px',
                borderRadius: '6px',
                border: activeConnType === 'D_to_D' ? '2px solid #3b82f6' : '1px solid var(--bg-border)',
                background: 'var(--bg-card)',
              }}
            >
              <span className="kpi-label" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Domestic ➔ Domestic</span>
              <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{hubProfile.D_to_D}</div>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>D_to_D MCT</span>
            </div>

            {/* D to I */}
            <div
              className={`kpi-box ${activeConnType === 'D_to_I' ? 'active-highlight' : ''}`}
              style={{
                padding: '10px',
                borderRadius: '6px',
                border: activeConnType === 'D_to_I' ? '2px solid #3b82f6' : '1px solid var(--bg-border)',
                background: 'var(--bg-card)',
              }}
            >
              <span className="kpi-label" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Domestic ➔ International</span>
              <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{hubProfile.D_to_I}</div>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>D_to_I MCT</span>
            </div>

            {/* I to D */}
            <div
              className={`kpi-box ${activeConnType === 'I_to_D' ? 'active-highlight' : ''}`}
              style={{
                padding: '10px',
                borderRadius: '6px',
                border: activeConnType === 'I_to_D' ? '2px solid #ef4444' : '1px solid var(--bg-border)',
                background: 'var(--bg-card)',
              }}
            >
              <span className="kpi-label" style={{ fontSize: '11px', color: '#ef4444' }}>International ➔ Domestic</span>
              <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px', color: '#ef4444' }}>{hubProfile.I_to_D}</div>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Border & TSA Re-screen</span>
            </div>

            {/* I to I */}
            <div
              className={`kpi-box ${activeConnType === 'I_to_I' ? 'active-highlight' : ''}`}
              style={{
                padding: '10px',
                borderRadius: '6px',
                border: activeConnType === 'I_to_I' ? '2px solid #3b82f6' : '1px solid var(--bg-border)',
                background: 'var(--bg-card)',
              }}
            >
              <span className="kpi-label" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>International ➔ International</span>
              <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{hubProfile.I_to_I}</div>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Airside Transit MCT</span>
            </div>
          </div>

          {/* Operational Guidance & Layover Adequacy Banner */}
          <div
            style={{
              marginTop: '12px',
              padding: '10px 14px',
              borderRadius: '6px',
              backgroundColor: 'var(--bg-card-hover)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '10px',
            }}
          >
            <Info size={18} color="#3b82f6" style={{ marginTop: '2px', flexShrink: 0 }} />
            <div style={{ fontSize: '13px' }}>
              <strong>Terminal & Transfer Notes:</strong> {hubProfile.notes}
            </div>
          </div>
        </div>
      )}

      {/* Results Deck */}
      {result ? (
        <div className="results-grid" style={{ marginTop: '16px' }}>
          {/* Missed Connection Gauge */}
          <div className="stat-card">
            <div className="card-top">
              <span className="title">Missed Connection Risk</span>
              <AlertTriangle size={18} className={`card-icon ${result.missed_connection_prob > 0.25 ? 'text-red' : 'text-green'}`} />
            </div>
            <div className={`value ${result.missed_connection_prob > 0.25 ? 'text-red' : 'text-green'}`}>
              {(result.missed_connection_prob * 100).toFixed(1)}%
            </div>
            <div className="caption">
              {result.slack_buffer_min !== undefined
                ? `Buffer slack: ${result.slack_buffer_min > 0 ? '+' : ''}${result.slack_buffer_min}m above MCT`
                : 'Under peak bank arrival delay distributions'}
            </div>
          </div>

          {/* Minimum Connection Time Threshold */}
          <div className="stat-card">
            <div className="card-top">
              <span className="title">Required Hub MCT</span>
              <Clock size={18} className="card-icon" />
            </div>
            <div className="value">
              {result.mct_min ? `${result.mct_min} min` : `${layoverMin} min`}
            </div>
            <div className="caption">
              {result.connection_type ? `Applicable connection: ${result.connection_type}` : 'Standard connection minimum'}
            </div>
          </div>

          {/* EU261 Compensation Rights */}
          <div className="stat-card">
            <div className="card-top">
              <span className="title">EU261 / UK261 Entitlement</span>
              <Scale size={18} className="card-icon text-blue" />
            </div>
            <div className="value text-blue">
              {result.compensation_eur > 0 ? `€${result.compensation_eur}` : '€0 (Under 3h)'}
            </div>
            <div className="caption">Statutory compensation per passenger if missed</div>
          </div>

          {/* Executive Recommendation Banner */}
          <div className="content-card full-width">
            <div className="card-header-row">
              <h3>FlightML Recommendation & Buffer Analysis</h3>
              <span className={`status-pill ${result.risk_level.toLowerCase()}`}>{result.risk_level} RISK</span>
            </div>
            <p className="recommendation-text" style={{ fontSize: '14px', lineHeight: '1.6', marginTop: '8px' }}>
              {result.transfer_note || result.recommendation}
            </p>

            <div className="journey-summary-strip" style={{ marginTop: '16px' }}>
              <span className="city-box">{result.origin}</span>
              <ArrowRight size={16} />
              {result.hub && (
                <>
                  <span className="city-box hub">
                    {result.hub} ({layoverMin}m layover • {result.mct_min ? `${result.mct_min}m MCT` : ''})
                  </span>
                  <ArrowRight size={16} />
                </>
              )}
              <span className="city-box">{result.destination}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="empty-copilot-state" style={{ marginTop: '24px' }}>
          <Luggage size={48} className="empty-icon" />
          <h3>Enter Your Itinerary Above to Analyze Travel Risk</h3>
          <p>
            FlightML matches your flight route against empirical airline delay distributions, transfer hub Minimum Connecting Time (MCT) buffers, and European statutory passenger rights.
          </p>
        </div>
      )}
    </div>
  );
};
