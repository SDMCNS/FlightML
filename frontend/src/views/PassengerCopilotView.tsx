import React, { useState } from 'react';
import { Luggage, Search, AlertTriangle, Scale, Clock, ArrowRight } from 'lucide-react';
import { assessPassengerItinerary } from '../services/api';
import type { ItineraryRisk } from '../types';

export const PassengerCopilotView: React.FC = () => {
  const [origin, setOrigin] = useState('EGLL');
  const [destination, setDestination] = useState('LTFM');
  const [hub, setHub] = useState('EDDF');
  const [layoverMin, setLayoverMin] = useState(55);
  const [result, setResult] = useState<ItineraryRisk | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const handleEvaluate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsEvaluating(true);
    try {
      const res = await assessPassengerItinerary(origin, destination, hub || undefined);
      setResult(res);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="view-container passenger-copilot-view">
      <div className="view-header">
        <div className="header-icon-box passenger">
          <Luggage size={26} />
        </div>
        <div>
          <h2>🧳 Passenger Travel Copilot & Multi-Leg Journey Predictor</h2>
          <p className="subtitle">
            Empirical Delay Distributions, Transfer Hub Missed Connection Probability, and EU261 / UK261 Compensation Rights
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
            <label>Transfer Hub (Optional):</label>
            <input
              type="text"
              value={hub}
              onChange={e => setHub(e.target.value.toUpperCase())}
              placeholder="e.g. EDDF / FRA (Leave blank for direct)"
              className="airport-input"
            />
          </div>

          <div className="form-col">
            <label>Destination Airport (ADES):</label>
            <input
              type="text"
              value={destination}
              onChange={e => setDestination(e.target.value.toUpperCase())}
              placeholder="e.g. LTFM / IST"
              className="airport-input"
              required
            />
          </div>

          <div className="form-col">
            <label>Layover (min):</label>
            <input
              type="number"
              value={layoverMin}
              onChange={e => setLayoverMin(Number(e.target.value))}
              className="airport-input"
              min="30"
              max="360"
            />
          </div>

          <div className="form-col btn-col">
            <button type="submit" disabled={isEvaluating} className="predict-btn">
              <Search size={16} /> {isEvaluating ? 'Evaluating...' : 'Predict Itinerary Risk'}
            </button>
          </div>
        </form>
      </div>

      {/* Results Deck */}
      {result ? (
        <div className="results-grid">
          <div className="stat-card">
            <div className="card-top">
              <span className="title">Missed Connection Risk</span>
              <AlertTriangle size={18} className="card-icon text-yellow" />
            </div>
            <div className={`value ${result.missed_connection_prob > 0.2 ? 'text-red' : 'text-green'}`}>
              {(result.missed_connection_prob * 100).toFixed(1)}%
            </div>
            <div className="caption">Under peak bank arrival delay distributions</div>
          </div>

          <div className="stat-card">
            <div className="card-top">
              <span className="title">Predicted Arr Delay</span>
              <Clock size={18} className="card-icon" />
            </div>
            <div className="value">+{result.predicted_delay_min} min</div>
            <div className="caption">Final destination arrival estimate</div>
          </div>

          <div className="stat-card">
            <div className="card-top">
              <span className="title">EU261 / UK261 Compensation</span>
              <Scale size={18} className="card-icon text-blue" />
            </div>
            <div className="value text-blue">
              {result.compensation_eur > 0 ? `€${result.compensation_eur}` : '€0 (Under 3h)'}
            </div>
            <div className="caption">Statutory compensation entitlement</div>
          </div>

          <div className="content-card full-width">
            <div className="card-header-row">
              <h3>FlightML Recommendation & Buffer Analysis</h3>
              <span className={`status-pill ${result.risk_level.toLowerCase()}`}>{result.risk_level} RISK</span>
            </div>
            <p className="recommendation-text">{result.recommendation}</p>

            <div className="journey-summary-strip">
              <span className="city-box">{result.origin}</span>
              <ArrowRight size={16} />
              {result.hub && (
                <>
                  <span className="city-box hub">{result.hub} ({layoverMin}m layover)</span>
                  <ArrowRight size={16} />
                </>
              )}
              <span className="city-box">{result.destination}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="empty-copilot-state">
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
