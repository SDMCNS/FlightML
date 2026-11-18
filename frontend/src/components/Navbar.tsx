import React from 'react';
import { Plane, Search, Database, Radio } from 'lucide-react';

interface NavbarProps {
  periods: string[];
  activePeriod: string;
  onPeriodChange: (p: string) => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  isLiveApi: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  periods,
  activePeriod,
  onPeriodChange,
  searchQuery,
  onSearchChange,
  isLiveApi,
}) => {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <div className="logo-icon">
          <Plane size={20} color="#fff" />
        </div>
        <div>
          <div className="brand-title">
            FlightML <span className="brand-tag">TS Cockpit</span>
          </div>
          <div className="brand-subtitle">Airspace Network & 4D Intelligence</div>
        </div>
      </div>

      <div className="navbar-search">
        <Search size={16} className="search-icon" />
        <input
          type="text"
          placeholder="Instant search: Flight ID, Call Sign, ADEP, ADES, Operator..."
          value={searchQuery}
          onChange={e => onSearchChange(e.target.value)}
          className="search-input"
        />
        {searchQuery && (
          <button className="clear-btn" onClick={() => onSearchChange('')}>
            ✕
          </button>
        )}
      </div>

      <div className="navbar-actions">
        <div className="period-selector">
          <Database size={14} className="action-icon" />
          <span className="action-label">AIRAC:</span>
          <select value={activePeriod} onChange={e => onPeriodChange(e.target.value)} className="period-select">
            {periods.map(p => (
              <option key={p} value={p}>
                {p} (Eurocontrol Archive)
              </option>
            ))}
          </select>
        </div>

        <div className={`status-badge ${isLiveApi ? 'live' : 'offline'}`}>
          <Radio size={12} className="pulse-icon" />
          <span>{isLiveApi ? 'API Connected' : 'Demo / Standalone'}</span>
        </div>
      </div>
    </header>
  );
};
