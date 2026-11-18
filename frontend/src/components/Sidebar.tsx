import React from 'react';
import {
  Compass,
  Building2,
  ShieldAlert,
  Radio,
  Globe2,
  TrendingUp,
  Leaf,
  Luggage,
  BookOpen,
} from 'lucide-react';
import type { NavigationTab } from '../types';

interface SidebarProps {
  activeTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
  flaggedCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab, flaggedCount = 0 }) => {
  const navItems: { id: NavigationTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: 'flight-inspector',
      label: '4D Flight Explorer',
      icon: <Compass size={18} />,
    },
    {
      id: 'airport-ops',
      label: 'Airport Ops & Runways',
      icon: <Building2 size={18} />,
      badge: 'MET & Crosswind',
    },
    {
      id: 'gnss-shield',
      label: 'GNSS Jamming Shield',
      icon: <ShieldAlert size={18} />,
      badge: flaggedCount > 0 ? `${flaggedCount} Confirmed` : undefined,
    },
    {
      id: 'dme-rnav',
      label: 'DME-DME RNAV Lab',
      icon: <Radio size={18} />,
      badge: 'PBN / Backup',
    },
    {
      id: 'airspace-network',
      label: 'Airspace & FIR Saturation',
      icon: <Globe2 size={18} />,
    },
    {
      id: 'longitudinal-trends',
      label: 'Longitudinal Modernization',
      icon: <TrendingUp size={18} />,
      badge: 'Capping & FRA',
    },
    {
      id: 'climate-eco',
      label: 'Contrail & Climate Lab',
      icon: <Leaf size={18} />,
    },
    {
      id: 'passenger-copilot',
      label: 'Passenger Copilot',
      icon: <Luggage size={18} />,
      badge: 'EU261',
    },
    {
      id: 'math-reference',
      label: 'Analytical Math & Proofs',
      icon: <BookOpen size={18} />,
    },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-title">NAVIGATION MODULES</span>
      </div>
      <nav className="sidebar-nav">
        {navItems.map(item => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`nav-button ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
              {item.badge && <span className="nav-badge">{item.badge}</span>}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="footer-card">
          <div className="footer-title">⚡ High-Performance Mode</div>
          <div className="footer-desc">
            Client-side reactive state. No python script re-evaluations or layout shifts.
          </div>
        </div>
      </div>
    </aside>
  );
};
