import React, { useState, useEffect } from 'react';
import type { NavigationTab, FlightSummary, Waypoint, GNSSNetworkScan } from './types';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { FlightInspectorView } from './views/FlightInspectorView';
import { GNSSShieldView } from './views/GNSSShieldView';
import { AirspaceNetworkView } from './views/AirspaceNetworkView';
import { ClimateEcoView } from './views/ClimateEcoView';
import { PassengerCopilotView } from './views/PassengerCopilotView';
import { MathReferenceView } from './views/MathReferenceView';
import {
  checkApiHealth,
  fetchPeriods,
  searchFlights,
  fetchFlightTrajectory,
  fetchGNSSNetworkScan,
} from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('flight-inspector');
  const [periods, setPeriods] = useState<string[]>(['202402', '202510']);
  const [activePeriod, setActivePeriod] = useState<string>('202402');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isLiveApi, setIsLiveApi] = useState<boolean>(false);

  const [flights, setFlights] = useState<FlightSummary[]>([]);
  const [selectedFlight, setSelectedFlight] = useState<FlightSummary | null>(null);
  const [waypoints, setWaypoints] = useState<Waypoint[]>([]);
  const [gnssScan, setGnssScan] = useState<GNSSNetworkScan | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Check backend health & load periods on mount
  useEffect(() => {
    async function init() {
      const live = await checkApiHealth();
      setIsLiveApi(live);

      const pList = await fetchPeriods();
      setPeriods(pList);
      if (pList.length > 0) {
        setActivePeriod(pList[0]);
      }
    }
    init();
  }, []);

  // Fetch flights when period or search query changes
  useEffect(() => {
    let cancelled = false;
    async function loadFlights() {
      setIsLoading(true);
      try {
        const list = await searchFlights(activePeriod, searchQuery);
        if (!cancelled) {
          setFlights(list);
          if (list.length > 0 && !selectedFlight) {
            setSelectedFlight(list[0]);
          }
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    loadFlights();
    return () => {
      cancelled = true;
    };
  }, [activePeriod, searchQuery]);

  // Fetch trajectory when selected flight changes
  useEffect(() => {
    let cancelled = false;
    async function loadTrajectory() {
      if (!selectedFlight) {
        setWaypoints([]);
        return;
      }
      try {
        const pts = await fetchFlightTrajectory(activePeriod, selectedFlight.ectrl_id);
        if (!cancelled) {
          setWaypoints(pts);
        }
      } catch {
        if (!cancelled) setWaypoints([]);
      }
    }
    loadTrajectory();
    return () => {
      cancelled = true;
    };
  }, [activePeriod, selectedFlight]);

  // Fetch GNSS Network Scan
  useEffect(() => {
    let cancelled = false;
    async function loadGNSS() {
      const scan = await fetchGNSSNetworkScan(activePeriod, 75.0);
      if (!cancelled) {
        setGnssScan(scan);
      }
    }
    loadGNSS();
    return () => {
      cancelled = true;
    };
  }, [activePeriod]);

  const handleRefreshGNSS = async (score: number, hotspot: string) => {
    setIsLoading(true);
    try {
      const scan = await fetchGNSSNetworkScan(activePeriod, score, hotspot);
      setGnssScan(scan);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flightml-app">
      <Navbar
        periods={periods}
        activePeriod={activePeriod}
        onPeriodChange={setActivePeriod}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        isLiveApi={isLiveApi}
      />

      <div className="app-body">
        <Sidebar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          flaggedCount={gnssScan?.spoofing_confirmed_count || 0}
        />

        <main className="main-content">
          {activeTab === 'flight-inspector' && (
            <FlightInspectorView
              flights={flights}
              selectedFlight={selectedFlight}
              onSelectFlight={setSelectedFlight}
              waypoints={waypoints}
              isLoading={isLoading}
            />
          )}

          {activeTab === 'gnss-shield' && (
            <GNSSShieldView
              scan={gnssScan}
              onRefreshScan={handleRefreshGNSS}
              isLoading={isLoading}
            />
          )}

          {activeTab === 'airspace-network' && <AirspaceNetworkView />}

          {activeTab === 'climate-eco' && <ClimateEcoView />}

          {activeTab === 'passenger-copilot' && <PassengerCopilotView />}

          {activeTab === 'math-reference' && <MathReferenceView />}
        </main>
      </div>
    </div>
  );
};

export default App;
