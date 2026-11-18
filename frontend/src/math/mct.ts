/**
 * Global Hub Minimum Connection Times (MCT) Database (TypeScript).
 * 
 * Provides official connection times across:
 * - Domestic to Domestic (D_to_D)
 * - Domestic to International (D_to_I)
 * - International to Domestic (I_to_D)
 * - International to International (I_to_I)
 */

export type ConnectionType = 'D_to_D' | 'D_to_I' | 'I_to_D' | 'I_to_I';

export interface HubMCTEntry {
  airport: string;
  city: string;
  country: string;
  country_code: string;
  icao: string;
  iata: string;
  D_to_D: string;
  D_to_I: string;
  I_to_D: string;
  I_to_I: string;
  d_to_d_min: number;
  d_to_i_min: number;
  i_to_d_min: number;
  i_to_i_min: number;
  notes: string;
}

export const GLOBAL_HUB_MCT_DATA: Record<string, HubMCTEntry> = {
  JFK: {
    airport: "John F. Kennedy International Airport",
    city: "New York",
    country: "United States",
    country_code: "US",
    icao: "KJFK",
    iata: "JFK",
    D_to_D: "45 min",
    D_to_I: "90 min",
    I_to_D: "120-135 min",
    I_to_I: "90 min",
    d_to_d_min: 45,
    d_to_i_min: 90,
    i_to_d_min: 135,
    i_to_i_min: 90,
    notes: "No airside terminal connections; AirTrain and TSA re-screening required."
  },
  LAX: {
    airport: "Los Angeles International Airport",
    city: "Los Angeles",
    country: "United States",
    country_code: "US",
    icao: "KLAX",
    iata: "LAX",
    D_to_D: "45 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 45,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "Inter-terminal transfers require landside shuttle and TSA re-screening."
  },
  LHR: {
    airport: "London Heathrow Airport",
    city: "London",
    country: "United Kingdom",
    country_code: "GB",
    icao: "EGLL",
    iata: "LHR",
    D_to_D: "60 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "75-90 min",
    d_to_d_min: 60,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "BA Terminal 5 raised to 75 minutes in 2024; T3 to T5 requires Border Force and bus transfer."
  },
  CDG: {
    airport: "Charles de Gaulle Airport",
    city: "Paris",
    country: "France",
    country_code: "FR",
    icao: "LFPG",
    iata: "CDG",
    D_to_D: "60 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "90 min+",
    d_to_d_min: 60,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "10,160 carrier exceptions; Terminal 2 sub-terminals vary significantly."
  },
  AMS: {
    airport: "Amsterdam Airport Schiphol",
    city: "Amsterdam",
    country: "Netherlands",
    country_code: "NL",
    icao: "EHAM",
    iata: "AMS",
    D_to_D: "40 min",
    D_to_I: "40 min",
    I_to_D: "40 min",
    I_to_I: "40 min",
    d_to_d_min: 40,
    d_to_i_min: 40,
    i_to_d_min: 40,
    i_to_i_min: 40,
    notes: "Single-terminal design; highly integrated hub."
  },
  FRA: {
    airport: "Frankfurt Airport",
    city: "Frankfurt",
    country: "Germany",
    country_code: "DE",
    icao: "EDDF",
    iata: "FRA",
    D_to_D: "45 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 45,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "Schengen-to-Schengen connections can be as low as 45 minutes; source-provided AirHelp risk score 4.05/10."
  },
  DXB: {
    airport: "Dubai International Airport",
    city: "Dubai",
    country: "United Arab Emirates",
    country_code: "AE",
    icao: "OMDB",
    iata: "DXB",
    D_to_D: "20 min",
    D_to_I: "60 min",
    I_to_D: "60 min",
    I_to_I: "60-75 min",
    d_to_d_min: 20,
    d_to_i_min: 60,
    i_to_d_min: 60,
    i_to_i_min: 75,
    notes: "T1 and T3 have airside train connections; T2 requires external transfer."
  },
  ATL: {
    airport: "Hartsfield-Jackson Atlanta International Airport",
    city: "Atlanta",
    country: "United States",
    country_code: "US",
    icao: "KATL",
    iata: "ATL",
    D_to_D: "60 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "120 min",
    d_to_d_min: 60,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 120,
    notes: "Single terminal complex with underground train between concourses."
  },
  ORD: {
    airport: "O'Hare International Airport",
    city: "Chicago",
    country: "United States",
    country_code: "US",
    icao: "KORD",
    iata: "ORD",
    D_to_D: "40 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 40,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "Source-provided statistic: 19% of flights delayed 30+ minutes during peak periods."
  },
  DFW: {
    airport: "Dallas Fort Worth International Airport",
    city: "Dallas-Fort Worth",
    country: "United States",
    country_code: "US",
    icao: "KDFW",
    iata: "DFW",
    D_to_D: "40 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 40,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "Source-provided AirHelp ranking: second riskiest globally, score 4.03/10, May 2026."
  },
  MIA: {
    airport: "Miami International Airport",
    city: "Miami",
    country: "United States",
    country_code: "US",
    icao: "KMIA",
    iata: "MIA",
    D_to_D: "55-60 min",
    D_to_I: "60 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 60,
    d_to_i_min: 60,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "American Airlines filed a 55-minute exception versus standard 60 minutes."
  },
  MUC: {
    airport: "Munich Airport",
    city: "Munich",
    country: "Germany",
    country_code: "DE",
    icao: "EDDM",
    iata: "MUC",
    D_to_D: "35 min",
    D_to_I: "35 min",
    I_to_D: "35 min",
    I_to_I: "35 min",
    d_to_d_min: 35,
    d_to_i_min: 35,
    i_to_d_min: 35,
    i_to_i_min: 35,
    notes: "Highly integrated layout; source-provided AirHelp risk score 4.33/10."
  },
  SIN: {
    airport: "Singapore Changi Airport",
    city: "Singapore",
    country: "Singapore",
    country_code: "SG",
    icao: "WSSS",
    iata: "SIN",
    D_to_D: "30 min",
    D_to_I: "60 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 30,
    d_to_i_min: 60,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "Highly integrated hub with multiple terminals."
  },
  HND: {
    airport: "Tokyo Haneda Airport",
    city: "Tokyo",
    country: "Japan",
    country_code: "JP",
    icao: "RJTT",
    iata: "HND",
    D_to_D: "30 min",
    D_to_I: "120 min",
    I_to_D: "120 min",
    I_to_I: "120 min",
    d_to_d_min: 30,
    d_to_i_min: 120,
    i_to_d_min: 120,
    i_to_i_min: 120,
    notes: "Separate domestic and international terminal facilities; transfer may require monorail or other transport."
  },
  NRT: {
    airport: "Narita International Airport",
    city: "Tokyo",
    country: "Japan",
    country_code: "JP",
    icao: "RJAA",
    iata: "NRT",
    D_to_D: "30 min",
    D_to_I: "90+ min",
    I_to_D: "90+ min",
    I_to_I: "45 min*",
    d_to_d_min: 30,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 45,
    notes: "*45 minutes applies to same-terminal connections for eligible Star Alliance itineraries; T1 to T2 airside transfer may be possible in eligible cases."
  },
  BKK: {
    airport: "Suvarnabhumi Airport",
    city: "Bangkok",
    country: "Thailand",
    country_code: "TH",
    icao: "VTBS",
    iata: "BKK",
    D_to_D: "75 min",
    D_to_I: "75 min",
    I_to_D: "75 min",
    I_to_I: "75 min",
    d_to_d_min: 75,
    d_to_i_min: 75,
    i_to_d_min: 75,
    i_to_i_min: 75,
    notes: "Uniform 75-minute MCT for Thai Airways, according to supplied data."
  },
  DEN: {
    airport: "Denver International Airport",
    city: "Denver",
    country: "United States",
    country_code: "US",
    icao: "KDEN",
    iata: "DEN",
    D_to_D: "40 min",
    D_to_I: "90 min",
    I_to_D: "90 min",
    I_to_I: "90 min",
    d_to_d_min: 40,
    d_to_i_min: 90,
    i_to_d_min: 90,
    i_to_i_min: 90,
    notes: "Weather delays are a significant operational factor, especially during winter."
  },
  SEA: {
    airport: "Seattle-Tacoma International Airport",
    city: "Seattle",
    country: "United States",
    country_code: "US",
    icao: "KSEA",
    iata: "SEA",
    D_to_D: "30 min",
    D_to_I: "30 min",
    I_to_D: "70 min",
    I_to_I: "70 min",
    d_to_d_min: 30,
    d_to_i_min: 30,
    i_to_d_min: 70,
    i_to_i_min: 70,
    notes: "International arrivals use a separate satellite facility."
  },
  BOS: {
    airport: "Boston Logan International Airport",
    city: "Boston",
    country: "United States",
    country_code: "US",
    icao: "KBOS",
    iata: "BOS",
    D_to_D: "30 min",
    D_to_I: "40 min",
    I_to_D: "80 min",
    I_to_I: "75 min",
    d_to_d_min: 30,
    d_to_i_min: 40,
    i_to_d_min: 80,
    i_to_i_min: 75,
    notes: "Terminal changes require landside transfer between some carriers."
  },
  SFO: {
    airport: "San Francisco International Airport",
    city: "San Francisco",
    country: "United States",
    country_code: "US",
    icao: "KSFO",
    iata: "SFO",
    D_to_D: "50 min",
    D_to_I: "60 min",
    I_to_D: "105 min",
    I_to_I: "105 min",
    d_to_d_min: 50,
    d_to_i_min: 60,
    i_to_d_min: 105,
    i_to_i_min: 105,
    notes: "International Terminal is separate; AirTrain and TSA re-screening may be required."
  },
  EWR: {
    airport: "Newark Liberty International Airport",
    city: "Newark",
    country: "United States",
    country_code: "US",
    icao: "KEWR",
    iata: "EWR",
    D_to_D: "60 min",
    D_to_I: "75 min",
    I_to_D: "90 min",
    I_to_I: "60 min",
    d_to_d_min: 60,
    d_to_i_min: 75,
    i_to_d_min: 90,
    i_to_i_min: 60,
    notes: "AirTrain between terminals adds approximately 10-15 minutes plus TSA screening."
  }
};

export function getHubMCTProfile(code: string): HubMCTEntry | null {
  if (!code) return null;
  const clean = code.trim().toUpperCase();
  // Direct match by IATA
  if (GLOBAL_HUB_MCT_DATA[clean]) {
    return GLOBAL_HUB_MCT_DATA[clean];
  }
  // Match by ICAO
  for (const entry of Object.values(GLOBAL_HUB_MCT_DATA)) {
    if (entry.icao === clean) {
      return entry;
    }
  }
  return null;
}
