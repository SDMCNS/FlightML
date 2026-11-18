"""
Global Hub Minimum Connection Times (MCT) Database & Connection Intelligence.

Provides standardized and carrier-specific minimum connection time profiles across
major international and domestic connecting hubs:
- Domestic to Domestic (D_to_D)
- Domestic to International (D_to_I)
- International to Domestic (I_to_D: immigration, customs, TSA re-screening)
- International to International (I_to_I: transit security & border checkpoints)
- Operational transfer notes and terminal transit requirements
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Literal

ConnectionType = Literal["D_to_D", "D_to_I", "I_to_D", "I_to_I"]


@dataclass
class HubMCTProfile:
    iata: str
    icao: str
    airport: str
    city: str
    country: str
    country_code: str
    d_to_d_min: int
    d_to_i_min: int
    i_to_d_min: int
    i_to_i_min: int
    notes: str
    raw_strings: dict[str, str]
    is_placeholder: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_mct_for_type(self, conn_type: ConnectionType) -> int:
        mapping = {
            "D_to_D": self.d_to_d_min,
            "D_to_I": self.d_to_i_min,
            "I_to_D": self.i_to_d_min,
            "I_to_I": self.i_to_i_min,
        }
        return mapping.get(conn_type, self.d_to_d_min)


def _parse_mct_minutes(val: str) -> int:
    """Parse string representation (e.g. '120-135 min', '90 min+', '45 min*') into conservative nominal minutes."""
    nums = [int(n) for n in re.findall(r"\d+", val)]
    if not nums:
        return 45
    return max(nums)


RAW_HUB_MCT_DATA: dict[str, dict[str, Any]] = {
    "JFK": {
        "airport": "John F. Kennedy International Airport",
        "city": "New York",
        "country": "United States",
        "country_code": "US",
        "icao": "KJFK",
        "D_to_D": "45 min",
        "D_to_I": "90 min",
        "I_to_D": "120-135 min",
        "I_to_I": "90 min",
        "notes": "No airside terminal connections; AirTrain and TSA re-screening required.",
    },
    "LAX": {
        "airport": "Los Angeles International Airport",
        "city": "Los Angeles",
        "country": "United States",
        "country_code": "US",
        "icao": "KLAX",
        "D_to_D": "45 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "Inter-terminal transfers require landside shuttle and TSA re-screening.",
    },
    "LHR": {
        "airport": "London Heathrow Airport",
        "city": "London",
        "country": "United Kingdom",
        "country_code": "GB",
        "icao": "EGLL",
        "D_to_D": "60 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "75-90 min",
        "notes": "BA Terminal 5 raised to 75 minutes in 2024; T3 to T5 requires Border Force and bus transfer.",
    },
    "CDG": {
        "airport": "Charles de Gaulle Airport",
        "city": "Paris",
        "country": "France",
        "country_code": "FR",
        "icao": "LFPG",
        "D_to_D": "60 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min+",
        "notes": "10,160 carrier exceptions; Terminal 2 sub-terminals vary significantly.",
    },
    "AMS": {
        "airport": "Amsterdam Airport Schiphol",
        "city": "Amsterdam",
        "country": "Netherlands",
        "country_code": "NL",
        "icao": "EHAM",
        "D_to_D": "40 min",
        "D_to_I": "40 min",
        "I_to_D": "40 min",
        "I_to_I": "40 min",
        "notes": "Single-terminal design; highly integrated hub.",
    },
    "FRA": {
        "airport": "Frankfurt Airport",
        "city": "Frankfurt",
        "country": "Germany",
        "country_code": "DE",
        "icao": "EDDF",
        "D_to_D": "45 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "Schengen-to-Schengen connections can be as low as 45 minutes; source-provided AirHelp risk score 4.05/10.",
    },
    "DXB": {
        "airport": "Dubai International Airport",
        "city": "Dubai",
        "country": "United Arab Emirates",
        "country_code": "AE",
        "icao": "OMDB",
        "D_to_D": "20 min",
        "D_to_I": "60 min",
        "I_to_D": "60 min",
        "I_to_I": "60-75 min",
        "notes": "T1 and T3 have airside train connections; T2 requires external transfer.",
    },
    "ATL": {
        "airport": "Hartsfield-Jackson Atlanta International Airport",
        "city": "Atlanta",
        "country": "United States",
        "country_code": "US",
        "icao": "KATL",
        "D_to_D": "60 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "120 min",
        "notes": "Single terminal complex with underground train between concourses.",
    },
    "ORD": {
        "airport": "O'Hare International Airport",
        "city": "Chicago",
        "country": "United States",
        "country_code": "US",
        "icao": "KORD",
        "D_to_D": "40 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "Source-provided statistic: 19% of flights delayed 30+ minutes during peak periods.",
    },
    "DFW": {
        "airport": "Dallas Fort Worth International Airport",
        "city": "Dallas-Fort Worth",
        "country": "United States",
        "country_code": "US",
        "icao": "KDFW",
        "D_to_D": "40 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "Source-provided AirHelp ranking: second riskiest globally, score 4.03/10, May 2026.",
    },
    "MIA": {
        "airport": "Miami International Airport",
        "city": "Miami",
        "country": "United States",
        "country_code": "US",
        "icao": "KMIA",
        "D_to_D": "55-60 min",
        "D_to_I": "60 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "American Airlines filed a 55-minute exception versus standard 60 minutes.",
    },
    "MUC": {
        "airport": "Munich Airport",
        "city": "Munich",
        "country": "Germany",
        "country_code": "DE",
        "icao": "EDDM",
        "D_to_D": "35 min",
        "D_to_I": "35 min",
        "I_to_D": "35 min",
        "I_to_I": "35 min",
        "notes": "Highly integrated layout; source-provided AirHelp risk score 4.33/10.",
    },
    "SIN": {
        "airport": "Singapore Changi Airport",
        "city": "Singapore",
        "country": "Singapore",
        "country_code": "SG",
        "icao": "WSSS",
        "D_to_D": "30 min",
        "D_to_I": "60 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "Highly integrated hub with multiple terminals.",
    },
    "HND": {
        "airport": "Tokyo Haneda Airport",
        "city": "Tokyo",
        "country": "Japan",
        "country_code": "JP",
        "icao": "RJTT",
        "D_to_D": "30 min",
        "D_to_I": "120 min",
        "I_to_D": "120 min",
        "I_to_I": "120 min",
        "notes": "Separate domestic and international terminal facilities; transfer may require monorail or other transport.",
    },
    "NRT": {
        "airport": "Narita International Airport",
        "city": "Tokyo",
        "country": "Japan",
        "country_code": "JP",
        "icao": "RJAA",
        "D_to_D": "30 min",
        "D_to_I": "90+ min",
        "I_to_D": "90+ min",
        "I_to_I": "45 min*",
        "notes": "*45 minutes applies to same-terminal connections for eligible Star Alliance itineraries; T1 to T2 airside transfer may be possible in eligible cases.",
    },
    "BKK": {
        "airport": "Suvarnabhumi Airport",
        "city": "Bangkok",
        "country": "Thailand",
        "country_code": "TH",
        "icao": "VTBS",
        "D_to_D": "75 min",
        "D_to_I": "75 min",
        "I_to_D": "75 min",
        "I_to_I": "75 min",
        "notes": "Uniform 75-minute MCT for Thai Airways, according to supplied data.",
    },
    "DEN": {
        "airport": "Denver International Airport",
        "city": "Denver",
        "country": "United States",
        "country_code": "US",
        "icao": "KDEN",
        "D_to_D": "40 min",
        "D_to_I": "90 min",
        "I_to_D": "90 min",
        "I_to_I": "90 min",
        "notes": "Weather delays are a significant operational factor, especially during winter.",
    },
    "SEA": {
        "airport": "Seattle-Tacoma International Airport",
        "city": "Seattle",
        "country": "United States",
        "country_code": "US",
        "icao": "KSEA",
        "D_to_D": "30 min",
        "D_to_I": "30 min",
        "I_to_D": "70 min",
        "I_to_I": "70 min",
        "notes": "International arrivals use a separate satellite facility.",
    },
    "BOS": {
        "airport": "Boston Logan International Airport",
        "city": "Boston",
        "country": "United States",
        "country_code": "US",
        "icao": "KBOS",
        "D_to_D": "30 min",
        "D_to_I": "40 min",
        "I_to_D": "80 min",
        "I_to_I": "75 min",
        "notes": "Terminal changes require landside transfer between some carriers.",
    },
    "SFO": {
        "airport": "San Francisco International Airport",
        "city": "San Francisco",
        "country": "United States",
        "country_code": "US",
        "icao": "KSFO",
        "D_to_D": "50 min",
        "D_to_I": "60 min",
        "I_to_D": "105 min",
        "I_to_I": "105 min",
        "notes": "International Terminal is separate; AirTrain and TSA re-screening may be required.",
    },
    "EWR": {
        "airport": "Newark Liberty International Airport",
        "city": "Newark",
        "country": "United States",
        "country_code": "US",
        "icao": "KEWR",
        "D_to_D": "60 min",
        "D_to_I": "75 min",
        "I_to_D": "90 min",
        "I_to_I": "60 min",
        "notes": "AirTrain between terminals adds approximately 10-15 minutes plus TSA screening.",
    },
    # Additional European Hubs from FlightML Base Database
    "BRU": {
        "airport": "Brussels Airport",
        "city": "Brussels",
        "country": "Belgium",
        "country_code": "BE",
        "icao": "EBBR",
        "D_to_D": "45 min",
        "D_to_I": "50 min",
        "I_to_D": "55 min",
        "I_to_I": "50 min",
        "notes": "Pier A (Schengen) and Pier B (Non-Schengen) connected via connector building with passport control.",
    },
    "MAD": {
        "airport": "Adolfo Suárez Madrid-Barajas Airport",
        "city": "Madrid",
        "country": "Spain",
        "country_code": "ES",
        "icao": "LEMD",
        "D_to_D": "45 min",
        "D_to_I": "65 min",
        "I_to_D": "75 min",
        "I_to_I": "60 min",
        "notes": "Transfers between T1/T2/T3 and T4 require free airport shuttle buses; T4 to T4S uses underground APM train.",
    },
    "DUB": {
        "airport": "Dublin Airport",
        "city": "Dublin",
        "country": "Ireland",
        "country_code": "IE",
        "icao": "EIDW",
        "D_to_D": "45 min",
        "D_to_I": "55 min",
        "I_to_D": "60 min",
        "I_to_I": "50 min",
        "notes": "Terminal 1 and Terminal 2 are connected via an indoor passenger covered walkway.",
    },
}

# Compile lookup index mapping both IATA and ICAO codes
HUB_MCT_INDEX: dict[str, HubMCTProfile] = {}
for iata_code, data in RAW_HUB_MCT_DATA.items():
    profile = HubMCTProfile(
        iata=iata_code,
        icao=data["icao"],
        airport=data["airport"],
        city=data["city"],
        country=data["country"],
        country_code=data["country_code"],
        d_to_d_min=_parse_mct_minutes(data["D_to_D"]),
        d_to_i_min=_parse_mct_minutes(data["D_to_I"]),
        i_to_d_min=_parse_mct_minutes(data["I_to_D"]),
        i_to_i_min=_parse_mct_minutes(data["I_to_I"]),
        notes=data["notes"],
        raw_strings={
            "D_to_D": data["D_to_D"],
            "D_to_I": data["D_to_I"],
            "I_to_D": data["I_to_D"],
            "I_to_I": data["I_to_I"],
        },
    )
    HUB_MCT_INDEX[iata_code] = profile
    HUB_MCT_INDEX[data["icao"]] = profile


def create_placeholder_mct_profile(
    ident: str,
    airport_name: str | None = None,
    city: str | None = None,
    country: str | None = None,
    country_code: str | None = None,
) -> HubMCTProfile:
    """
    Generate common-sense placeholder connection times for any unlisted transit hub:
    - D_to_D: 45 min (standard domestic terminal / same-concourse transfer)
    - D_to_I: 60 min (standard passport exit check & international concourse transfer)
    - I_to_D: 90 min (standard immigration border control, baggage reclaim, and TSA re-screening)
    - I_to_I: 60 min (standard international transit security re-screening)
    """
    clean = ident.strip().upper() if ident else "DEFAULT"
    is_icao = (len(clean) == 4 and clean.isalpha())
    iata = clean if not is_icao else clean[1:]
    icao = clean if is_icao else f"K{clean}"
    name = airport_name or f"Transit Hub ({clean})"
    cit = city or "Transit Hub"
    cntry = country or "Standard Airspace"
    c_code = country_code or "XX"

    return HubMCTProfile(
        iata=iata,
        icao=icao,
        airport=name,
        city=cit,
        country=cntry,
        country_code=c_code,
        d_to_d_min=45,
        d_to_i_min=60,
        i_to_d_min=90,
        i_to_i_min=60,
        notes="Standard industry MCT baseline: 45m Domestic, 60m International transit, 90m Border/Customs re-screening.",
        raw_strings={
            "D_to_D": "45 min",
            "D_to_I": "60 min",
            "I_to_D": "90 min",
            "I_to_I": "60 min",
        },
        is_placeholder=True,
    )


def resolve_hub_mct(ident: str, fallback_to_default: bool = True) -> HubMCTProfile | None:
    """
    Resolve Hub MCT profile by either 3-letter IATA (e.g. 'JFK') or 4-letter ICAO (e.g. 'KJFK').
    If the airport is not in the curated list and fallback_to_default=True,
    returns a standard common-sense placeholder profile (45m/60m/90m/60m).
    """
    if not ident:
        return create_placeholder_mct_profile("DEFAULT") if fallback_to_default else None
    code = ident.strip().upper()
    if code in HUB_MCT_INDEX:
        return HUB_MCT_INDEX[code]
    if fallback_to_default:
        return create_placeholder_mct_profile(code)
    return None


get_hub_mct_profile = resolve_hub_mct

# Direct dictionary mapping for backward compatibility and dashboard views
HUB_MCT_DATABASE: dict[str, dict[str, Any]] = {}
for code_k, prof_v in HUB_MCT_INDEX.items():
    d = prof_v.to_dict()
    raw = d.get("raw_strings") or {}
    d["D_to_D"] = raw.get("D_to_D") or f"{prof_v.d_to_d_min} min"
    d["D_to_I"] = raw.get("D_to_I") or f"{prof_v.d_to_i_min} min"
    d["I_to_D"] = raw.get("I_to_D") or f"{prof_v.i_to_d_min} min"
    d["I_to_I"] = raw.get("I_to_I") or f"{prof_v.i_to_i_min} min"
    HUB_MCT_DATABASE[code_k] = d



def _match_country(c1: str, c2_code: str, c2_name: str) -> bool:
    if not c1:
        return False
    c1_upper = c1.strip().upper()
    return c1_upper == c2_code.strip().upper() or c1_upper == c2_name.strip().upper()


def infer_connection_type(
    origin_country: str | None,
    hub_country: str | None,
    dest_country: str | None,
) -> ConnectionType:
    """
    Infer whether an itinerary connection is:
    - D_to_D: Domestic inbound, Domestic outbound
    - D_to_I: Domestic inbound, International outbound
    - I_to_D: International inbound, Domestic outbound (requires customs, baggage re-check, TSA re-screening)
    - I_to_I: International inbound, International outbound (airside/transit transfer)
    """
    if not origin_country or not hub_country or not dest_country:
        # Default fallback to standard international-to-international transfer
        return "I_to_I"

    o_c = origin_country.strip().upper()
    h_c = hub_country.strip().upper()
    d_c = dest_country.strip().upper()

    inbound_is_domestic = (o_c == h_c)
    outbound_is_domestic = (h_c == d_c)

    if inbound_is_domestic and outbound_is_domestic:
        return "D_to_D"
    elif inbound_is_domestic and not outbound_is_domestic:
        return "D_to_I"
    elif not inbound_is_domestic and outbound_is_domestic:
        return "I_to_D"
    else:
        return "I_to_I"


def get_applicable_mct(
    hub_ident: str,
    conn_type: ConnectionType = "I_to_I",
    origin_country: str | None = None,
    dest_country: str | None = None,
    hub_country: str | None = None,
) -> tuple[int, ConnectionType, HubMCTProfile]:
    """
    Retrieve applicable minimum connection time (MCT) in minutes, connection type,
    and profile for a specified hub. If unlisted, uses common-sense placeholder numbers.
    """
    profile = resolve_hub_mct(hub_ident, fallback_to_default=True)
    if profile is None:
        profile = create_placeholder_mct_profile(hub_ident)

    resolved_type = conn_type
    effective_hub_country = hub_country or (profile.country_code if profile.country_code != "XX" else None)

    if origin_country and dest_country:
        if effective_hub_country:
            inbound_is_domestic = _match_country(origin_country, effective_hub_country, profile.country)
            outbound_is_domestic = _match_country(dest_country, effective_hub_country, profile.country)
            if inbound_is_domestic and outbound_is_domestic:
                resolved_type = "D_to_D"
            elif inbound_is_domestic and not outbound_is_domestic:
                resolved_type = "D_to_I"
            elif not inbound_is_domestic and outbound_is_domestic:
                resolved_type = "I_to_D"
            else:
                resolved_type = "I_to_I"
        else:
            if origin_country.strip().upper() == dest_country.strip().upper():
                resolved_type = "D_to_D"
            else:
                resolved_type = "I_to_I"

    mct_min = profile.get_mct_for_type(resolved_type)
    return mct_min, resolved_type, profile
