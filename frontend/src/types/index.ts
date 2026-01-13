// API Types

export interface FlightInput {
  airline: string;
  origin_airport: string;
  dest_airport: string;
  flight_date: string;
  scheduled_departure_hour: number;
  scheduled_departure_minute?: number;
  distance?: number;
  scheduled_duration?: number;
  origin_temp_c?: number;
  origin_wind_speed_ms?: number;
  origin_visibility_m?: number;
  origin_precipitation_mm?: number;
  dest_temp_c?: number;
  dest_wind_speed_ms?: number;
  dest_visibility_m?: number;
  dest_precipitation_mm?: number;
}

export interface PredictionResponse {
  is_delayed: boolean;
  delay_probability: number;
  estimated_delay_minutes: number | null;
  confidence: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'SEVERE';
  airline: string;
  origin_airport: string;
  dest_airport: string;
  flight_date: string;
  scheduled_departure: string;
  explanation?: PredictionExplanation;
}

export interface PredictionExplanation {
  base_value: number;
  features: FeatureExplanation[];
}

export interface FeatureExplanation {
  name: string;
  value: number;
  shap_value: number;
  impact: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  model_info: {
    classifier_loaded: boolean;
    regressor_loaded: boolean;
    model_version: string;
    features_count: number;
  };
  uptime_seconds: number;
}

export interface AirportsResponse {
  airports: string[];
  count: number;
}

export interface AirlinesResponse {
  airlines: string[];
  count: number;
}

// UI Types
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'SEVERE';

export const RISK_COLORS: Record<RiskLevel, string> = {
  LOW: '#22c55e',
  MEDIUM: '#f59e0b',
  HIGH: '#ef4444',
  SEVERE: '#7c2d12'
};

export const AIRLINES: Record<string, string> = {
  // US Airlines
  'AA': 'American Airlines (US)',
  'DL': 'Delta Air Lines (US)',
  'UA': 'United Airlines (US)',
  'WN': 'Southwest Airlines (US)',
  'AS': 'Alaska Airlines (US)',
  'B6': 'JetBlue Airways (US)',
  'NK': 'Spirit Airlines (US)',
  'F9': 'Frontier Airlines (US)',
  // European Airlines
  'AF': 'Air France (FR)',
  'BA': 'British Airways (UK)',
  'LH': 'Lufthansa (DE)',
  'KL': 'KLM Royal Dutch (NL)',
  'IB': 'Iberia (ES)',
  'AZ': 'ITA Airways (IT)',
  'SK': 'SAS Scandinavian (SE)',
  'LX': 'Swiss International (CH)',
  'OS': 'Austrian Airlines (AT)',
  'AY': 'Finnair (FI)',
  'TP': 'TAP Portugal (PT)',
  'EI': 'Aer Lingus (IE)',
  'U2': 'easyJet (UK)',
  'FR': 'Ryanair (IE)',
  'VY': 'Vueling (ES)',
  // Middle East Airlines
  'EK': 'Emirates (UAE)',
  'QR': 'Qatar Airways (QA)',
  'EY': 'Etihad Airways (UAE)',
  'TK': 'Turkish Airlines (TR)',
  'SV': 'Saudia (SA)',
  // Asian Airlines
  'CX': 'Cathay Pacific (HK)',
  'SQ': 'Singapore Airlines (SG)',
  'JL': 'Japan Airlines (JP)',
  'NH': 'ANA All Nippon (JP)',
  'KE': 'Korean Air (KR)',
  'OZ': 'Asiana Airlines (KR)',
  'CI': 'China Airlines (TW)',
  'BR': 'EVA Air (TW)',
  'MH': 'Malaysia Airlines (MY)',
  'TG': 'Thai Airways (TH)',
  'VN': 'Vietnam Airlines (VN)',
  'AI': 'Air India (IN)',
  'GA': 'Garuda Indonesia (ID)',
  // Americas (non-US)
  'AC': 'Air Canada (CA)',
  'WS': 'WestJet (CA)',
  'AM': 'Aeromexico (MX)',
  'LA': 'LATAM Airlines (CL)',
  'AV': 'Avianca (CO)',
  'CM': 'Copa Airlines (PA)',
  'G3': 'Gol Linhas (BR)',
  'AD': 'Azul Brazilian (BR)',
  // Oceania
  'QF': 'Qantas (AU)',
  'VA': 'Virgin Australia (AU)',
  'NZ': 'Air New Zealand (NZ)'
};

// Airport coordinates for distance calculation
export const AIRPORT_COORDS: Record<string, { lat: number; lon: number }> = {
  // USA
  'ATL': { lat: 33.6407, lon: -84.4277 },
  'DFW': { lat: 32.8998, lon: -97.0403 },
  'DEN': { lat: 39.8561, lon: -104.6737 },
  'ORD': { lat: 41.9742, lon: -87.9073 },
  'LAX': { lat: 33.9416, lon: -118.4085 },
  'JFK': { lat: 40.6413, lon: -73.7781 },
  'SFO': { lat: 37.6213, lon: -122.3790 },
  'MIA': { lat: 25.7959, lon: -80.2870 },
  'SEA': { lat: 47.4502, lon: -122.3088 },
  'BOS': { lat: 42.3656, lon: -71.0096 },
  'EWR': { lat: 40.6895, lon: -74.1745 },
  'LAS': { lat: 36.0840, lon: -115.1537 },
  // Europe
  'LHR': { lat: 51.4700, lon: -0.4543 },
  'LGW': { lat: 51.1537, lon: -0.1821 },
  'CDG': { lat: 49.0097, lon: 2.5479 },
  'ORY': { lat: 48.7262, lon: 2.3652 },
  'FRA': { lat: 50.0379, lon: 8.5622 },
  'MUC': { lat: 48.3537, lon: 11.7750 },
  'AMS': { lat: 52.3105, lon: 4.7683 },
  'MAD': { lat: 40.4983, lon: -3.5676 },
  'BCN': { lat: 41.2974, lon: 2.0833 },
  'FCO': { lat: 41.8003, lon: 12.2389 },
  'MXP': { lat: 45.6306, lon: 8.7281 },
  'ZRH': { lat: 47.4647, lon: 8.5492 },
  'VIE': { lat: 48.1103, lon: 16.5697 },
  'BRU': { lat: 50.9014, lon: 4.4844 },
  'LIS': { lat: 38.7756, lon: -9.1354 },
  'DUB': { lat: 53.4264, lon: -6.2499 },
  'CPH': { lat: 55.6180, lon: 12.6508 },
  'OSL': { lat: 60.1976, lon: 11.0004 },
  'ARN': { lat: 59.6519, lon: 17.9186 },
  'HEL': { lat: 60.3172, lon: 24.9633 },
  'IST': { lat: 41.2753, lon: 28.7519 },
  // Middle East
  'DXB': { lat: 25.2532, lon: 55.3657 },
  'AUH': { lat: 24.4330, lon: 54.6511 },
  'DOH': { lat: 25.2731, lon: 51.6081 },
  'JED': { lat: 21.6796, lon: 39.1565 },
  'RUH': { lat: 24.9578, lon: 46.6989 },
  'TLV': { lat: 32.0055, lon: 34.8854 },
  // Asia
  'HKG': { lat: 22.3080, lon: 113.9185 },
  'SIN': { lat: 1.3644, lon: 103.9915 },
  'NRT': { lat: 35.7720, lon: 140.3929 },
  'HND': { lat: 35.5494, lon: 139.7798 },
  'ICN': { lat: 37.4602, lon: 126.4407 },
  'PEK': { lat: 40.0799, lon: 116.6031 },
  'PVG': { lat: 31.1443, lon: 121.8083 },
  'TPE': { lat: 25.0797, lon: 121.2342 },
  'BKK': { lat: 13.6900, lon: 100.7501 },
  'KUL': { lat: 2.7456, lon: 101.7072 },
  'CGK': { lat: -6.1256, lon: 106.6559 },
  'DEL': { lat: 28.5562, lon: 77.1000 },
  'BOM': { lat: 19.0896, lon: 72.8656 },
  'SGN': { lat: 10.8188, lon: 106.6520 },
  // Americas
  'YYZ': { lat: 43.6777, lon: -79.6248 },
  'YVR': { lat: 49.1967, lon: -123.1815 },
  'YUL': { lat: 45.4706, lon: -73.7408 },
  'MEX': { lat: 19.4363, lon: -99.0721 },
  'CUN': { lat: 21.0365, lon: -86.8771 },
  'GRU': { lat: -23.4356, lon: -46.4731 },
  'GIG': { lat: -22.8100, lon: -43.2505 },
  'EZE': { lat: -34.8222, lon: -58.5358 },
  'SCL': { lat: -33.3930, lon: -70.7858 },
  'BOG': { lat: 4.7016, lon: -74.1469 },
  'LIM': { lat: -12.0219, lon: -77.1143 },
  'PTY': { lat: 9.0714, lon: -79.3835 },
  // Oceania
  'SYD': { lat: -33.9399, lon: 151.1753 },
  'MEL': { lat: -37.6690, lon: 144.8410 },
  'BNE': { lat: -27.3942, lon: 153.1218 },
  'AKL': { lat: -37.0082, lon: 174.7850 },
};

// Calculate distance between two airports
export const calculateDistance = (origin: string, dest: string, unit: 'km' | 'miles' = 'km'): number => {
  const orig = AIRPORT_COORDS[origin];
  const dst = AIRPORT_COORDS[dest];

  if (!orig || !dst) return 0;

  const R = unit === 'km' ? 6371 : 3959; // Earth radius in km or miles
  const dLat = (dst.lat - orig.lat) * Math.PI / 180;
  const dLon = (dst.lon - orig.lon) * Math.PI / 180;
  const a =
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(orig.lat * Math.PI / 180) * Math.cos(dst.lat * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return Math.round(R * c);
};

export const AIRPORTS: Record<string, string> = {
  // USA
  'ATL': 'Atlanta (US)',
  'DFW': 'Dallas/Fort Worth (US)',
  'DEN': 'Denver (US)',
  'ORD': 'Chicago O\'Hare (US)',
  'LAX': 'Los Angeles (US)',
  'JFK': 'New York JFK (US)',
  'SFO': 'San Francisco (US)',
  'MIA': 'Miami (US)',
  'SEA': 'Seattle (US)',
  'BOS': 'Boston (US)',
  'EWR': 'Newark (US)',
  'LAS': 'Las Vegas (US)',
  // Europe
  'LHR': 'London Heathrow (UK)',
  'LGW': 'London Gatwick (UK)',
  'CDG': 'Paris CDG (FR)',
  'ORY': 'Paris Orly (FR)',
  'FRA': 'Frankfurt (DE)',
  'MUC': 'Munich (DE)',
  'AMS': 'Amsterdam (NL)',
  'MAD': 'Madrid (ES)',
  'BCN': 'Barcelona (ES)',
  'FCO': 'Rome Fiumicino (IT)',
  'MXP': 'Milan Malpensa (IT)',
  'ZRH': 'Zurich (CH)',
  'VIE': 'Vienna (AT)',
  'BRU': 'Brussels (BE)',
  'LIS': 'Lisbon (PT)',
  'DUB': 'Dublin (IE)',
  'CPH': 'Copenhagen (DK)',
  'OSL': 'Oslo (NO)',
  'ARN': 'Stockholm (SE)',
  'HEL': 'Helsinki (FI)',
  'IST': 'Istanbul (TR)',
  // Middle East
  'DXB': 'Dubai (UAE)',
  'AUH': 'Abu Dhabi (UAE)',
  'DOH': 'Doha (QA)',
  'JED': 'Jeddah (SA)',
  'RUH': 'Riyadh (SA)',
  'TLV': 'Tel Aviv (IL)',
  // Asia
  'HKG': 'Hong Kong (HK)',
  'SIN': 'Singapore (SG)',
  'NRT': 'Tokyo Narita (JP)',
  'HND': 'Tokyo Haneda (JP)',
  'ICN': 'Seoul Incheon (KR)',
  'PEK': 'Beijing (CN)',
  'PVG': 'Shanghai (CN)',
  'TPE': 'Taipei (TW)',
  'BKK': 'Bangkok (TH)',
  'KUL': 'Kuala Lumpur (MY)',
  'CGK': 'Jakarta (ID)',
  'DEL': 'New Delhi (IN)',
  'BOM': 'Mumbai (IN)',
  'SGN': 'Ho Chi Minh (VN)',
  // Americas
  'YYZ': 'Toronto (CA)',
  'YVR': 'Vancouver (CA)',
  'YUL': 'Montreal (CA)',
  'MEX': 'Mexico City (MX)',
  'CUN': 'Cancun (MX)',
  'GRU': 'Sao Paulo (BR)',
  'GIG': 'Rio de Janeiro (BR)',
  'EZE': 'Buenos Aires (AR)',
  'SCL': 'Santiago (CL)',
  'BOG': 'Bogota (CO)',
  'LIM': 'Lima (PE)',
  'PTY': 'Panama City (PA)',
  // Oceania
  'SYD': 'Sydney (AU)',
  'MEL': 'Melbourne (AU)',
  'BNE': 'Brisbane (AU)',
  'AKL': 'Auckland (NZ)'
};
