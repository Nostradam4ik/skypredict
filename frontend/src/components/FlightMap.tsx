import React, { memo } from 'react';
import {
  ComposableMap,
  Geographies,
  Geography,
  Line,
  Marker
} from 'react-simple-maps';
import { AIRPORT_COORDS } from '../types';

interface FlightMapProps {
  origin: string;
  destination: string;
  riskLevel: string;
}

const geoUrl = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json';

const FlightMap: React.FC<FlightMapProps> = memo(({ origin, destination, riskLevel }) => {
  const originCoords = AIRPORT_COORDS[origin];
  const destCoords = AIRPORT_COORDS[destination];

  if (!originCoords || !destCoords) {
    return null;
  }

  const getRiskColor = () => {
    switch (riskLevel) {
      case 'LOW': return '#22c55e';
      case 'MEDIUM': return '#f59e0b';
      case 'HIGH': return '#ef4444';
      case 'SEVERE': return '#7c2d12';
      default: return '#3b82f6';
    }
  };

  // Calculate center point for map projection
  const centerLon = (originCoords.lon + destCoords.lon) / 2;
  const centerLat = (originCoords.lat + destCoords.lat) / 2;

  // Calculate scale based on distance
  const distance = Math.sqrt(
    Math.pow(destCoords.lon - originCoords.lon, 2) +
    Math.pow(destCoords.lat - originCoords.lat, 2)
  );
  const scale = Math.max(100, Math.min(400, 800 / distance));

  return (
    <div className="flight-map">
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{
          center: [centerLon, centerLat],
          scale: scale
        }}
        style={{ width: '100%', height: '280px' }}
      >
        <Geographies geography={geoUrl}>
          {({ geographies }) =>
            geographies.map((geo) => (
              <Geography
                key={geo.rsmKey}
                geography={geo}
                fill="var(--bg-input)"
                stroke="var(--border)"
                strokeWidth={0.5}
                style={{
                  default: { outline: 'none' },
                  hover: { outline: 'none' },
                  pressed: { outline: 'none' }
                }}
              />
            ))
          }
        </Geographies>

        {/* Flight path */}
        <Line
          from={[originCoords.lon, originCoords.lat]}
          to={[destCoords.lon, destCoords.lat]}
          stroke={getRiskColor()}
          strokeWidth={2}
          strokeLinecap="round"
          strokeDasharray="5,3"
        />

        {/* Origin marker */}
        <Marker coordinates={[originCoords.lon, originCoords.lat]}>
          <circle r={6} fill={getRiskColor()} stroke="#fff" strokeWidth={2} />
          <text
            textAnchor="middle"
            y={-12}
            style={{
              fontFamily: 'Inter, sans-serif',
              fill: 'var(--text-primary)',
              fontSize: '10px',
              fontWeight: 600
            }}
          >
            {origin}
          </text>
        </Marker>

        {/* Destination marker */}
        <Marker coordinates={[destCoords.lon, destCoords.lat]}>
          <circle r={6} fill={getRiskColor()} stroke="#fff" strokeWidth={2} />
          <text
            textAnchor="middle"
            y={-12}
            style={{
              fontFamily: 'Inter, sans-serif',
              fill: 'var(--text-primary)',
              fontSize: '10px',
              fontWeight: 600
            }}
          >
            {destination}
          </text>
        </Marker>

        {/* Plane icon in the middle */}
        <Marker coordinates={[centerLon, centerLat]}>
          <g transform="translate(-8, -8)">
            <path
              d="M16 8l-3-6H9l1.5 6H6L4.5 6H3l1 4-1 4h1.5L6 12h4.5L9 18h4l3-6h2a2 2 0 000-4h-2z"
              fill={getRiskColor()}
            />
          </g>
        </Marker>
      </ComposableMap>
    </div>
  );
});

export default FlightMap;
