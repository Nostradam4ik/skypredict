import React from 'react';
import { X, Plane, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { PredictionResponse } from '../types';

interface FlightComparisonProps {
  flights: PredictionResponse[];
  onRemove: (index: number) => void;
  onClear: () => void;
}

const FlightComparison: React.FC<FlightComparisonProps> = ({
  flights,
  onRemove,
  onClear
}) => {
  if (flights.length < 2) {
    return null;
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return '#22c55e';
      case 'MEDIUM': return '#f59e0b';
      case 'HIGH': return '#ef4444';
      case 'SEVERE': return '#7c2d12';
      default: return '#94a3b8';
    }
  };

  const getBestFlight = () => {
    return flights.reduce((best, current) =>
      current.delay_probability < best.delay_probability ? current : best
    );
  };

  const bestFlight = getBestFlight();

  const getTrend = (flight: PredictionResponse) => {
    if (flight === bestFlight) return 'best';
    const diff = flight.delay_probability - bestFlight.delay_probability;
    if (diff > 0.2) return 'worse';
    return 'similar';
  };

  return (
    <div className="flight-comparison">
      <div className="comparison-header">
        <h3><Plane size={18} /> Flight Comparison</h3>
        <button className="clear-comparison-btn" onClick={onClear}>
          Clear All
        </button>
      </div>

      <div className="comparison-grid">
        {flights.map((flight, index) => {
          const trend = getTrend(flight);
          const isBest = flight === bestFlight;

          return (
            <div
              key={index}
              className={`comparison-card ${isBest ? 'best' : ''}`}
            >
              <button
                className="remove-flight-btn"
                onClick={() => onRemove(index)}
              >
                <X size={14} />
              </button>

              {isBest && <div className="best-badge">Best Option</div>}

              <div className="comparison-route">
                <span className="comp-airport">{flight.origin_airport}</span>
                <Plane size={16} className="comp-plane" />
                <span className="comp-airport">{flight.dest_airport}</span>
              </div>

              <div className="comparison-airline">{flight.airline}</div>

              <div className="comparison-stats">
                <div className="comp-stat">
                  <span className="comp-stat-label">Delay Risk</span>
                  <span
                    className="comp-stat-value"
                    style={{ color: getRiskColor(flight.risk_level) }}
                  >
                    {Math.round(flight.delay_probability * 100)}%
                    {trend === 'best' && <TrendingDown size={14} />}
                    {trend === 'worse' && <TrendingUp size={14} />}
                    {trend === 'similar' && <Minus size={14} />}
                  </span>
                </div>

                <div className="comp-stat">
                  <span className="comp-stat-label">Risk Level</span>
                  <span
                    className="comp-risk-badge"
                    style={{ backgroundColor: getRiskColor(flight.risk_level) }}
                  >
                    {flight.risk_level}
                  </span>
                </div>

                <div className="comp-stat">
                  <span className="comp-stat-label">Est. Delay</span>
                  <span className="comp-stat-value">
                    {flight.estimated_delay_minutes
                      ? `${flight.estimated_delay_minutes} min`
                      : 'N/A'}
                  </span>
                </div>

                <div className="comp-stat">
                  <span className="comp-stat-label">Confidence</span>
                  <span className="comp-stat-value">
                    {Math.round(flight.confidence * 100)}%
                  </span>
                </div>
              </div>

              <div className="comparison-date">
                {flight.flight_date} • {flight.scheduled_departure}
              </div>
            </div>
          );
        })}
      </div>

      {flights.length >= 2 && (
        <div className="comparison-summary">
          <p>
            <strong>Recommendation:</strong> The flight from{' '}
            <span className="highlight">{bestFlight.origin_airport}</span> to{' '}
            <span className="highlight">{bestFlight.dest_airport}</span> with{' '}
            <span className="highlight">{bestFlight.airline}</span> has the lowest
            delay risk ({Math.round(bestFlight.delay_probability * 100)}%).
          </p>
        </div>
      )}
    </div>
  );
};

export default FlightComparison;
