import React from 'react';
import { X, AlertTriangle, CheckCircle, Clock, Plane, TrendingUp, Download, QrCode } from 'lucide-react';
import { PredictionResponse, RISK_COLORS, AIRLINES } from '../types';
import FlightMap from './FlightMap';
import ProbabilityChart from './ProbabilityChart';
import ShareButtons from './ShareButtons';
import { exportToPdf } from '../utils/exportPdf';

interface FullscreenResultProps {
  prediction: PredictionResponse;
  onClose: () => void;
  onShowQR: () => void;
}

const FullscreenResult: React.FC<FullscreenResultProps> = ({
  prediction,
  onClose,
  onShowQR
}) => {
  const riskColor = RISK_COLORS[prediction.risk_level];

  const getRiskIcon = () => {
    switch (prediction.risk_level) {
      case 'LOW':
        return <CheckCircle size={64} color={riskColor} />;
      default:
        return <AlertTriangle size={64} color={riskColor} />;
    }
  };

  const getRiskMessage = () => {
    switch (prediction.risk_level) {
      case 'LOW':
        return 'Low delay risk. Your flight is expected to be on time.';
      case 'MEDIUM':
        return 'Moderate delay risk. Monitor your flight status.';
      case 'HIGH':
        return 'High delay risk. Consider checking alternatives.';
      case 'SEVERE':
        return 'Severe delay risk! Expect significant delays.';
      default:
        return '';
    }
  };

  return (
    <div className="fullscreen-overlay">
      <div className="fullscreen-result">
        <button className="fullscreen-close" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="fullscreen-header" style={{ borderColor: riskColor }}>
          <div className="fullscreen-status">
            {getRiskIcon()}
            <div>
              <h1 style={{ color: riskColor }}>
                {prediction.is_delayed ? 'LIKELY DELAYED' : 'ON TIME'}
              </h1>
              <span className="fullscreen-risk" style={{ backgroundColor: riskColor }}>
                {prediction.risk_level} RISK
              </span>
            </div>
          </div>
        </div>

        <div className="fullscreen-map">
          <FlightMap
            origin={prediction.origin_airport}
            destination={prediction.dest_airport}
            riskLevel={prediction.risk_level}
          />
        </div>

        <div className="fullscreen-route">
          <div className="route-endpoint">
            <span className="route-code">{prediction.origin_airport}</span>
          </div>
          <div className="route-line">
            <Plane size={24} style={{ color: riskColor }} />
          </div>
          <div className="route-endpoint">
            <span className="route-code">{prediction.dest_airport}</span>
          </div>
        </div>

        <div className="fullscreen-details">
          <span>{AIRLINES[prediction.airline] || prediction.airline}</span>
          <span>{prediction.flight_date}</span>
          <span>{prediction.scheduled_departure}</span>
        </div>

        <div className="fullscreen-stats">
          <div className="fullscreen-stat">
            <TrendingUp size={24} style={{ color: riskColor }} />
            <span className="stat-value">{Math.round(prediction.delay_probability * 100)}%</span>
            <span className="stat-label">Delay Probability</span>
          </div>
          <div className="fullscreen-stat">
            <Clock size={24} />
            <span className="stat-value">
              {prediction.estimated_delay_minutes
                ? `${Math.round(prediction.estimated_delay_minutes)} min`
                : 'N/A'}
            </span>
            <span className="stat-label">Est. Delay</span>
          </div>
          <div className="fullscreen-stat">
            <span className="stat-value">{Math.round(prediction.confidence * 100)}%</span>
            <span className="stat-label">Confidence</span>
          </div>
        </div>

        <div className="fullscreen-chart">
          <ProbabilityChart
            probability={prediction.delay_probability}
            riskLevel={prediction.risk_level}
          />
        </div>

        <p className="fullscreen-message" style={{ borderLeftColor: riskColor }}>
          {getRiskMessage()}
        </p>

        <div className="fullscreen-actions">
          <button className="action-btn" onClick={() => exportToPdf(prediction)}>
            <Download size={18} /> Export PDF
          </button>
          <button className="action-btn" onClick={onShowQR}>
            <QrCode size={18} /> QR Code
          </button>
          <ShareButtons prediction={prediction} />
        </div>
      </div>
    </div>
  );
};

export default FullscreenResult;
