import React from 'react';
import { AlertTriangle, CheckCircle, Clock, Plane, TrendingUp, Download, Plus } from 'lucide-react';
import { PredictionResponse, RISK_COLORS, AIRLINES } from '../types';
import ProbabilityChart from './ProbabilityChart';
import { exportToPdf } from '../utils/exportPdf';

interface PredictionResultProps {
  prediction: PredictionResponse;
  onAddToCompare?: () => void;
  showCompareButton?: boolean;
}

const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  onAddToCompare,
  showCompareButton = true
}) => {
  const riskColor = RISK_COLORS[prediction.risk_level];

  const getRiskIcon = () => {
    switch (prediction.risk_level) {
      case 'LOW':
        return <CheckCircle size={48} color={riskColor} />;
      case 'MEDIUM':
      case 'HIGH':
      case 'SEVERE':
        return <AlertTriangle size={48} color={riskColor} />;
      default:
        return <Clock size={48} />;
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

  const handleExportPdf = () => {
    exportToPdf(prediction);
  };

  return (
    <div className="prediction-result">
      <div className="result-actions">
        <button className="action-btn" onClick={handleExportPdf} title="Export to PDF">
          <Download size={18} />
          <span>Export PDF</span>
        </button>
        {showCompareButton && onAddToCompare && (
          <button className="action-btn compare" onClick={onAddToCompare} title="Add to comparison">
            <Plus size={18} />
            <span>Compare</span>
          </button>
        )}
      </div>

      <div className="result-header" style={{ borderColor: riskColor }}>
        <div className="risk-icon">
          {getRiskIcon()}
        </div>
        <div className="risk-info">
          <h2 style={{ color: riskColor }}>
            {prediction.is_delayed ? 'DELAYED' : 'ON TIME'}
          </h2>
          <p className="risk-level" style={{ backgroundColor: riskColor }}>
            {prediction.risk_level} RISK
          </p>
        </div>
      </div>

      <div className="flight-info">
        <div className="route">
          <span className="airport">{prediction.origin_airport}</span>
          <Plane size={20} className="plane-icon" />
          <span className="airport">{prediction.dest_airport}</span>
        </div>
        <div className="flight-details">
          <span>{AIRLINES[prediction.airline] || prediction.airline}</span>
          <span>{prediction.flight_date}</span>
          <span>{prediction.scheduled_departure}</span>
        </div>
      </div>

      <ProbabilityChart
        probability={prediction.delay_probability}
        riskLevel={prediction.risk_level}
      />

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Delay Probability</div>
          <div className="stat-value">
            <TrendingUp size={20} />
            {(prediction.delay_probability * 100).toFixed(1)}%
          </div>
          <div className="probability-bar">
            <div
              className="probability-fill"
              style={{
                width: `${prediction.delay_probability * 100}%`,
                backgroundColor: riskColor
              }}
            />
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Estimated Delay</div>
          <div className="stat-value">
            <Clock size={20} />
            {prediction.estimated_delay_minutes
              ? `${Math.round(prediction.estimated_delay_minutes)} min`
              : 'N/A'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Confidence</div>
          <div className="stat-value">
            {(prediction.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="risk-message" style={{ borderLeftColor: riskColor }}>
        <p>{getRiskMessage()}</p>
      </div>

      {prediction.explanation && (
        <div className="explanation">
          <h3>Why this prediction?</h3>
          <div className="factors">
            {prediction.explanation.features.map((feature, index) => (
              <div
                key={index}
                className={`factor ${feature.shap_value > 0 ? 'positive' : 'negative'}`}
              >
                <span className="factor-name">{feature.name}</span>
                <span className="factor-impact">{feature.impact}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;
