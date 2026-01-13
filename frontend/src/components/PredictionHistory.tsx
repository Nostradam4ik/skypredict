import React from 'react';
import { useTranslation } from 'react-i18next';
import { History, Trash2, ChevronRight } from 'lucide-react';
import { PredictionResponse } from '../types';

export interface HistoryItem {
  id: string;
  timestamp: Date;
  prediction: PredictionResponse;
}

interface PredictionHistoryProps {
  history: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  onClear: () => void;
  onDelete: (id: string) => void;
}

const PredictionHistory: React.FC<PredictionHistoryProps> = ({
  history,
  onSelect,
  onClear,
  onDelete
}) => {
  const { t, i18n } = useTranslation();

  if (history.length === 0) {
    return null;
  }

  const locale = i18n.language === 'fr' ? 'fr-FR' : i18n.language === 'es' ? 'es-ES' : 'en-US';

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString(locale, {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString(locale, {
      day: '2-digit',
      month: 'short'
    });
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return '#22c55e';
      case 'MEDIUM': return '#f59e0b';
      case 'HIGH': return '#ef4444';
      case 'SEVERE': return '#7c2d12';
      default: return '#94a3b8';
    }
  };

  return (
    <div className="prediction-history">
      <div className="history-header">
        <h3><History size={18} /> {t('recentPredictions')}</h3>
        <button className="clear-btn" onClick={onClear} title={t('clearHistory')}>
          <Trash2 size={16} />
        </button>
      </div>
      <div className="history-list">
        {history.slice(0, 5).map((item) => (
          <div
            key={item.id}
            className="history-item"
            onClick={() => onSelect(item)}
          >
            <div className="history-route">
              <span className="history-airports">
                {item.prediction.origin_airport} → {item.prediction.dest_airport}
              </span>
              <span
                className="history-risk"
                style={{ backgroundColor: getRiskColor(item.prediction.risk_level) }}
              >
                {item.prediction.risk_level}
              </span>
            </div>
            <div className="history-details">
              <span className="history-airline">{item.prediction.airline}</span>
              <span className="history-time">
                {formatDate(item.timestamp)} {formatTime(item.timestamp)}
              </span>
              <span className="history-prob">
                {Math.round(item.prediction.delay_probability * 100)}%
              </span>
            </div>
            <button
              className="history-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(item.id);
              }}
            >
              <Trash2 size={14} />
            </button>
            <ChevronRight size={16} className="history-arrow" />
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionHistory;
