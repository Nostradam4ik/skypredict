import React, { useState, useEffect, useCallback } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import { useTranslation } from 'react-i18next';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import Header from './components/Header';
import FlightForm from './components/FlightForm';
import PredictionResult from './components/PredictionResult';
import PredictionHistory, { HistoryItem } from './components/PredictionHistory';
import FlightComparison from './components/FlightComparison';
import PlaneLoader from './components/PlaneLoader';
import FlightMap from './components/FlightMap';
import StatsDashboard from './components/StatsDashboard';
import FullscreenResult from './components/FullscreenResult';
import QRCodeShare from './components/QRCodeShare';
import LanguageSelector from './components/LanguageSelector';
import KeyboardShortcuts from './components/KeyboardShortcuts';
import { predictFlight, getHealth } from './services/api';
import { FlightInput, PredictionResponse } from './types';
import { useLocalStorage } from './hooks/useLocalStorage';

function App() {
  const { t } = useTranslation();
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');

  // Theme
  const [isDarkMode, setIsDarkMode] = useLocalStorage('skypredict-theme', true);

  // History
  const [history, setHistory] = useLocalStorage<HistoryItem[]>('skypredict-history', []);

  // Compare mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareFlights, setCompareFlights] = useState<PredictionResponse[]>([]);

  // New feature states
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [showQRCode, setShowQRCode] = useState(false);
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false);
  const [showStats, setShowStats] = useState(false);

  useEffect(() => {
    checkApiStatus();
  }, []);

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  // Keyboard shortcuts
  const handleKeyPress = useCallback((e: KeyboardEvent) => {
    // Ignore if typing in an input
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) {
      return;
    }

    switch (e.key.toLowerCase()) {
      case 't':
        setIsDarkMode((prev: boolean) => !prev);
        break;
      case 'c':
        setCompareMode(prev => !prev);
        break;
      case 'f':
        if (prediction) setShowFullscreen(prev => !prev);
        break;
      case 'escape':
        setShowFullscreen(false);
        setShowQRCode(false);
        setShowKeyboardShortcuts(false);
        break;
      case '?':
        setShowKeyboardShortcuts(prev => !prev);
        break;
    }
  }, [prediction, setIsDarkMode]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  const checkApiStatus = async () => {
    try {
      await getHealth();
      setApiStatus('connected');
    } catch {
      setApiStatus('disconnected');
    }
  };

  const showRiskNotification = (result: PredictionResponse) => {
    const prob = Math.round(result.delay_probability * 100);

    if (result.risk_level === 'SEVERE') {
      toast.error(
        `High Alert: ${result.origin_airport} → ${result.dest_airport} has ${prob}% delay probability!`,
        { autoClose: 8000 }
      );
    } else if (result.risk_level === 'HIGH') {
      toast.warning(
        `Warning: ${result.origin_airport} → ${result.dest_airport} has high delay risk (${prob}%)`,
        { autoClose: 6000 }
      );
    } else if (result.risk_level === 'MEDIUM') {
      toast.info(
        `Moderate risk for ${result.origin_airport} → ${result.dest_airport} (${prob}%)`,
        { autoClose: 4000 }
      );
    } else {
      toast.success(
        `Good news! ${result.origin_airport} → ${result.dest_airport} is likely on time`,
        { autoClose: 3000 }
      );
    }
  };

  const addToHistory = (result: PredictionResponse) => {
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      timestamp: new Date(),
      prediction: result
    };
    setHistory(prev => [newItem, ...prev.slice(0, 9)]); // Keep last 10
  };

  const handlePredict = async (flight: FlightInput) => {
    setLoading(true);
    setError(null);

    try {
      const result = await predictFlight(flight);
      setPrediction(result);
      addToHistory(result);
      showRiskNotification(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Please try again.');
      setPrediction(null);
      toast.error('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddToCompare = async (flight: FlightInput) => {
    if (compareFlights.length >= 4) {
      toast.warning('Maximum 4 flights can be compared at once');
      return;
    }

    setLoading(true);
    try {
      const result = await predictFlight(flight);
      setCompareFlights(prev => [...prev, result]);
      toast.success(`Added ${result.origin_airport} → ${result.dest_airport} to comparison`);
    } catch (err: any) {
      toast.error('Failed to add flight to comparison');
    } finally {
      setLoading(false);
    }
  };

  const handleAddCurrentToCompare = () => {
    if (prediction) {
      if (compareFlights.length >= 4) {
        toast.warning('Maximum 4 flights can be compared at once');
        return;
      }
      setCompareFlights(prev => [...prev, prediction]);
      toast.success('Added to comparison');
    }
  };

  const handleRemoveFromCompare = (index: number) => {
    setCompareFlights(prev => prev.filter((_, i) => i !== index));
  };

  const handleClearCompare = () => {
    setCompareFlights([]);
  };

  const handleHistorySelect = (item: HistoryItem) => {
    setPrediction(item.prediction);
  };

  const handleClearHistory = () => {
    setHistory([]);
    toast.info('History cleared');
  };

  const handleDeleteHistoryItem = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  const toggleTheme = () => {
    setIsDarkMode(prev => !prev);
  };

  const toggleCompareMode = () => {
    setCompareMode(prev => !prev);
  };

  return (
    <div className={`app ${isDarkMode ? 'dark' : 'light'}`}>
      <Header
        apiStatus={apiStatus}
        isDarkMode={isDarkMode}
        onThemeToggle={toggleTheme}
        compareMode={compareMode}
        onCompareModeToggle={toggleCompareMode}
        compareCount={compareFlights.length}
      />

      <main className="main-content">
        <div className="container">
          {/* Stats Dashboard Toggle */}
          {history.length >= 2 && (
            <button
              className="stats-toggle-btn"
              onClick={() => setShowStats(prev => !prev)}
            >
              {showStats ? t('hideStats') : t('showStats')}
            </button>
          )}

          {/* Stats Dashboard */}
          {showStats && <StatsDashboard history={history} />}

          {compareMode && compareFlights.length >= 2 && (
            <FlightComparison
              flights={compareFlights}
              onRemove={handleRemoveFromCompare}
              onClear={handleClearCompare}
            />
          )}

          <div className="grid">
            <div className="form-section">
              <FlightForm
                onSubmit={handlePredict}
                onAddToCompare={handleAddToCompare}
                loading={loading}
                compareMode={compareMode}
              />

              <PredictionHistory
                history={history}
                onSelect={handleHistorySelect}
                onClear={handleClearHistory}
                onDelete={handleDeleteHistoryItem}
              />
            </div>

            <div className="result-section">
              {error && (
                <div className="error-message">
                  <p>{error}</p>
                </div>
              )}

              {loading && <PlaneLoader />}

              {prediction && !error && !loading && (
                <>
                  {/* Flight Map */}
                  <div className="map-container">
                    <FlightMap
                      origin={prediction.origin_airport}
                      destination={prediction.dest_airport}
                      riskLevel={prediction.risk_level}
                    />
                  </div>

                  <PredictionResult
                    prediction={prediction}
                    onAddToCompare={handleAddCurrentToCompare}
                    showCompareButton={compareMode}
                  />

                  {/* Fullscreen & QR buttons */}
                  <div className="extra-actions">
                    <button
                      className="action-btn fullscreen"
                      onClick={() => setShowFullscreen(true)}
                      title={t('fullscreen')}
                    >
                      {t('fullscreen')}
                    </button>
                    <button
                      className="action-btn qr"
                      onClick={() => setShowQRCode(true)}
                      title={t('qrCode')}
                    >
                      {t('qrCode')}
                    </button>
                  </div>
                </>
              )}

              {!prediction && !error && !loading && (
                <div className="placeholder">
                  <div className="placeholder-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </div>
                  <h3>{t('enterFlightDetails')}</h3>
                  <p>{t('fillFormToPredict')}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>SkyPredict - Built with XGBoost, FastAPI & React</p>
          <p>Model trained on US domestic flight data</p>
        </div>
        <div className="footer-actions">
          <LanguageSelector />
          <button
            className="shortcuts-btn"
            onClick={() => setShowKeyboardShortcuts(true)}
            title={t('keyboardShortcuts')}
          >
            ?
          </button>
        </div>
      </footer>

      {/* Modals */}
      {showFullscreen && prediction && (
        <FullscreenResult
          prediction={prediction}
          onClose={() => setShowFullscreen(false)}
          onShowQR={() => {
            setShowFullscreen(false);
            setShowQRCode(true);
          }}
        />
      )}

      {showQRCode && prediction && (
        <QRCodeShare
          prediction={prediction}
          onClose={() => setShowQRCode(false)}
        />
      )}

      {showKeyboardShortcuts && (
        <KeyboardShortcuts onClose={() => setShowKeyboardShortcuts(false)} />
      )}

      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme={isDarkMode ? 'dark' : 'light'}
      />
    </div>
  );
}

export default App;
