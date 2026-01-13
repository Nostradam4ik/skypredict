import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Plane, Calendar, Clock, MapPin, Cloud, ArrowRightLeft, Plus } from 'lucide-react';
import { FlightInput, AIRLINES, AIRPORTS, calculateDistance } from '../types';
import SearchableSelect from './SearchableSelect';

interface FlightFormProps {
  onSubmit: (flight: FlightInput) => void;
  onAddToCompare?: (flight: FlightInput) => void;
  loading: boolean;
  compareMode?: boolean;
}

type DistanceUnit = 'km' | 'miles';

const FlightForm: React.FC<FlightFormProps> = ({ onSubmit, onAddToCompare, loading, compareMode }) => {
  const { t } = useTranslation();
  const [formData, setFormData] = useState<FlightInput>({
    airline: 'AF',
    origin_airport: 'CDG',
    dest_airport: 'JFK',
    flight_date: new Date().toISOString().split('T')[0],
    scheduled_departure_hour: 14,
    scheduled_departure_minute: 0,
    distance: 0,
  });

  const [distanceUnit, setDistanceUnit] = useState<DistanceUnit>('km');
  const [calculatedDistance, setCalculatedDistance] = useState<number>(0);

  const [showWeather, setShowWeather] = useState(false);
  const [weatherData, setWeatherData] = useState({
    origin_temp_c: 20,
    origin_wind_speed_ms: 5,
    origin_visibility_m: 10000,
    origin_precipitation_mm: 0,
    dest_temp_c: 25,
    dest_wind_speed_ms: 3,
    dest_visibility_m: 10000,
    dest_precipitation_mm: 0,
  });

  // Calculate distance when airports change
  useEffect(() => {
    if (formData.origin_airport && formData.dest_airport) {
      const dist = calculateDistance(formData.origin_airport, formData.dest_airport, distanceUnit);
      setCalculatedDistance(dist);
      // Convert to miles for API (API expects miles)
      const distInMiles = distanceUnit === 'km' ? Math.round(dist * 0.621371) : dist;
      setFormData(prev => ({ ...prev, distance: distInMiles }));
    }
  }, [formData.origin_airport, formData.dest_airport, distanceUnit]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('hour') || name.includes('minute')
        ? parseInt(value) || 0
        : value
    }));
  };

  const handleSelectChange = (name: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleWeatherChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setWeatherData(prev => ({
      ...prev,
      [name]: value === '' ? 0 : parseFloat(value)
    }));
  };

  const swapAirports = () => {
    setFormData(prev => ({
      ...prev,
      origin_airport: prev.dest_airport,
      dest_airport: prev.origin_airport
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const submitData = showWeather
      ? { ...formData, ...weatherData }
      : formData;
    onSubmit(submitData);
  };

  const handleAddToCompare = () => {
    if (onAddToCompare) {
      const submitData = showWeather
        ? { ...formData, ...weatherData }
        : formData;
      onAddToCompare(submitData);
    }
  };

  // Estimate flight duration based on distance
  const estimatedDuration = Math.round((calculatedDistance / (distanceUnit === 'km' ? 800 : 500)) * 60);
  const durationHours = Math.floor(estimatedDuration / 60);
  const durationMins = estimatedDuration % 60;

  return (
    <form onSubmit={handleSubmit} className="flight-form">
      <h2><Plane size={24} /> {t('flightDetails')}</h2>

      <div className="form-row">
        <div className="form-group">
          <SearchableSelect
            options={AIRLINES}
            value={formData.airline}
            onChange={(value) => handleSelectChange('airline', value)}
            placeholder={t('searchAirline')}
            label={t('airline')}
            icon={<Plane size={16} />}
          />
        </div>

        <div className="form-group">
          <label><Calendar size={16} /> {t('date')}</label>
          <input
            type="date"
            name="flight_date"
            value={formData.flight_date}
            onChange={handleChange}
          />
        </div>
      </div>

      <div className="form-row airports-row">
        <div className="form-group">
          <SearchableSelect
            options={AIRPORTS}
            value={formData.origin_airport}
            onChange={(value) => handleSelectChange('origin_airport', value)}
            placeholder={t('searchAirport')}
            label={t('origin')}
            icon={<MapPin size={16} />}
          />
        </div>

        <button type="button" className="swap-btn" onClick={swapAirports} title="Swap airports">
          <ArrowRightLeft size={20} />
        </button>

        <div className="form-group">
          <SearchableSelect
            options={AIRPORTS}
            value={formData.dest_airport}
            onChange={(value) => handleSelectChange('dest_airport', value)}
            placeholder={t('searchAirport')}
            label={t('destination')}
            icon={<MapPin size={16} />}
          />
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label><Clock size={16} /> {t('departureTime')}</label>
          <div className="time-inputs">
            <input
              type="number"
              name="scheduled_departure_hour"
              value={formData.scheduled_departure_hour}
              onChange={handleChange}
              min={0}
              max={23}
              placeholder="Hour"
            />
            <span>:</span>
            <input
              type="number"
              name="scheduled_departure_minute"
              value={formData.scheduled_departure_minute || 0}
              onChange={handleChange}
              min={0}
              max={59}
              placeholder="Min"
            />
          </div>
        </div>

        <div className="form-group">
          <label>{t('distance')}</label>
          <div className="distance-display">
            <span className="distance-value">{calculatedDistance.toLocaleString()}</span>
            <div className="unit-toggle">
              <button
                type="button"
                className={distanceUnit === 'km' ? 'active' : ''}
                onClick={() => setDistanceUnit('km')}
              >
                km
              </button>
              <button
                type="button"
                className={distanceUnit === 'miles' ? 'active' : ''}
                onClick={() => setDistanceUnit('miles')}
              >
                miles
              </button>
            </div>
          </div>
        </div>
      </div>

      {calculatedDistance > 0 && (
        <div className="flight-info-banner">
          <span>{t('estimatedFlightTime')}: <strong>{durationHours}h {durationMins}min</strong></span>
        </div>
      )}

      <div className="weather-toggle">
        <label>
          <input
            type="checkbox"
            checked={showWeather}
            onChange={() => setShowWeather(!showWeather)}
          />
          <Cloud size={16} /> {t('addWeatherConditions')}
        </label>
      </div>

      {showWeather && (
        <div className="weather-section">
          <h3>{t('originWeather')}</h3>
          <div className="form-row weather-row">
            <div className="form-group">
              <label>{t('temp')}</label>
              <input
                type="number"
                name="origin_temp_c"
                value={weatherData.origin_temp_c || ''}
                onChange={handleWeatherChange}
                placeholder="20"
              />
            </div>
            <div className="form-group">
              <label>{t('wind')}</label>
              <input
                type="number"
                name="origin_wind_speed_ms"
                value={weatherData.origin_wind_speed_ms || ''}
                onChange={handleWeatherChange}
                placeholder="5"
              />
            </div>
            <div className="form-group">
              <label>{t('visibility')}</label>
              <input
                type="number"
                name="origin_visibility_m"
                value={weatherData.origin_visibility_m || ''}
                onChange={handleWeatherChange}
                placeholder="10000"
              />
            </div>
            <div className="form-group">
              <label>{t('rain')}</label>
              <input
                type="number"
                name="origin_precipitation_mm"
                value={weatherData.origin_precipitation_mm || ''}
                onChange={handleWeatherChange}
                placeholder="0"
              />
            </div>
          </div>

          <h3>{t('destWeather')}</h3>
          <div className="form-row weather-row">
            <div className="form-group">
              <label>{t('temp')}</label>
              <input
                type="number"
                name="dest_temp_c"
                value={weatherData.dest_temp_c || ''}
                onChange={handleWeatherChange}
                placeholder="25"
              />
            </div>
            <div className="form-group">
              <label>{t('wind')}</label>
              <input
                type="number"
                name="dest_wind_speed_ms"
                value={weatherData.dest_wind_speed_ms || ''}
                onChange={handleWeatherChange}
                placeholder="3"
              />
            </div>
            <div className="form-group">
              <label>{t('visibility')}</label>
              <input
                type="number"
                name="dest_visibility_m"
                value={weatherData.dest_visibility_m || ''}
                onChange={handleWeatherChange}
                placeholder="10000"
              />
            </div>
            <div className="form-group">
              <label>{t('rain')}</label>
              <input
                type="number"
                name="dest_precipitation_mm"
                value={weatherData.dest_precipitation_mm || ''}
                onChange={handleWeatherChange}
                placeholder="0"
              />
            </div>
          </div>
        </div>
      )}

      <div className="form-actions">
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? t('predicting') : t('predictDelay')}
        </button>
        {compareMode && onAddToCompare && (
          <button
            type="button"
            className="compare-btn"
            onClick={handleAddToCompare}
            disabled={loading}
          >
            <Plus size={18} /> {t('addToCompare')}
          </button>
        )}
      </div>
    </form>
  );
};

export default FlightForm;
