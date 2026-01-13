import axios from 'axios';
import { FlightInput, PredictionResponse, HealthResponse, AirportsResponse, AirlinesResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictFlight = async (flight: FlightInput): Promise<PredictionResponse> => {
  const response = await api.post<PredictionResponse>('/api/v1/predictions/single', flight);
  return response.data;
};

export const predictFlightWithExplanation = async (flight: FlightInput): Promise<PredictionResponse> => {
  const response = await api.post<PredictionResponse>(
    '/api/v1/predictions/single?include_explanation=true',
    flight
  );
  return response.data;
};

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export const getAirports = async (): Promise<AirportsResponse> => {
  const response = await api.get<AirportsResponse>('/api/v1/predictions/airports');
  return response.data;
};

export const getAirlines = async (): Promise<AirlinesResponse> => {
  const response = await api.get<AirlinesResponse>('/api/v1/predictions/airlines');
  return response.data;
};

export default api;
