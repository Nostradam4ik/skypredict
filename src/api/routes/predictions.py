"""
Prediction endpoints for SkyPredict API
"""

import logging
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Query, Depends

from ..schemas.flight import (
    FlightInput,
    FlightPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionExplanation,
    FeatureExplanation,
    RiskLevel
)
from ...models.predict import FlightDelayPredictor
from ...data.feature_engineering import FeatureEngineer

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])

# Global predictor instance (initialized in main.py)
predictor: Optional[FlightDelayPredictor] = None
feature_engineer: Optional[FeatureEngineer] = None


def get_predictor() -> FlightDelayPredictor:
    """Dependency to get predictor instance."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not initialized. Models may not be trained yet."
        )
    return predictor


def flight_input_to_features(flight: FlightInput) -> dict:
    """
    Convert FlightInput to feature dictionary for model prediction.

    This transforms the API input into the features expected by the model.
    """
    # Extract date components
    flight_date = flight.flight_date
    hour = flight.scheduled_departure_hour

    # Calculate temporal features
    day_of_week = flight_date.weekday()
    month = flight_date.month
    day_of_month = flight_date.day
    is_weekend = 1 if day_of_week >= 5 else 0

    # Peak hours
    is_morning_peak = 1 if 6 <= hour <= 9 else 0
    is_evening_peak = 1 if 16 <= hour <= 20 else 0
    is_peak_hour = 1 if is_morning_peak or is_evening_peak else 0

    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Hub airports
    hubs = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO"]
    origin_is_hub = 1 if flight.origin_airport in hubs else 0
    dest_is_hub = 1 if flight.dest_airport in hubs else 0

    # Distance features
    distance = flight.distance or 1000  # Default if not provided
    log_distance = np.log1p(distance)

    # Duration features
    scheduled_duration = flight.scheduled_duration or 150
    log_duration = np.log1p(scheduled_duration)

    # Weather features (use defaults if not provided)
    origin_temp = flight.origin_temp_c or 20
    origin_wind = flight.origin_wind_speed_ms or 5
    origin_vis = flight.origin_visibility_m or 10000
    origin_precip = flight.origin_precipitation_mm or 0
    dest_temp = flight.dest_temp_c or 20
    dest_wind = flight.dest_wind_speed_ms or 5
    dest_vis = flight.dest_visibility_m or 10000
    dest_precip = flight.dest_precipitation_mm or 0

    # Weather severity features
    origin_low_visibility = 1 if origin_vis < 5000 else 0
    dest_low_visibility = 1 if dest_vis < 5000 else 0
    origin_high_wind = 1 if origin_wind > 10 else 0
    dest_high_wind = 1 if dest_wind > 10 else 0
    origin_has_precipitation = 1 if origin_precip > 0 else 0
    dest_has_precipitation = 1 if dest_precip > 0 else 0
    weather_severity_score = (
        origin_low_visibility + dest_low_visibility +
        origin_high_wind + dest_high_wind +
        origin_has_precipitation + dest_has_precipitation
    )

    # Historical rates (defaults based on industry averages)
    airline_delay_rates = {
        "AA": 0.20, "DL": 0.15, "UA": 0.22, "WN": 0.18,
        "AS": 0.14, "B6": 0.24, "NK": 0.28, "F9": 0.26
    }
    airline_delay_rate = airline_delay_rates.get(flight.airline, 0.20)

    # Build feature dictionary
    features = {
        # Temporal
        "month": month,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "hour": hour,
        "is_weekend": is_weekend,
        "is_morning_peak": is_morning_peak,
        "is_evening_peak": is_evening_peak,
        "is_peak_hour": is_peak_hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,

        # Flight
        "distance": distance,
        "log_distance": log_distance,
        "scheduled_duration": scheduled_duration,
        "log_duration": log_duration,
        "origin_is_hub": origin_is_hub,
        "dest_is_hub": dest_is_hub,

        # Weather
        "origin_temp_c": origin_temp,
        "origin_wind_speed_ms": origin_wind,
        "origin_visibility_m": origin_vis,
        "origin_precipitation_mm": origin_precip,
        "dest_temp_c": dest_temp,
        "dest_wind_speed_ms": dest_wind,
        "dest_visibility_m": dest_vis,
        "dest_precipitation_mm": dest_precip,
        "origin_low_visibility": origin_low_visibility,
        "dest_low_visibility": dest_low_visibility,
        "origin_high_wind": origin_high_wind,
        "dest_high_wind": dest_high_wind,
        "origin_has_precipitation": origin_has_precipitation,
        "dest_has_precipitation": dest_has_precipitation,
        "weather_severity_score": weather_severity_score,

        # Historical
        "airline_delay_rate": airline_delay_rate,
        "origin_delay_rate": 0.18,  # Default
        "dest_delay_rate": 0.18,
        "hour_delay_rate": 0.15 + (0.05 * is_peak_hour),

        # Congestion (default estimate)
        "origin_congestion": 0.5 + (0.2 * origin_is_hub) + (0.1 * is_peak_hour),
    }

    return features


@router.post(
    "/single",
    response_model=FlightPredictionResponse,
    summary="Predict delay for a single flight",
    description="Returns delay prediction with probability, estimated delay time, and risk level."
)
async def predict_single_flight(
    flight: FlightInput,
    include_explanation: bool = Query(False, description="Include SHAP explanation"),
    pred: FlightDelayPredictor = Depends(get_predictor)
) -> FlightPredictionResponse:
    """
    Predict whether a flight will be delayed.

    Returns:
    - is_delayed: Boolean prediction
    - delay_probability: Probability of delay (0-1)
    - estimated_delay_minutes: Estimated delay if delayed
    - risk_level: LOW, MEDIUM, HIGH, or SEVERE
    - explanation: SHAP-based explanation (optional)
    """
    try:
        # Convert input to features
        features = flight_input_to_features(flight)

        # Get prediction
        result = pred.predict_single(features)

        # Build response
        response = FlightPredictionResponse(
            is_delayed=result["is_delayed"],
            delay_probability=round(result["delay_probability"], 4),
            estimated_delay_minutes=round(result["estimated_delay_minutes"], 1) if result["estimated_delay_minutes"] else None,
            confidence=round(result["confidence"], 4),
            risk_level=RiskLevel(result["risk_level"]),
            airline=flight.airline,
            origin_airport=flight.origin_airport,
            dest_airport=flight.dest_airport,
            flight_date=str(flight.flight_date),
            scheduled_departure=f"{flight.scheduled_departure_hour:02d}:{flight.scheduled_departure_minute:02d}"
        )

        # Add explanation if requested
        if include_explanation and "explanation" in result:
            exp = result["explanation"]
            response.explanation = PredictionExplanation(
                base_value=exp["base_value"],
                features=[
                    FeatureExplanation(**f) for f in exp["features"]
                ]
            )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict delays for multiple flights",
    description="Batch prediction for up to 100 flights at once."
)
async def predict_batch_flights(
    request: BatchPredictionRequest,
    pred: FlightDelayPredictor = Depends(get_predictor)
) -> BatchPredictionResponse:
    """
    Predict delays for multiple flights.

    Maximum 100 flights per request.
    """
    try:
        predictions = []

        for flight in request.flights:
            features = flight_input_to_features(flight)
            result = pred.predict_single(features)

            predictions.append(FlightPredictionResponse(
                is_delayed=result["is_delayed"],
                delay_probability=round(result["delay_probability"], 4),
                estimated_delay_minutes=round(result["estimated_delay_minutes"], 1) if result["estimated_delay_minutes"] else None,
                confidence=round(result["confidence"], 4),
                risk_level=RiskLevel(result["risk_level"]),
                airline=flight.airline,
                origin_airport=flight.origin_airport,
                dest_airport=flight.dest_airport,
                flight_date=str(flight.flight_date),
                scheduled_departure=f"{flight.scheduled_departure_hour:02d}:{flight.scheduled_departure_minute:02d}"
            ))

        # Calculate summary statistics
        delayed_count = sum(1 for p in predictions if p.is_delayed)
        avg_prob = sum(p.delay_probability for p in predictions) / len(predictions)

        return BatchPredictionResponse(
            predictions=predictions,
            total_flights=len(predictions),
            delayed_count=delayed_count,
            on_time_count=len(predictions) - delayed_count,
            average_delay_probability=round(avg_prob, 4)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get(
    "/airports",
    summary="Get supported airports",
    description="Returns list of airports supported by the model."
)
async def get_supported_airports():
    """Get list of supported airports."""
    from ...config import airport_config
    return {
        "airports": airport_config.TOP_AIRPORTS,
        "count": len(airport_config.TOP_AIRPORTS)
    }


@router.get(
    "/airlines",
    summary="Get supported airlines",
    description="Returns list of airlines supported by the model."
)
async def get_supported_airlines():
    """Get list of supported airlines."""
    from ...config import airport_config
    return {
        "airlines": airport_config.MAJOR_AIRLINES,
        "count": len(airport_config.MAJOR_AIRLINES)
    }
