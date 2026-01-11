"""
Prediction Module for SkyPredict
=================================

Handles inference for flight delay predictions.
Two-stage prediction: Classification → Regression
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from ..config import ml_config, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightDelayPredictor:
    """
    Production-ready predictor for flight delays.

    Features:
    - Two-stage prediction (classification + regression)
    - Single flight and batch prediction support
    - Confidence scores
    - Explanation-ready output
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.classifier = None
        self.regressor = None
        self.feature_names = None
        self._load_models()

    def _load_models(self):
        """Load trained models from disk."""
        clf_path = self.models_dir / "skypredict_classifier_latest.joblib"
        reg_path = self.models_dir / "skypredict_regressor_latest.joblib"

        if clf_path.exists():
            self.classifier = joblib.load(clf_path)
            logger.info("Loaded classifier model")
        else:
            logger.warning(f"Classifier not found at: {clf_path}")

        if reg_path.exists():
            self.regressor = joblib.load(reg_path)
            logger.info("Loaded regressor model")
        else:
            logger.warning(f"Regressor not found at: {reg_path}")

    def predict_single(
        self,
        flight_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict delay for a single flight.

        Args:
            flight_data: Dictionary with flight features

        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([flight_data])

        # Get prediction
        result = self.predict_batch(df)

        return {
            "is_delayed": bool(result["predictions"][0]),
            "delay_probability": float(result["probabilities"][0]),
            "estimated_delay_minutes": float(result["estimated_delays"][0]) if result["estimated_delays"][0] else None,
            "confidence": float(result["confidence"][0]),
            "risk_level": result["risk_levels"][0]
        }

    def predict_batch(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict delays for multiple flights.

        Args:
            df: DataFrame with flight features

        Returns:
            Dictionary with batch prediction results
        """
        if self.classifier is None:
            raise ValueError("Classifier model not loaded")

        # Ensure numeric features only
        numeric_df = df.select_dtypes(include=[np.number])

        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.median())

        # Classification: Is the flight delayed?
        delay_proba = self.classifier.predict_proba(numeric_df)[:, 1]
        is_delayed = (delay_proba >= 0.5).astype(int)

        # Regression: How long is the delay? (only for predicted delayed flights)
        estimated_delays = np.zeros(len(df))
        if self.regressor is not None:
            delayed_mask = is_delayed == 1
            if delayed_mask.any():
                delayed_features = numeric_df[delayed_mask]
                estimated_delays[delayed_mask] = self.regressor.predict(delayed_features)
                # Clip to reasonable range
                estimated_delays = np.clip(estimated_delays, 0, 720)

        # Calculate confidence (distance from 0.5 threshold)
        confidence = np.abs(delay_proba - 0.5) * 2  # Scale to 0-1

        # Risk levels
        risk_levels = self._calculate_risk_levels(delay_proba, estimated_delays)

        return {
            "predictions": is_delayed.tolist(),
            "probabilities": delay_proba.tolist(),
            "estimated_delays": estimated_delays.tolist(),
            "confidence": confidence.tolist(),
            "risk_levels": risk_levels
        }

    def _calculate_risk_levels(
        self,
        probabilities: np.ndarray,
        delays: np.ndarray
    ) -> list:
        """
        Calculate risk level categories.

        Categories:
        - LOW: < 30% delay probability
        - MEDIUM: 30-60% probability or delay < 30 min
        - HIGH: 60-80% probability or delay 30-60 min
        - SEVERE: > 80% probability or delay > 60 min
        """
        risk_levels = []

        for prob, delay in zip(probabilities, delays):
            if prob < 0.3:
                risk_levels.append("LOW")
            elif prob < 0.6 and delay < 30:
                risk_levels.append("MEDIUM")
            elif prob < 0.8 and delay < 60:
                risk_levels.append("HIGH")
            else:
                risk_levels.append("SEVERE")

        return risk_levels

    def predict_with_explanation(
        self,
        flight_data: Dict[str, Any],
        explainer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Predict with SHAP explanation.

        Args:
            flight_data: Dictionary with flight features
            explainer: ModelExplainer instance (optional)

        Returns:
            Dictionary with prediction and explanation
        """
        # Get basic prediction
        result = self.predict_single(flight_data)

        # Add explanation if explainer provided
        if explainer is not None:
            df = pd.DataFrame([flight_data])
            explanation = explainer.explain_single(df)
            result["explanation"] = explanation

        return result


class RealTimePredictionService:
    """
    Service for real-time predictions with caching and batching.
    """

    def __init__(self, predictor: FlightDelayPredictor):
        self.predictor = predictor
        self.prediction_cache = {}
        self.cache_ttl_seconds = 300  # 5 minutes

    def get_prediction(
        self,
        flight_id: str,
        flight_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get prediction with caching support.

        Args:
            flight_id: Unique flight identifier
            flight_data: Flight features
            use_cache: Whether to use cache

        Returns:
            Prediction result
        """
        # Check cache
        if use_cache and flight_id in self.prediction_cache:
            cached = self.prediction_cache[flight_id]
            if (datetime.now() - cached["timestamp"]).total_seconds() < self.cache_ttl_seconds:
                return cached["prediction"]

        # Get fresh prediction
        prediction = self.predictor.predict_single(flight_data)

        # Update cache
        self.prediction_cache[flight_id] = {
            "prediction": prediction,
            "timestamp": datetime.now()
        }

        return prediction

    def clear_cache(self):
        """Clear prediction cache."""
        self.prediction_cache = {}


if __name__ == "__main__":
    # Test prediction
    print("Testing FlightDelayPredictor...")

    # Sample flight data
    sample_flight = {
        "month": 7,
        "day_of_week": 4,
        "hour": 14,
        "is_weekend": 0,
        "is_peak_hour": 0,
        "distance": 1500,
        "log_distance": 7.3,
        "origin_is_hub": 1,
        "dest_is_hub": 1,
        "airline_delay_rate": 0.22,
        "origin_delay_rate": 0.18,
        "dest_delay_rate": 0.20,
        "weather_severity_score": 2,
        "origin_congestion": 0.7
    }

    try:
        predictor = FlightDelayPredictor()
        result = predictor.predict_single(sample_flight)
        print(f"\nPrediction Result:")
        print(f"  Is Delayed: {result['is_delayed']}")
        print(f"  Probability: {result['delay_probability']:.2%}")
        print(f"  Estimated Delay: {result['estimated_delay_minutes']} minutes")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
    except ValueError as e:
        print(f"Models not trained yet: {e}")
        print("Run the training pipeline first to generate models.")
