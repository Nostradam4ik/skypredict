"""
Pydantic schemas for API request/response validation
"""

from .flight import (
    FlightInput,
    FlightPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureExplanation,
    PredictionExplanation
)
from .health import HealthResponse, ModelInfo

__all__ = [
    "FlightInput",
    "FlightPredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "FeatureExplanation",
    "PredictionExplanation",
    "HealthResponse",
    "ModelInfo"
]
