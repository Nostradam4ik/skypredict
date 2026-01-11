"""
Flight-related Pydantic schemas for API validation
"""

from typing import Optional, List
from datetime import date, time
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class FlightInput(BaseModel):
    """
    Input schema for flight delay prediction.

    All fields should be available 2 hours before departure.
    """
    # Flight identifiers
    airline: str = Field(..., description="Airline code (e.g., 'AA', 'DL', 'UA')", min_length=2, max_length=3)
    flight_number: Optional[int] = Field(None, description="Flight number", ge=1, le=9999)
    origin_airport: str = Field(..., description="Origin airport IATA code (e.g., 'JFK')", min_length=3, max_length=3)
    dest_airport: str = Field(..., description="Destination airport IATA code", min_length=3, max_length=3)

    # Schedule
    flight_date: date = Field(..., description="Flight date")
    scheduled_departure_hour: int = Field(..., description="Scheduled departure hour (0-23)", ge=0, le=23)
    scheduled_departure_minute: int = Field(0, description="Scheduled departure minute (0-59)", ge=0, le=59)

    # Flight details
    distance: Optional[float] = Field(None, description="Flight distance in miles", ge=0)
    scheduled_duration: Optional[int] = Field(None, description="Scheduled flight duration in minutes", ge=0)

    # Weather (optional - will be fetched if not provided)
    origin_temp_c: Optional[float] = Field(None, description="Origin temperature (Celsius)")
    origin_wind_speed_ms: Optional[float] = Field(None, description="Origin wind speed (m/s)", ge=0)
    origin_visibility_m: Optional[float] = Field(None, description="Origin visibility (meters)", ge=0)
    origin_precipitation_mm: Optional[float] = Field(None, description="Origin precipitation (mm)", ge=0)
    dest_temp_c: Optional[float] = Field(None, description="Destination temperature (Celsius)")
    dest_wind_speed_ms: Optional[float] = Field(None, description="Destination wind speed (m/s)", ge=0)
    dest_visibility_m: Optional[float] = Field(None, description="Destination visibility (meters)", ge=0)
    dest_precipitation_mm: Optional[float] = Field(None, description="Destination precipitation (mm)", ge=0)

    @field_validator("airline", "origin_airport", "dest_airport")
    @classmethod
    def uppercase_codes(cls, v: str) -> str:
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "airline": "AA",
                "flight_number": 1234,
                "origin_airport": "JFK",
                "dest_airport": "LAX",
                "flight_date": "2024-07-15",
                "scheduled_departure_hour": 14,
                "scheduled_departure_minute": 30,
                "distance": 2475,
                "scheduled_duration": 330
            }
        }
    }


class FeatureExplanation(BaseModel):
    """Single feature contribution to prediction"""
    name: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    shap_value: float = Field(..., description="SHAP contribution value")
    impact: str = Field(..., description="Impact direction (increases/decreases delay risk)")


class PredictionExplanation(BaseModel):
    """SHAP-based prediction explanation"""
    base_value: float = Field(..., description="Base prediction probability")
    features: List[FeatureExplanation] = Field(..., description="Feature contributions")


class FlightPredictionResponse(BaseModel):
    """Response schema for single flight prediction"""
    # Prediction results
    is_delayed: bool = Field(..., description="Whether flight is predicted to be delayed (>15 min)")
    delay_probability: float = Field(..., description="Probability of delay (0-1)", ge=0, le=1)
    estimated_delay_minutes: Optional[float] = Field(None, description="Estimated delay in minutes if delayed")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    risk_level: RiskLevel = Field(..., description="Risk level category")

    # Input echo
    airline: str
    origin_airport: str
    dest_airport: str
    flight_date: str
    scheduled_departure: str

    # Explanation (optional)
    explanation: Optional[PredictionExplanation] = Field(None, description="SHAP explanation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_delayed": True,
                "delay_probability": 0.72,
                "estimated_delay_minutes": 45.5,
                "confidence": 0.85,
                "risk_level": "HIGH",
                "airline": "AA",
                "origin_airport": "JFK",
                "dest_airport": "LAX",
                "flight_date": "2024-07-15",
                "scheduled_departure": "14:30"
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    flights: List[FlightInput] = Field(..., description="List of flights to predict", max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[FlightPredictionResponse]
    total_flights: int
    delayed_count: int
    on_time_count: int
    average_delay_probability: float
