"""
Health check and system info schemas
"""

from typing import Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about loaded ML models"""
    classifier_loaded: bool = Field(..., description="Whether classifier is loaded")
    regressor_loaded: bool = Field(..., description="Whether regressor is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    last_trained: Optional[datetime] = Field(None, description="Last training timestamp")
    features_count: Optional[int] = Field(None, description="Number of input features")


class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")
    model_info: ModelInfo = Field(..., description="Model information")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-07-15T14:30:00Z",
                "model_info": {
                    "classifier_loaded": True,
                    "regressor_loaded": True,
                    "model_version": "v1.0",
                    "features_count": 45
                },
                "uptime_seconds": 3600.5
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
