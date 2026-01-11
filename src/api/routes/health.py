"""
Health check and system info endpoints
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter

from ..schemas.health import HealthResponse, ModelInfo
from ... import __version__

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Health"])

# Track startup time
startup_time: Optional[datetime] = None


def set_startup_time():
    """Set the startup time (called from main.py)."""
    global startup_time
    startup_time = datetime.utcnow()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and model information."
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, version, and model information.
    """
    from .predictions import predictor

    # Calculate uptime
    uptime = 0.0
    if startup_time:
        uptime = (datetime.utcnow() - startup_time).total_seconds()

    # Check model status
    classifier_loaded = predictor is not None and predictor.classifier is not None
    regressor_loaded = predictor is not None and predictor.regressor is not None

    model_info = ModelInfo(
        classifier_loaded=classifier_loaded,
        regressor_loaded=regressor_loaded,
        model_version="1.0.0",
        features_count=45 if classifier_loaded else None
    )

    # Determine overall status
    status = "healthy" if classifier_loaded else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.utcnow(),
        model_info=model_info,
        uptime_seconds=uptime
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information."
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SkyPredict API",
        "description": "Flight Delay Prediction Service",
        "version": __version__,
        "docs_url": "/docs",
        "health_url": "/health"
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the service is ready to handle requests."
)
async def readiness_check():
    """
    Kubernetes-style readiness probe.

    Returns 200 if models are loaded and ready.
    """
    from .predictions import predictor

    if predictor is None or predictor.classifier is None:
        return {"ready": False, "reason": "Models not loaded"}

    return {"ready": True}


@router.get(
    "/live",
    summary="Liveness check",
    description="Check if the service is alive."
)
async def liveness_check():
    """
    Kubernetes-style liveness probe.

    Returns 200 if service is running.
    """
    return {"alive": True}
