"""
API Routes
"""

from .predictions import router as predictions_router
from .health import router as health_router

__all__ = ["predictions_router", "health_router"]
