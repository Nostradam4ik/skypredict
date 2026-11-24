"""
API Routes
"""

from .health import router as health_router
from .predictions import router as predictions_router

__all__ = ["predictions_router", "health_router"]
