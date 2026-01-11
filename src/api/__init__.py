"""
API module for SkyPredict
- FastAPI REST API
- Endpoints for predictions, explanations, health checks
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
