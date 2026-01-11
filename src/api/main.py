"""
SkyPredict API - Main Application
==================================

FastAPI application for flight delay prediction.

Features:
- Single and batch prediction endpoints
- SHAP-based explanations
- Health checks
- OpenAPI documentation
- CORS support
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .routes import predictions_router, health_router
from .routes.health import set_startup_time
from .routes import predictions as predictions_module
from ..models.predict import FlightDelayPredictor
from ..config import config
from .. import __version__

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes models on startup, cleans up on shutdown.
    """
    # Startup
    logger.info("Starting SkyPredict API...")
    set_startup_time()

    # Initialize predictor
    try:
        predictions_module.predictor = FlightDelayPredictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize predictor: {e}")
        logger.warning("API will start but predictions will be unavailable until models are trained")

    yield

    # Shutdown
    logger.info("Shutting down SkyPredict API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="SkyPredict API",
        description="""
## Flight Delay Prediction API

SkyPredict uses machine learning to predict flight delays with high accuracy.

### Features

- **Single Prediction**: Predict delay for one flight
- **Batch Prediction**: Predict delays for multiple flights (up to 100)
- **Explanations**: Get SHAP-based explanations for predictions
- **Real-time Weather**: Automatic weather data integration

### How It Works

1. **Input**: Provide flight details (airline, origin, destination, date, time)
2. **Processing**: ML model analyzes 40+ features including:
   - Historical delay patterns
   - Weather conditions
   - Airport congestion
   - Time-based factors
3. **Output**: Get delay probability, estimated delay time, and risk level

### Model Information

- **Algorithm**: Ensemble of XGBoost, LightGBM, and CatBoost
- **Accuracy**: ~85% classification accuracy
- **Training Data**: US domestic flights 2018-2023

### Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| LOW | < 30% | No action needed |
| MEDIUM | 30-60% | Monitor flight status |
| HIGH | 60-80% | Consider alternatives |
| SEVERE | > 80% | Expect significant delay |
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(predictions_router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": str(exc) if config.API_DEBUG else None
            }
        )

    return app


# Create application instance
app = create_app()


def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="SkyPredict API",
        version=__version__,
        description=app.description,
        routes=app.routes,
    )

    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_DEBUG,
        log_level="info"
    )
