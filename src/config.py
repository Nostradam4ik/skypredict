"""
Configuration module for SkyPredict
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class"""

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skypredict.db")

    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_DEBUG = os.getenv("API_DEBUG", "true").lower() == "true"
    API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-key")

    # External APIs
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY", "")

    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "skypredict")

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR))
    MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

    # Data paths
    DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", str(RAW_DATA_DIR))
    DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", str(PROCESSED_DATA_DIR))
    DATA_FEATURES_PATH = os.getenv("DATA_FEATURES_PATH", str(FEATURES_DIR))


class MLConfig:
    """ML-specific configuration"""

    # Target variable
    TARGET_CLASSIFICATION = "is_delayed"  # Binary: 0 = on-time, 1 = delayed
    TARGET_REGRESSION = "arrival_delay_minutes"  # Continuous

    # Delay threshold (minutes)
    DELAY_THRESHOLD = 15  # FAA definition: >15 min = delayed

    # Train/test split
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42

    # Cross-validation
    N_FOLDS = 5

    # Class imbalance
    USE_SMOTE = True
    SMOTE_SAMPLING_STRATEGY = 0.8  # Minority class ratio

    # Feature groups (available 2h before flight)
    TEMPORAL_FEATURES = [
        "month", "day_of_week", "day_of_month", "hour",
        "is_weekend", "is_holiday", "is_peak_hour", "season"
    ]

    FLIGHT_FEATURES = [
        "airline", "origin_airport", "dest_airport",
        "distance", "scheduled_duration", "flight_number_prefix"
    ]

    WEATHER_FEATURES = [
        "origin_temp", "origin_wind_speed", "origin_visibility",
        "origin_precipitation", "origin_humidity",
        "dest_temp", "dest_wind_speed", "dest_visibility",
        "dest_precipitation", "dest_humidity"
    ]

    HISTORICAL_FEATURES = [
        "airline_delay_rate", "origin_delay_rate", "dest_delay_rate",
        "route_delay_rate", "airline_avg_delay", "origin_avg_delay"
    ]

    CONGESTION_FEATURES = [
        "origin_departures_hour", "origin_arrivals_hour",
        "dest_departures_hour", "dest_arrivals_hour"
    ]

    # Features to EXCLUDE (data leakage - not available before flight)
    LEAKAGE_FEATURES = [
        "dep_delay", "departure_delay", "taxi_out", "taxi_in",
        "wheels_off", "wheels_on", "air_time", "actual_elapsed_time",
        "arrival_time", "diverted", "cancelled"
    ]


class AirportConfig:
    """Airport-related configuration"""

    # Top US airports by traffic (for initial focus)
    TOP_AIRPORTS = [
        "ATL", "DFW", "DEN", "ORD", "LAX",
        "CLT", "MCO", "LAS", "PHX", "MIA",
        "SEA", "IAH", "JFK", "EWR", "SFO",
        "MSP", "BOS", "DTW", "FLL", "PHL"
    ]

    # Major airlines
    MAJOR_AIRLINES = [
        "AA",  # American Airlines
        "DL",  # Delta Air Lines
        "UA",  # United Airlines
        "WN",  # Southwest Airlines
        "AS",  # Alaska Airlines
        "B6",  # JetBlue Airways
        "NK",  # Spirit Airlines
        "F9",  # Frontier Airlines
        "G4",  # Allegiant Air
        "HA",  # Hawaiian Airlines
    ]


# Export configs
config = Config()
ml_config = MLConfig()
airport_config = AirportConfig()
