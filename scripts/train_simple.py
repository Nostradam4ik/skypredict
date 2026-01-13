"""
Simple Training Script for SkyPredict
======================================

Trains models with only the features available at API prediction time.
This ensures consistency between training and inference.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Features that match the API's flight_input_to_features function
API_FEATURES = [
    # Temporal
    "month", "day_of_week", "day_of_month", "hour",
    "is_weekend", "is_morning_peak", "is_evening_peak", "is_peak_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",

    # Flight
    "distance", "log_distance", "scheduled_duration", "log_duration",
    "origin_is_hub", "dest_is_hub",

    # Weather
    "origin_temp_c", "origin_wind_speed_ms", "origin_visibility_m", "origin_precipitation_mm",
    "dest_temp_c", "dest_wind_speed_ms", "dest_visibility_m", "dest_precipitation_mm",
    "origin_low_visibility", "dest_low_visibility",
    "origin_high_wind", "dest_high_wind",
    "origin_has_precipitation", "dest_has_precipitation",
    "weather_severity_score",

    # Historical
    "airline_delay_rate", "origin_delay_rate", "dest_delay_rate", "hour_delay_rate",

    # Congestion
    "origin_congestion",
]


def create_training_dataset(n_samples: int = 50000) -> pd.DataFrame:
    """
    Create a realistic synthetic training dataset.

    Features match exactly what the API generates.
    """
    logger.info(f"Creating training dataset with {n_samples} samples...")
    np.random.seed(42)

    # Generate base temporal features
    months = np.random.randint(1, 13, n_samples)
    days_of_week = np.random.randint(0, 7, n_samples)
    days_of_month = np.random.randint(1, 29, n_samples)
    hours = np.random.randint(5, 24, n_samples)

    # Derived temporal features
    is_weekend = (days_of_week >= 5).astype(int)
    is_morning_peak = ((hours >= 6) & (hours <= 9)).astype(int)
    is_evening_peak = ((hours >= 16) & (hours <= 20)).astype(int)
    is_peak_hour = (is_morning_peak | is_evening_peak).astype(int)

    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    dow_sin = np.sin(2 * np.pi * days_of_week / 7)
    dow_cos = np.cos(2 * np.pi * days_of_week / 7)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

    # Flight features
    distances = np.random.uniform(200, 3000, n_samples)
    log_distances = np.log1p(distances)
    scheduled_durations = distances / 8 + np.random.normal(30, 10, n_samples)
    scheduled_durations = np.clip(scheduled_durations, 45, 400)
    log_durations = np.log1p(scheduled_durations)
    origin_is_hub = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    dest_is_hub = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

    # Weather features
    origin_temp = np.random.normal(20, 15, n_samples)
    origin_wind = np.abs(np.random.normal(5, 5, n_samples))
    origin_visibility = np.random.choice([10000, 8000, 5000, 2000, 1000], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])
    origin_precipitation = np.random.exponential(2, n_samples) * np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    dest_temp = np.random.normal(20, 15, n_samples)
    dest_wind = np.abs(np.random.normal(5, 5, n_samples))
    dest_visibility = np.random.choice([10000, 8000, 5000, 2000, 1000], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])
    dest_precipitation = np.random.exponential(2, n_samples) * np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Weather severity indicators
    origin_low_visibility = (origin_visibility < 5000).astype(int)
    dest_low_visibility = (dest_visibility < 5000).astype(int)
    origin_high_wind = (origin_wind > 10).astype(int)
    dest_high_wind = (dest_wind > 10).astype(int)
    origin_has_precipitation = (origin_precipitation > 0).astype(int)
    dest_has_precipitation = (dest_precipitation > 0).astype(int)
    weather_severity_score = (
        origin_low_visibility + dest_low_visibility +
        origin_high_wind + dest_high_wind +
        origin_has_precipitation + dest_has_precipitation
    )

    # Historical features
    airline_delay_rate = np.random.uniform(0.12, 0.30, n_samples)
    origin_delay_rate = np.random.uniform(0.10, 0.25, n_samples)
    dest_delay_rate = np.random.uniform(0.10, 0.25, n_samples)
    hour_delay_rate = 0.15 + 0.05 * is_peak_hour + np.random.normal(0, 0.02, n_samples)

    # Congestion
    origin_congestion = 0.3 + 0.3 * origin_is_hub + 0.2 * is_peak_hour + np.random.normal(0, 0.1, n_samples)
    origin_congestion = np.clip(origin_congestion, 0, 1)

    # Create DataFrame
    df = pd.DataFrame({
        "month": months,
        "day_of_week": days_of_week,
        "day_of_month": days_of_month,
        "hour": hours,
        "is_weekend": is_weekend,
        "is_morning_peak": is_morning_peak,
        "is_evening_peak": is_evening_peak,
        "is_peak_hour": is_peak_hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "distance": distances,
        "log_distance": log_distances,
        "scheduled_duration": scheduled_durations,
        "log_duration": log_durations,
        "origin_is_hub": origin_is_hub,
        "dest_is_hub": dest_is_hub,
        "origin_temp_c": origin_temp,
        "origin_wind_speed_ms": origin_wind,
        "origin_visibility_m": origin_visibility,
        "origin_precipitation_mm": origin_precipitation,
        "dest_temp_c": dest_temp,
        "dest_wind_speed_ms": dest_wind,
        "dest_visibility_m": dest_visibility,
        "dest_precipitation_mm": dest_precipitation,
        "origin_low_visibility": origin_low_visibility,
        "dest_low_visibility": dest_low_visibility,
        "origin_high_wind": origin_high_wind,
        "dest_high_wind": dest_high_wind,
        "origin_has_precipitation": origin_has_precipitation,
        "dest_has_precipitation": dest_has_precipitation,
        "weather_severity_score": weather_severity_score,
        "airline_delay_rate": airline_delay_rate,
        "origin_delay_rate": origin_delay_rate,
        "dest_delay_rate": dest_delay_rate,
        "hour_delay_rate": hour_delay_rate,
        "origin_congestion": origin_congestion,
    })

    # Generate realistic target variable
    # Probability of delay based on features
    delay_prob = (
        0.10 +                                    # Base rate
        0.15 * weather_severity_score / 6 +      # Weather impact
        0.10 * is_peak_hour +                    # Peak hour impact
        0.05 * is_weekend +                      # Weekend impact
        0.20 * airline_delay_rate +              # Airline history
        0.10 * origin_congestion +               # Congestion
        0.05 * (distances > 2000).astype(int) +  # Long flights
        np.random.normal(0, 0.05, n_samples)     # Noise
    )
    delay_prob = np.clip(delay_prob, 0.05, 0.95)

    # Binary target
    df["is_delayed"] = (np.random.random(n_samples) < delay_prob).astype(int)

    # Regression target (delay minutes for delayed flights)
    base_delay = 20 + 30 * weather_severity_score / 6 + 15 * is_peak_hour
    df["delay_minutes"] = np.where(
        df["is_delayed"] == 1,
        base_delay + np.random.exponential(20, n_samples),
        0
    )
    df["delay_minutes"] = np.clip(df["delay_minutes"], 0, 300)

    logger.info(f"Dataset created. Shape: {df.shape}")
    logger.info(f"Delay rate: {df['is_delayed'].mean():.2%}")

    return df


def train_models(df: pd.DataFrame, output_dir: Path):
    """Train classifier and regressor models."""

    # Prepare features
    X = df[API_FEATURES]
    y_class = df["is_delayed"]
    y_reg = df["delay_minutes"]

    logger.info(f"Training with {len(API_FEATURES)} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Train delay rate: {y_train.mean():.2%}")

    # Apply SMOTE
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {len(X_train_balanced)} samples")

    # Train classifier
    logger.info("Training XGBoost classifier...")
    classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    classifier.fit(X_train_balanced, y_train_balanced)

    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Classifier Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")

    # Train regressor on delayed flights only
    logger.info("Training XGBoost regressor...")
    delayed_mask = df["is_delayed"] == 1
    X_delayed = df.loc[delayed_mask, API_FEATURES]
    y_delayed = df.loc[delayed_mask, "delay_minutes"]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_delayed, y_delayed, test_size=0.2, random_state=42
    )

    regressor = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    regressor.fit(X_reg_train, y_reg_train)

    # Evaluate regressor
    y_reg_pred = regressor.predict(X_reg_test)
    mae = np.mean(np.abs(y_reg_test - y_reg_pred))
    rmse = np.sqrt(np.mean((y_reg_test - y_reg_pred) ** 2))

    logger.info(f"Regressor Results:")
    logger.info(f"  MAE: {mae:.2f} minutes")
    logger.info(f"  RMSE: {rmse:.2f} minutes")

    # Save models
    output_dir.mkdir(parents=True, exist_ok=True)

    clf_path = output_dir / "skypredict_classifier_latest.joblib"
    reg_path = output_dir / "skypredict_regressor_latest.joblib"

    joblib.dump(classifier, clf_path)
    joblib.dump(regressor, reg_path)

    logger.info(f"Models saved to: {output_dir}")

    # Save feature names
    import json
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump({"features": API_FEATURES}, f, indent=2)

    return classifier, regressor


def main():
    logger.info("=" * 60)
    logger.info("SkyPredict Model Training")
    logger.info("=" * 60)

    # Create dataset
    df = create_training_dataset(n_samples=50000)

    # Train models
    output_dir = project_root / "models"
    classifier, regressor = train_models(df, output_dir)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Restart the API to load the new models.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
