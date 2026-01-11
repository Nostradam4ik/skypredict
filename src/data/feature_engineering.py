"""
Feature Engineering Module for SkyPredict
==========================================

Creates ML-ready features from preprocessed data.
All features are designed to be available 2 hours before flight departure.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config import ml_config, airport_config, FEATURES_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features for flight delay prediction.

    Feature categories:
    1. Temporal features (time-based patterns)
    2. Flight features (airline, route, distance)
    3. Weather features (origin and destination)
    4. Historical features (delay rates, averages)
    5. Congestion features (airport traffic)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or FEATURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Encoders and scalers (fitted during training)
        self.label_encoders = {}
        self.scaler = StandardScaler()

        # Historical statistics (computed from training data)
        self.historical_stats = {}

    def create_all_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            df: Preprocessed DataFrame
            fit: If True, fit encoders/scalers (training). If False, transform only (inference).

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering. Shape: {df.shape}")

        # 1. Temporal features
        df = self._create_temporal_features(df)

        # 2. Flight features
        df = self._create_flight_features(df)

        # 3. Weather-derived features
        df = self._create_weather_features(df)

        # 4. Historical features (requires fit on training data)
        if fit:
            df = self._create_historical_features(df, fit=True)
        else:
            df = self._create_historical_features(df, fit=False)

        # 5. Congestion features
        df = self._create_congestion_features(df)

        # 6. Interaction features
        df = self._create_interaction_features(df)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.

        These capture patterns like:
        - Rush hours have more delays
        - Fridays/Sundays have more traffic
        - Holidays have different patterns
        - Seasons affect weather-related delays
        """
        logger.info("Creating temporal features...")

        # Basic time features (if not already created)
        if "hour" not in df.columns:
            dep_col = "scheduled_departure" if "scheduled_departure" in df.columns else "crs_dep_time"
            if dep_col in df.columns:
                df["hour"] = (df[dep_col] // 100).astype(int).clip(0, 23)

        if "hour" in df.columns:
            # Peak hours (6-9 AM, 4-8 PM)
            df["is_morning_peak"] = df["hour"].between(6, 9).astype(int)
            df["is_evening_peak"] = df["hour"].between(16, 20).astype(int)
            df["is_peak_hour"] = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)

            # Late night (less staff, potential delays)
            df["is_late_night"] = df["hour"].between(22, 23).astype(int) | df["hour"].between(0, 5).astype(int)

            # Hour buckets (cyclical encoding for hour)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        if "day_of_week" in df.columns:
            # Weekend flag
            if "is_weekend" not in df.columns:
                df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Business days (Tue-Thu typically less busy)
            df["is_midweek"] = df["day_of_week"].isin([1, 2, 3]).astype(int)

            # Travel-heavy days (Fri, Sun, Mon)
            df["is_travel_heavy_day"] = df["day_of_week"].isin([0, 4, 6]).astype(int)

            # Cyclical encoding for day of week
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        if "month" in df.columns:
            # Season
            df["season"] = df["month"].map({
                12: "winter", 1: "winter", 2: "winter",
                3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer",
                9: "fall", 10: "fall", 11: "fall"
            })

            # Peak travel months
            df["is_summer_travel"] = df["month"].isin([6, 7, 8]).astype(int)
            df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

            # Cyclical encoding for month
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # US holidays (simplified - major holidays)
        if "month" in df.columns and "day_of_month" in df.columns:
            df["is_holiday"] = self._is_holiday(df["month"], df["day_of_month"])

        return df

    def _is_holiday(self, month: pd.Series, day: pd.Series) -> pd.Series:
        """Check if date is near a US holiday."""
        # Major US holidays (approximate dates)
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (12, 31), # New Year's Eve
        ]

        # Thanksgiving (4th Thursday of November - approximate as Nov 22-28)
        is_thanksgiving = (month == 11) & (day >= 22) & (day <= 28)

        # Fixed holidays
        is_fixed_holiday = pd.Series([False] * len(month))
        for h_month, h_day in holidays:
            is_fixed_holiday |= ((month == h_month) & (day >= h_day - 2) & (day <= h_day + 2))

        return (is_thanksgiving | is_fixed_holiday).astype(int)

    def _create_flight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create flight-related features.

        These capture patterns like:
        - Certain airlines have better/worse on-time performance
        - Longer routes have different delay patterns
        - Specific routes may have consistent delays
        """
        logger.info("Creating flight features...")

        origin_col = "origin_airport" if "origin_airport" in df.columns else "origin"
        dest_col = "dest_airport" if "dest_airport" in df.columns else "dest"
        airline_col = "airline" if "airline" in df.columns else "op_unique_carrier"

        # Route identifier
        if origin_col in df.columns and dest_col in df.columns:
            df["route"] = df[origin_col] + "_" + df[dest_col]

        # Distance buckets
        if "distance" in df.columns:
            df["distance_category"] = pd.cut(
                df["distance"],
                bins=[0, 500, 1000, 2000, np.inf],
                labels=["short", "medium", "long", "ultra_long"]
            )

            # Log distance (reduces skewness)
            df["log_distance"] = np.log1p(df["distance"])

            # Distance squared (for non-linear effects)
            df["distance_squared"] = df["distance"] ** 2

        # Scheduled duration features
        if "scheduled_duration" in df.columns:
            df["log_duration"] = np.log1p(df["scheduled_duration"])

            # Speed estimate (mph)
            if "distance" in df.columns:
                df["avg_speed"] = df["distance"] / (df["scheduled_duration"] / 60)
                df["avg_speed"] = df["avg_speed"].replace([np.inf, -np.inf], np.nan).fillna(500)

        # Flight number prefix (first digit often indicates type)
        if "flight_number" in df.columns:
            df["flight_number_prefix"] = df["flight_number"].astype(str).str[0]

        # Is connecting hub (major hubs have more connections)
        hubs = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO"]
        if origin_col in df.columns:
            df["origin_is_hub"] = df[origin_col].isin(hubs).astype(int)
        if dest_col in df.columns:
            df["dest_is_hub"] = df[dest_col].isin(hubs).astype(int)

        return df

    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-derived features.

        Categorize weather conditions into operational impact levels.
        """
        logger.info("Creating weather features...")

        # Weather severity at origin
        for prefix in ["origin", "dest"]:
            vis_col = f"{prefix}_visibility_m"
            wind_col = f"{prefix}_wind_speed_ms"
            precip_col = f"{prefix}_precipitation_mm"
            temp_col = f"{prefix}_temp_c"

            # Visibility categories
            if vis_col in df.columns:
                df[f"{prefix}_low_visibility"] = (df[vis_col] < 5000).astype(int)
                df[f"{prefix}_very_low_visibility"] = (df[vis_col] < 1000).astype(int)

            # Wind severity
            if wind_col in df.columns:
                df[f"{prefix}_high_wind"] = (df[wind_col] > 10).astype(int)
                df[f"{prefix}_severe_wind"] = (df[wind_col] > 20).astype(int)

            # Precipitation
            if precip_col in df.columns:
                df[f"{prefix}_has_precipitation"] = (df[precip_col] > 0).astype(int)
                df[f"{prefix}_heavy_precipitation"] = (df[precip_col] > 10).astype(int)

            # Temperature extremes
            if temp_col in df.columns:
                df[f"{prefix}_extreme_cold"] = (df[temp_col] < -10).astype(int)
                df[f"{prefix}_extreme_hot"] = (df[temp_col] > 35).astype(int)

        # Combined weather severity score
        severity_cols = [
            "origin_low_visibility", "dest_low_visibility",
            "origin_high_wind", "dest_high_wind",
            "origin_has_precipitation", "dest_has_precipitation"
        ]

        existing_severity_cols = [col for col in severity_cols if col in df.columns]
        if existing_severity_cols:
            df["weather_severity_score"] = df[existing_severity_cols].sum(axis=1)

        # Weather difference between origin and destination
        if "origin_temp_c" in df.columns and "dest_temp_c" in df.columns:
            df["temp_difference"] = abs(df["origin_temp_c"] - df["dest_temp_c"])

        return df

    def _create_historical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create features based on historical delay patterns.

        These are computed from the training data and stored for inference.
        """
        logger.info("Creating historical features...")

        origin_col = "origin_airport" if "origin_airport" in df.columns else "origin"
        dest_col = "dest_airport" if "dest_airport" in df.columns else "dest"
        airline_col = "airline" if "airline" in df.columns else "op_unique_carrier"

        if fit and "is_delayed" in df.columns:
            # Compute and store historical statistics
            self._compute_historical_stats(df, origin_col, dest_col, airline_col)

        # Apply historical features
        if self.historical_stats:
            # Airline delay rate
            if "airline_delay_rate" in self.historical_stats and airline_col in df.columns:
                df["airline_delay_rate"] = df[airline_col].map(
                    self.historical_stats["airline_delay_rate"]
                ).fillna(self.historical_stats.get("overall_delay_rate", 0.2))

            # Origin airport delay rate
            if "origin_delay_rate" in self.historical_stats and origin_col in df.columns:
                df["origin_delay_rate"] = df[origin_col].map(
                    self.historical_stats["origin_delay_rate"]
                ).fillna(self.historical_stats.get("overall_delay_rate", 0.2))

            # Destination airport delay rate
            if "dest_delay_rate" in self.historical_stats and dest_col in df.columns:
                df["dest_delay_rate"] = df[dest_col].map(
                    self.historical_stats["dest_delay_rate"]
                ).fillna(self.historical_stats.get("overall_delay_rate", 0.2))

            # Route delay rate
            if "route_delay_rate" in self.historical_stats and "route" in df.columns:
                df["route_delay_rate"] = df["route"].map(
                    self.historical_stats["route_delay_rate"]
                ).fillna(self.historical_stats.get("overall_delay_rate", 0.2))

            # Average delay by airline
            if "airline_avg_delay" in self.historical_stats and airline_col in df.columns:
                df["airline_avg_delay"] = df[airline_col].map(
                    self.historical_stats["airline_avg_delay"]
                ).fillna(0)

            # Average delay by hour
            if "hour_delay_rate" in self.historical_stats and "hour" in df.columns:
                df["hour_delay_rate"] = df["hour"].map(
                    self.historical_stats["hour_delay_rate"]
                ).fillna(self.historical_stats.get("overall_delay_rate", 0.2))

        return df

    def _compute_historical_stats(
        self,
        df: pd.DataFrame,
        origin_col: str,
        dest_col: str,
        airline_col: str
    ):
        """Compute and store historical statistics from training data."""
        logger.info("Computing historical statistics...")

        # Overall delay rate
        self.historical_stats["overall_delay_rate"] = df["is_delayed"].mean()

        # Delay rate by airline
        if airline_col in df.columns:
            self.historical_stats["airline_delay_rate"] = df.groupby(airline_col)["is_delayed"].mean().to_dict()

        # Delay rate by origin airport
        if origin_col in df.columns:
            self.historical_stats["origin_delay_rate"] = df.groupby(origin_col)["is_delayed"].mean().to_dict()

        # Delay rate by destination airport
        if dest_col in df.columns:
            self.historical_stats["dest_delay_rate"] = df.groupby(dest_col)["is_delayed"].mean().to_dict()

        # Delay rate by route
        if "route" in df.columns:
            self.historical_stats["route_delay_rate"] = df.groupby("route")["is_delayed"].mean().to_dict()

        # Average delay minutes by airline
        if airline_col in df.columns and "arrival_delay_minutes" in df.columns:
            self.historical_stats["airline_avg_delay"] = df.groupby(airline_col)["arrival_delay_minutes"].mean().to_dict()

        # Delay rate by hour
        if "hour" in df.columns:
            self.historical_stats["hour_delay_rate"] = df.groupby("hour")["is_delayed"].mean().to_dict()

        logger.info(f"Computed {len(self.historical_stats)} historical statistics")

    def _create_congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create airport congestion features.

        Estimates how busy the airport is at the scheduled departure time.
        """
        logger.info("Creating congestion features...")

        origin_col = "origin_airport" if "origin_airport" in df.columns else "origin"
        date_col = "flight_date" if "flight_date" in df.columns else "fl_date"

        # Count flights per airport-hour
        if origin_col in df.columns and date_col in df.columns and "hour" in df.columns:
            # Create a composite key for grouping
            df["date_hour"] = df[date_col].astype(str) + "_" + df["hour"].astype(str)
            df["origin_date_hour"] = df[origin_col] + "_" + df["date_hour"]

            # Count departures per origin airport per hour
            origin_counts = df.groupby("origin_date_hour").size()
            df["origin_departures_hour"] = df["origin_date_hour"].map(origin_counts)

            # Normalize by max (0-1 scale)
            max_deps = df["origin_departures_hour"].max()
            if max_deps > 0:
                df["origin_congestion"] = df["origin_departures_hour"] / max_deps

            # Clean up temporary columns
            df = df.drop(columns=["date_hour", "origin_date_hour"], errors="ignore")

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different feature groups.

        These capture complex patterns like:
        - Airline + Weather interaction
        - Time + Airport interaction
        """
        logger.info("Creating interaction features...")

        # Peak hour + Hub interaction
        if "is_peak_hour" in df.columns and "origin_is_hub" in df.columns:
            df["peak_hour_at_hub"] = df["is_peak_hour"] * df["origin_is_hub"]

        # Bad weather + Long flight interaction
        if "weather_severity_score" in df.columns and "distance" in df.columns:
            median_distance = df["distance"].median()
            df["long_flight"] = (df["distance"] > median_distance).astype(int)
            df["bad_weather_long_flight"] = df["weather_severity_score"] * df["long_flight"]

        # Weekend + Holiday interaction
        if "is_weekend" in df.columns and "is_holiday" in df.columns:
            df["holiday_weekend"] = df["is_weekend"] * df["is_holiday"]

        return df

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.

        For tree-based models (XGBoost, LightGBM), label encoding is often sufficient.
        For linear models, consider one-hot encoding.

        Args:
            df: DataFrame with features
            categorical_cols: List of columns to encode. If None, auto-detect.
            fit: If True, fit encoders. If False, use existing encoders.

        Returns:
            DataFrame with encoded features
        """
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                # Handle unseen labels during transform
                df[col] = df[col].astype(str)
                le.fit(df[col])
                self.label_encoders[col] = le
                df[f"{col}_encoded"] = le.transform(df[col])
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    # Handle unseen labels
                    df[f"{col}_encoded"] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def get_feature_names(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Get lists of feature names by type.

        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        # Exclude target and identifier columns
        exclude_cols = [
            "is_delayed", "arrival_delay_minutes", "delay_category",
            "flight_date", "fl_date", "year"
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

        return numeric_cols, categorical_cols

    def save_features(self, df: pd.DataFrame, filename: str = "features.parquet") -> Path:
        """Save engineered features to parquet format."""
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved features to: {output_path}")
        return output_path

    def save_feature_metadata(self, filename: str = "feature_metadata.json"):
        """Save feature engineering metadata (encoders, stats)."""
        import json
        import joblib

        # Save label encoders
        encoders_path = self.output_dir / "label_encoders.joblib"
        joblib.dump(self.label_encoders, encoders_path)

        # Save historical stats
        stats_path = self.output_dir / "historical_stats.json"
        with open(stats_path, "w") as f:
            # Convert numpy types to Python types
            stats_serializable = {}
            for key, value in self.historical_stats.items():
                if isinstance(value, dict):
                    stats_serializable[key] = {k: float(v) for k, v in value.items()}
                else:
                    stats_serializable[key] = float(value)
            json.dump(stats_serializable, f, indent=2)

        logger.info(f"Saved feature metadata to: {self.output_dir}")


if __name__ == "__main__":
    # Test feature engineering
    from .download import FlightDataDownloader, WeatherDataDownloader
    from .preprocessing import DataPreprocessor

    print("Loading and preprocessing data...")
    flight_dl = FlightDataDownloader()
    weather_dl = WeatherDataDownloader()
    preprocessor = DataPreprocessor()

    flights = flight_dl.load_kaggle_sample()
    weather = weather_dl.create_sample_weather_data(flights)
    processed = preprocessor.preprocess_flights(flights)
    merged = preprocessor.merge_weather_data(processed, weather)

    print("\nEngineering features...")
    feature_eng = FeatureEngineer()
    featured = feature_eng.create_all_features(merged, fit=True)

    print(f"\nFinal shape: {featured.shape}")
    print(f"\nFeature columns ({len(featured.columns)}):")
    numeric, categorical = feature_eng.get_feature_names(featured)
    print(f"Numeric features ({len(numeric)}): {numeric[:10]}...")
    print(f"Categorical features ({len(categorical)}): {categorical}")
