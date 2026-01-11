"""
Data Preprocessing Module for SkyPredict
=========================================

Cleans and prepares raw flight and weather data for feature engineering.
Handles missing values, outliers, and data quality issues.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

import pandas as pd
import numpy as np

from ..config import ml_config, airport_config, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw flight and weather data.

    Key responsibilities:
    - Remove cancelled/diverted flights
    - Handle missing values
    - Remove data leakage features
    - Create target variables
    - Filter to major airports/airlines
    - Merge weather data
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_flights(
        self,
        df: pd.DataFrame,
        remove_cancelled: bool = True,
        filter_airports: bool = True,
        filter_airlines: bool = True
    ) -> pd.DataFrame:
        """
        Main preprocessing pipeline for flight data.

        Args:
            df: Raw flight DataFrame
            remove_cancelled: Remove cancelled/diverted flights
            filter_airports: Keep only major airports
            filter_airlines: Keep only major airlines

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting preprocessing. Initial shape: {df.shape}")

        # Standardize column names
        df = self._standardize_columns(df)

        # Remove cancelled and diverted flights
        if remove_cancelled:
            df = self._remove_cancelled_diverted(df)

        # Filter to major airports
        if filter_airports:
            df = self._filter_airports(df)

        # Filter to major airlines
        if filter_airlines:
            df = self._filter_airlines(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove outliers
        df = self._remove_outliers(df)

        # Create target variables
        df = self._create_targets(df)

        # Remove data leakage features
        df = self._remove_leakage_features(df)

        # Parse datetime features
        df = self._parse_datetime(df)

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        # Common column mappings
        column_mapping = {
            "op_unique_carrier": "airline",
            "op_carrier_fl_num": "flight_number",
            "fl_date": "flight_date",
            "crs_dep_time": "scheduled_departure",
            "dep_time": "actual_departure",
            "crs_arr_time": "scheduled_arrival",
            "arr_time": "actual_arrival",
            "crs_elapsed_time": "scheduled_duration",
            "actual_elapsed_time": "actual_duration",
            "dep_delay": "departure_delay",
            "arr_delay": "arrival_delay",
            "origin": "origin_airport",
            "dest": "dest_airport",
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        return df

    def _remove_cancelled_diverted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove cancelled and diverted flights."""
        initial_count = len(df)

        if "cancelled" in df.columns:
            df = df[df["cancelled"] == 0]

        if "diverted" in df.columns:
            df = df[df["diverted"] == 0]

        removed = initial_count - len(df)
        logger.info(f"Removed {removed} cancelled/diverted flights ({removed/initial_count*100:.2f}%)")

        return df

    def _filter_airports(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only flights from/to major airports."""
        initial_count = len(df)

        origin_col = "origin_airport" if "origin_airport" in df.columns else "origin"
        dest_col = "dest_airport" if "dest_airport" in df.columns else "dest"

        df = df[
            (df[origin_col].isin(airport_config.TOP_AIRPORTS)) &
            (df[dest_col].isin(airport_config.TOP_AIRPORTS))
        ]

        removed = initial_count - len(df)
        logger.info(f"Filtered to major airports. Removed {removed} flights ({removed/initial_count*100:.2f}%)")

        return df

    def _filter_airlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only major airlines."""
        initial_count = len(df)

        airline_col = "airline" if "airline" in df.columns else "op_unique_carrier"

        if airline_col in df.columns:
            df = df[df[airline_col].isin(airport_config.MAJOR_AIRLINES)]
            removed = initial_count - len(df)
            logger.info(f"Filtered to major airlines. Removed {removed} flights ({removed/initial_count*100:.2f}%)")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately."""
        # Log missing value counts
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        logger.info(f"Missing values:\n{missing[missing > 0]}")

        # Drop rows with missing target (arrival_delay)
        delay_col = "arrival_delay" if "arrival_delay" in df.columns else "arr_delay"
        if delay_col in df.columns:
            before = len(df)
            df = df.dropna(subset=[delay_col])
            logger.info(f"Dropped {before - len(df)} rows with missing arrival delay")

        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(mode_val)

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove extreme outliers from delay columns."""
        delay_col = "arrival_delay" if "arrival_delay" in df.columns else "arr_delay"

        if delay_col in df.columns:
            initial_count = len(df)

            # Remove extreme delays (> 6 hours early or > 12 hours late)
            df = df[(df[delay_col] >= -360) & (df[delay_col] <= 720)]

            removed = initial_count - len(df)
            logger.info(f"Removed {removed} outliers ({removed/initial_count*100:.2f}%)")

        return df

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for classification and regression."""
        delay_col = "arrival_delay" if "arrival_delay" in df.columns else "arr_delay"

        if delay_col in df.columns:
            # Binary classification target (is_delayed)
            # FAA definition: delay > 15 minutes
            df["is_delayed"] = (df[delay_col] > ml_config.DELAY_THRESHOLD).astype(int)

            # Regression target (arrival_delay_minutes)
            df["arrival_delay_minutes"] = df[delay_col]

            # Multi-class target (delay_category)
            df["delay_category"] = pd.cut(
                df[delay_col],
                bins=[-np.inf, -15, 15, 60, 120, np.inf],
                labels=["early", "on_time", "minor_delay", "moderate_delay", "severe_delay"]
            )

            # Log class distribution
            delay_pct = df["is_delayed"].mean() * 100
            logger.info(f"Delay distribution: {delay_pct:.2f}% delayed, {100-delay_pct:.2f}% on-time")

        return df

    def _remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that cause data leakage.

        These features are not available at prediction time (2h before flight):
        - departure_delay, taxi_out, wheels_off, air_time
        - actual_arrival, actual_duration
        - delay breakdown columns
        """
        leakage_cols = [
            # Post-departure features
            "departure_delay", "dep_delay", "taxi_out", "taxi_in",
            "wheels_off", "wheels_on", "air_time",
            "actual_departure", "dep_time", "actual_arrival", "arr_time",
            "actual_duration", "actual_elapsed_time",

            # Delay breakdown (only known after arrival)
            "carrier_delay", "weather_delay", "nas_delay",
            "security_delay", "late_aircraft_delay",

            # Cancellation/diversion flags (we already filtered these)
            "cancelled", "cancellation_code", "diverted",
        ]

        # Keep track of original arrival_delay for regression target
        cols_to_drop = [col for col in leakage_cols if col in df.columns]

        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} leakage features: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop, errors="ignore")

        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and extract datetime components."""
        date_col = "flight_date" if "flight_date" in df.columns else "fl_date"

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df["year"] = df[date_col].dt.year
            df["month"] = df[date_col].dt.month
            df["day_of_month"] = df[date_col].dt.day
            df["day_of_week"] = df[date_col].dt.dayofweek
            df["day_name"] = df[date_col].dt.day_name()
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Parse scheduled departure time
        dep_time_col = "scheduled_departure" if "scheduled_departure" in df.columns else "crs_dep_time"
        if dep_time_col in df.columns:
            df["hour"] = (df[dep_time_col] // 100).astype(int).clip(0, 23)
            df["minute"] = (df[dep_time_col] % 100).astype(int).clip(0, 59)

        return df

    def merge_weather_data(
        self,
        flight_df: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge weather data with flight data.

        Joins weather data for both origin and destination airports.

        Args:
            flight_df: Preprocessed flight DataFrame
            weather_df: Weather DataFrame with columns [date, airport, ...]

        Returns:
            Merged DataFrame with weather features
        """
        logger.info("Merging weather data...")

        date_col = "flight_date" if "flight_date" in flight_df.columns else "fl_date"
        origin_col = "origin_airport" if "origin_airport" in flight_df.columns else "origin"
        dest_col = "dest_airport" if "dest_airport" in flight_df.columns else "dest"

        # Ensure date columns are compatible
        flight_df[date_col] = pd.to_datetime(flight_df[date_col]).dt.date
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

        # Merge origin weather
        origin_weather = weather_df.add_prefix("origin_")
        origin_weather = origin_weather.rename(columns={
            "origin_date": "date",
            "origin_airport": "airport"
        })

        flight_df = flight_df.merge(
            origin_weather,
            left_on=[date_col, origin_col],
            right_on=["date", "airport"],
            how="left"
        ).drop(columns=["date", "airport"], errors="ignore")

        # Merge destination weather
        dest_weather = weather_df.add_prefix("dest_")
        dest_weather = dest_weather.rename(columns={
            "dest_date": "date",
            "dest_airport": "airport"
        })

        flight_df = flight_df.merge(
            dest_weather,
            left_on=[date_col, dest_col],
            right_on=["date", "airport"],
            how="left"
        ).drop(columns=["date", "airport"], errors="ignore")

        logger.info(f"Weather merge complete. Shape: {flight_df.shape}")

        return flight_df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str = "flights_processed.parquet"
    ) -> Path:
        """Save processed data to parquet format."""
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to: {output_path}")
        return output_path

    def load_processed_data(self, filename: str = "flights_processed.parquet") -> pd.DataFrame:
        """Load processed data from parquet format."""
        input_path = self.output_dir / filename
        if input_path.exists():
            df = pd.read_parquet(input_path)
            logger.info(f"Loaded processed data from: {input_path}")
            return df
        else:
            raise FileNotFoundError(f"Processed data not found: {input_path}")


class DataQualityChecker:
    """Checks data quality and generates reports."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_report(self) -> dict:
        """Generate a comprehensive data quality report."""
        report = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_pct": (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            "duplicates": self.df.duplicated().sum(),
            "memory_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
        }

        # Numeric column statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        report["numeric_stats"] = self.df[numeric_cols].describe().to_dict()

        # Categorical column statistics
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        report["categorical_unique"] = {col: self.df[col].nunique() for col in categorical_cols}

        return report

    def check_leakage_features(self) -> List[str]:
        """Check if any leakage features are still present."""
        leakage_found = []
        for col in ml_config.LEAKAGE_FEATURES:
            if col in self.df.columns or col.lower() in [c.lower() for c in self.df.columns]:
                leakage_found.append(col)
        return leakage_found

    def check_class_balance(self) -> dict:
        """Check target variable class balance."""
        if "is_delayed" in self.df.columns:
            counts = self.df["is_delayed"].value_counts()
            return {
                "on_time": int(counts.get(0, 0)),
                "delayed": int(counts.get(1, 0)),
                "imbalance_ratio": round(counts.get(0, 1) / max(counts.get(1, 1), 1), 2)
            }
        return {}


if __name__ == "__main__":
    # Test preprocessing
    from .download import FlightDataDownloader, WeatherDataDownloader

    print("Loading sample data...")
    flight_dl = FlightDataDownloader()
    weather_dl = WeatherDataDownloader()

    flights = flight_dl.load_kaggle_sample()
    weather = weather_dl.create_sample_weather_data(flights)

    print("\nPreprocessing flights...")
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_flights(flights)

    print("\nMerging weather data...")
    final = preprocessor.merge_weather_data(processed, weather)

    print("\nData quality check...")
    checker = DataQualityChecker(final)
    report = checker.generate_report()
    print(f"Final shape: {report['shape']}")
    print(f"Leakage features found: {checker.check_leakage_features()}")
    print(f"Class balance: {checker.check_class_balance()}")
