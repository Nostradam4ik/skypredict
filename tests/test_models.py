"""
Model Tests for SkyPredict
"""

import pytest
import numpy as np
import pandas as pd


class TestDataPreprocessing:
    """Tests for data preprocessing module."""

    def test_create_sample_dataset(self):
        """Test synthetic dataset creation."""
        from src.data.download import FlightDataDownloader

        downloader = FlightDataDownloader()
        df = downloader._create_sample_dataset(n_samples=1000)

        assert len(df) > 0
        assert "ORIGIN" in df.columns or "origin" in df.columns.str.lower()
        assert "DEST" in df.columns or "dest" in df.columns.str.lower()

    def test_preprocessor_removes_cancelled(self):
        """Test that preprocessor removes cancelled flights."""
        from src.data.preprocessing import DataPreprocessor
        from src.data.download import FlightDataDownloader

        downloader = FlightDataDownloader()
        df = downloader._create_sample_dataset(n_samples=1000)

        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_flights(df, remove_cancelled=True)

        if "cancelled" in processed.columns:
            assert processed["cancelled"].sum() == 0

    def test_preprocessor_creates_targets(self):
        """Test that preprocessor creates target variables."""
        from src.data.preprocessing import DataPreprocessor
        from src.data.download import FlightDataDownloader

        downloader = FlightDataDownloader()
        df = downloader._create_sample_dataset(n_samples=1000)

        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_flights(df)

        assert "is_delayed" in processed.columns
        assert "arrival_delay_minutes" in processed.columns
        assert processed["is_delayed"].isin([0, 1]).all()


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_temporal_features(self):
        """Test temporal feature creation."""
        from src.data.feature_engineering import FeatureEngineer

        # Create sample data
        df = pd.DataFrame({
            "flight_date": pd.date_range("2024-01-01", periods=100),
            "scheduled_departure": [1400] * 100,
            "is_delayed": np.random.randint(0, 2, 100)
        })
        df["month"] = df["flight_date"].dt.month
        df["day_of_week"] = df["flight_date"].dt.dayofweek
        df["day_of_month"] = df["flight_date"].dt.day

        engineer = FeatureEngineer()
        featured = engineer._create_temporal_features(df)

        assert "is_weekend" in featured.columns
        assert "hour_sin" in featured.columns or "hour" not in df.columns

    def test_no_data_leakage(self):
        """Test that leakage features are removed."""
        from src.data.preprocessing import DataPreprocessor, DataQualityChecker
        from src.data.download import FlightDataDownloader

        downloader = FlightDataDownloader()
        df = downloader._create_sample_dataset(n_samples=1000)

        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess_flights(df)

        checker = DataQualityChecker(processed)
        leakage = checker.check_leakage_features()

        # Should have no leakage features
        assert len(leakage) == 0, f"Leakage features found: {leakage}"


class TestModelTraining:
    """Tests for model training module."""

    def test_prepare_data(self):
        """Test data preparation for training."""
        from src.models.train import ModelTrainer

        # Create sample data
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "is_delayed": np.random.randint(0, 2, 100)
        })

        trainer = ModelTrainer()
        X, y = trainer.prepare_data(df, target_col="is_delayed")

        assert len(X) == len(y)
        assert "is_delayed" not in X.columns

    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        from src.models.train import ModelTrainer

        trainer = ModelTrainer()

        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9])

        metrics = trainer._calculate_classification_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestPrediction:
    """Tests for prediction module."""

    def test_risk_level_calculation(self):
        """Test risk level calculation."""
        from src.models.predict import FlightDelayPredictor

        predictor = FlightDelayPredictor.__new__(FlightDelayPredictor)

        # Test risk levels
        probs = np.array([0.1, 0.4, 0.7, 0.9])
        delays = np.array([0, 20, 45, 90])

        risk_levels = predictor._calculate_risk_levels(probs, delays)

        assert risk_levels[0] == "LOW"
        assert risk_levels[1] == "MEDIUM"
        assert risk_levels[2] == "HIGH"
        assert risk_levels[3] == "SEVERE"


# Pytest fixtures
@pytest.fixture
def sample_features():
    """Sample feature DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "month": np.random.randint(1, 13, 1000),
        "day_of_week": np.random.randint(0, 7, 1000),
        "hour": np.random.randint(0, 24, 1000),
        "distance": np.random.uniform(100, 3000, 1000),
        "origin_is_hub": np.random.randint(0, 2, 1000),
        "dest_is_hub": np.random.randint(0, 2, 1000),
        "weather_severity_score": np.random.randint(0, 6, 1000),
        "airline_delay_rate": np.random.uniform(0.1, 0.3, 1000),
        "is_delayed": np.random.randint(0, 2, 1000),
        "arrival_delay_minutes": np.random.uniform(-30, 120, 1000)
    })
