#!/usr/bin/env python
"""
SkyPredict Model Training Script
=================================

Complete pipeline for training flight delay prediction models.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --model xgboost --no-smote
    python scripts/train_models.py --data path/to/data.csv

Steps:
    1. Load or generate sample data
    2. Preprocess data
    3. Engineer features
    4. Train classification model (is_delayed)
    5. Train regression model (delay_minutes)
    6. Evaluate and save models
    7. Generate feature importance report
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.download import FlightDataDownloader, WeatherDataDownloader
from src.data.preprocessing import DataPreprocessor, DataQualityChecker
from src.data.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer, EnsembleModel
from src.config import MODELS_DIR, RAW_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SkyPredict flight delay prediction models"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to custom data file (CSV or Parquet)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "catboost", "ensemble"],
        default="xgboost",
        help="Model type to train (default: xgboost)"
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE for class imbalance"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Sample size for training (default: 100000)"
    )

    return parser.parse_args()


def load_data(data_path: str = None, sample_size: int = 100000):
    """
    Load flight and weather data.

    Args:
        data_path: Path to custom data file
        sample_size: Number of samples for synthetic data

    Returns:
        Tuple of (flights_df, weather_df)
    """
    logger.info("Loading data...")

    if data_path:
        # Load custom data
        path = Path(data_path)
        if path.suffix == ".parquet":
            import pandas as pd
            flights = pd.read_parquet(path)
        else:
            import pandas as pd
            flights = pd.read_csv(path)
        logger.info(f"Loaded {len(flights)} flights from {data_path}")

        # Generate matching weather data
        weather_dl = WeatherDataDownloader()
        weather = weather_dl.create_sample_weather_data(flights)
    else:
        # Use sample data
        logger.info("No data path provided. Generating sample data...")
        flight_dl = FlightDataDownloader()
        weather_dl = WeatherDataDownloader()

        # Check for existing sample data
        sample_path = RAW_DATA_DIR / "sample_flights.csv"
        if sample_path.exists():
            import pandas as pd
            flights = pd.read_csv(sample_path)
            logger.info(f"Loaded existing sample data: {len(flights)} flights")
        else:
            flights = flight_dl._create_sample_dataset(n_samples=sample_size)
            flights.to_csv(sample_path, index=False)
            logger.info(f"Created sample data: {len(flights)} flights")

        weather = weather_dl.create_sample_weather_data(flights)

    return flights, weather


def preprocess_data(flights, weather):
    """
    Preprocess flight and weather data.

    Args:
        flights: Raw flight DataFrame
        weather: Weather DataFrame

    Returns:
        Preprocessed and merged DataFrame
    """
    logger.info("Preprocessing data...")

    preprocessor = DataPreprocessor()

    # Preprocess flights
    processed = preprocessor.preprocess_flights(
        flights,
        remove_cancelled=True,
        filter_airports=True,
        filter_airlines=True
    )

    # Merge weather
    merged = preprocessor.merge_weather_data(processed, weather)

    # Quality check
    checker = DataQualityChecker(merged)
    leakage = checker.check_leakage_features()
    if leakage:
        logger.warning(f"Data leakage features found: {leakage}")
    else:
        logger.info("No data leakage features detected ✓")

    balance = checker.check_class_balance()
    logger.info(f"Class balance: {balance}")

    return merged


def engineer_features(df):
    """
    Create ML features from preprocessed data.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")

    feature_eng = FeatureEngineer()
    featured = feature_eng.create_all_features(df, fit=True)

    # Encode categorical features
    featured = feature_eng.encode_categorical_features(featured, fit=True)

    # Save feature metadata for inference
    feature_eng.save_feature_metadata()

    logger.info(f"Created {len(featured.columns)} features")

    return featured, feature_eng


def train_models(df, model_type: str, use_smote: bool, test_size: float, cv_folds: int):
    """
    Train classification and regression models.

    Args:
        df: Feature DataFrame
        model_type: Type of model to train
        use_smote: Whether to use SMOTE
        test_size: Test set proportion
        cv_folds: Number of CV folds

    Returns:
        ModelTrainer instance with trained models
    """
    logger.info(f"Training {model_type} models...")

    trainer = ModelTrainer()

    # Prepare classification data
    X_clf, y_clf = trainer.prepare_data(df, target_col="is_delayed")

    # Cross-validation
    if cv_folds > 0:
        logger.info("Running cross-validation...")
        cv_metrics = trainer.cross_validate(X_clf, y_clf, n_folds=cv_folds, model_type=model_type)

    # Train classifier
    clf_metrics = trainer.train_classifier(
        X_clf, y_clf,
        model_type=model_type,
        use_smote=use_smote,
        test_size=test_size
    )

    # Train regressor
    X_reg, y_reg = trainer.prepare_data(df, target_col="arrival_delay_minutes")
    reg_metrics = trainer.train_regressor(
        X_reg, y_reg,
        model_type=model_type,
        test_size=test_size
    )

    # Feature importance
    logger.info("\n" + "="*50)
    logger.info("TOP 15 MOST IMPORTANT FEATURES")
    logger.info("="*50)
    importance = trainer.get_feature_importance("classifier")
    for i, row in importance.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return trainer


def train_ensemble(df, use_smote: bool, test_size: float):
    """
    Train ensemble of multiple models.

    Args:
        df: Feature DataFrame
        use_smote: Whether to use SMOTE
        test_size: Test set proportion

    Returns:
        Ensemble model and individual trainers
    """
    logger.info("Training ensemble model...")

    trainers = {}
    ensemble = EnsembleModel()

    for model_type in ["xgboost", "lightgbm"]:
        logger.info(f"\nTraining {model_type}...")
        trainer = ModelTrainer()
        X, y = trainer.prepare_data(df, target_col="is_delayed")
        trainer.train_classifier(X, y, model_type=model_type, use_smote=use_smote, test_size=test_size)
        trainers[model_type] = trainer
        ensemble.add_model(model_type, trainer.classifier, weight=1.0)

    return ensemble, trainers


def main():
    """Main training pipeline."""
    args = parse_args()

    start_time = datetime.now()
    logger.info("="*60)
    logger.info("SKYPREDICT MODEL TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"Model type: {args.model}")
    logger.info(f"SMOTE: {not args.no_smote}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"CV folds: {args.cv_folds}")
    logger.info("="*60 + "\n")

    # Step 1: Load data
    flights, weather = load_data(args.data, args.sample_size)

    # Step 2: Preprocess
    processed = preprocess_data(flights, weather)

    # Step 3: Feature engineering
    featured, feature_eng = engineer_features(processed)

    # Step 4: Train models
    if args.model == "ensemble":
        ensemble, trainers = train_ensemble(
            featured,
            use_smote=not args.no_smote,
            test_size=args.test_size
        )
        # Save first model (xgboost) as default
        trainers["xgboost"].save_models()
    else:
        trainer = train_models(
            featured,
            model_type=args.model,
            use_smote=not args.no_smote,
            test_size=args.test_size,
            cv_folds=args.cv_folds
        )
        # Save models
        trainer.save_models()

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
