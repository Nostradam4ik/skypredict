"""
Model Training Module for SkyPredict
=====================================

Trains ensemble models for flight delay prediction:
- Stage 1: Classification (is_delayed: yes/no)
- Stage 2: Regression (delay_minutes for delayed flights)

Models: XGBoost, LightGBM, CatBoost
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from ..config import ml_config, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates ML models for flight delay prediction.

    Features:
    - Two-stage approach: classification + regression
    - Multiple model support (XGBoost, LightGBM, CatBoost)
    - SMOTE for class imbalance
    - Cross-validation
    - Hyperparameter tuning support
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or MODELS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = None
        self.regressor = None
        self.feature_names = None
        self.training_metrics = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "is_delayed",
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Args:
            df: DataFrame with features
            target_col: Target column name
            feature_cols: List of feature columns. If None, auto-select.

        Returns:
            Tuple of (X, y)
        """
        # Auto-select feature columns if not provided
        if feature_cols is None:
            exclude_cols = [
                "is_delayed", "arrival_delay_minutes", "delay_category",
                "flight_date", "fl_date", "year", "flight_number",
                "route", "day_name", "season", "distance_category"
            ]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Keep only numeric columns for now (encoded categoricals)
            numeric_df = df[feature_cols].select_dtypes(include=[np.number])
            feature_cols = numeric_df.columns.tolist()

        self.feature_names = feature_cols
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle any remaining NaN values
        X = X.fillna(X.median())

        logger.info(f"Prepared data: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"Features: {len(feature_cols)} columns")

        return X, y

    def train_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "xgboost",
        use_smote: bool = True,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train classification model (is_delayed: yes/no).

        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: "xgboost", "lightgbm", or "catboost"
            use_smote: Whether to use SMOTE for class imbalance
            test_size: Test set proportion

        Returns:
            Dictionary with metrics and model info
        """
        logger.info(f"Training {model_type} classifier...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=ml_config.RANDOM_STATE, stratify=y
        )

        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")

        # Apply SMOTE for class imbalance
        if use_smote:
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(
                sampling_strategy=ml_config.SMOTE_SAMPLING_STRATEGY,
                random_state=ml_config.RANDOM_STATE
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {len(X_train_resampled)} samples")
            logger.info(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Create model
        self.classifier = self._create_classifier(model_type)

        # Train
        start_time = datetime.now()
        self.classifier.fit(X_train_resampled, y_train_resampled)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]

        metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        metrics["training_time_seconds"] = training_time
        metrics["model_type"] = model_type
        metrics["use_smote"] = use_smote

        self.training_metrics["classifier"] = metrics

        logger.info(f"Classification Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def train_regressor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "xgboost",
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train regression model (delay_minutes for delayed flights).

        Args:
            X: Feature DataFrame
            y: Target Series (arrival_delay_minutes)
            model_type: "xgboost", "lightgbm", or "catboost"
            test_size: Test set proportion

        Returns:
            Dictionary with metrics and model info
        """
        logger.info(f"Training {model_type} regressor...")

        # Filter to only delayed flights for regression
        delayed_mask = y > ml_config.DELAY_THRESHOLD
        X_delayed = X[delayed_mask]
        y_delayed = y[delayed_mask]

        logger.info(f"Training regressor on {len(X_delayed)} delayed flights")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_delayed, y_delayed, test_size=test_size, random_state=ml_config.RANDOM_STATE
        )

        # Create model
        self.regressor = self._create_regressor(model_type)

        # Train
        start_time = datetime.now()
        self.regressor.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        y_pred = self.regressor.predict(X_test)

        metrics = self._calculate_regression_metrics(y_test, y_pred)
        metrics["training_time_seconds"] = training_time
        metrics["model_type"] = model_type

        self.training_metrics["regressor"] = metrics

        logger.info(f"Regression Results:")
        logger.info(f"  MAE: {metrics['mae']:.2f} minutes")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} minutes")
        logger.info(f"  R2 Score: {metrics['r2']:.4f}")

        return metrics

    def _create_classifier(self, model_type: str):
        """Create classifier based on model type."""
        if model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=ml_config.RANDOM_STATE,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=ml_config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            )
        elif model_type == "catboost":
            return CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=3,
                random_seed=ml_config.RANDOM_STATE,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_regressor(self, model_type: str):
        """Create regressor based on model type."""
        if model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=ml_config.RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=ml_config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            )
        elif model_type == "catboost":
            return CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=3,
                random_seed=ml_config.RANDOM_STATE,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _calculate_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    def _calculate_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        model_type: str = "xgboost"
    ) -> Dict[str, float]:
        """
        Perform cross-validation for robust evaluation.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_folds: Number of CV folds
            model_type: Model type to evaluate

        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Running {n_folds}-fold cross-validation...")

        model = self._create_classifier(model_type)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=ml_config.RANDOM_STATE)

        # Calculate multiple metrics
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        metrics = {
            "cv_accuracy_mean": accuracy_scores.mean(),
            "cv_accuracy_std": accuracy_scores.std(),
            "cv_f1_mean": f1_scores.mean(),
            "cv_f1_std": f1_scores.std(),
            "cv_roc_auc_mean": roc_auc_scores.mean(),
            "cv_roc_auc_std": roc_auc_scores.std()
        }

        logger.info(f"CV Results:")
        logger.info(f"  Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
        logger.info(f"  F1 Score: {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
        logger.info(f"  ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']:.4f})")

        return metrics

    def get_feature_importance(self, model_type: str = "classifier") -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model_type: "classifier" or "regressor"

        Returns:
            DataFrame with feature names and importance scores
        """
        model = self.classifier if model_type == "classifier" else self.regressor

        if model is None:
            raise ValueError(f"No trained {model_type} available")

        importance = model.feature_importances_

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        return df

    def save_models(self, prefix: str = "skypredict"):
        """Save trained models and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save classifier
        if self.classifier:
            clf_path = self.output_dir / f"{prefix}_classifier_{timestamp}.joblib"
            joblib.dump(self.classifier, clf_path)
            logger.info(f"Saved classifier to: {clf_path}")

        # Save regressor
        if self.regressor:
            reg_path = self.output_dir / f"{prefix}_regressor_{timestamp}.joblib"
            joblib.dump(self.regressor, reg_path)
            logger.info(f"Saved regressor to: {reg_path}")

        # Save feature names
        features_path = self.output_dir / f"{prefix}_features_{timestamp}.json"
        with open(features_path, "w") as f:
            json.dump({"feature_names": self.feature_names}, f, indent=2)

        # Save metrics
        metrics_path = self.output_dir / f"{prefix}_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)

        # Save latest version pointers
        for name, path in [("classifier", clf_path if self.classifier else None),
                           ("regressor", reg_path if self.regressor else None)]:
            if path:
                latest_path = self.output_dir / f"{prefix}_{name}_latest.joblib"
                joblib.dump(joblib.load(path), latest_path)

        logger.info(f"All models saved with timestamp: {timestamp}")

    def load_models(self, prefix: str = "skypredict"):
        """Load latest trained models."""
        clf_path = self.output_dir / f"{prefix}_classifier_latest.joblib"
        reg_path = self.output_dir / f"{prefix}_regressor_latest.joblib"

        if clf_path.exists():
            self.classifier = joblib.load(clf_path)
            logger.info(f"Loaded classifier from: {clf_path}")

        if reg_path.exists():
            self.regressor = joblib.load(reg_path)
            logger.info(f"Loaded regressor from: {reg_path}")


class EnsembleModel:
    """
    Ensemble model combining multiple base models.

    Strategies:
    - Voting (hard/soft)
    - Averaging
    - Stacking
    """

    def __init__(self):
        self.models = {}
        self.weights = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using weighted averaging.

        Args:
            X: Feature DataFrame

        Returns:
            Averaged probability predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            weight = self.weights[name] / total_weight
            predictions.append(proba * weight)

        return np.sum(predictions, axis=0)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature DataFrame
            threshold: Classification threshold

        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


if __name__ == "__main__":
    # Test training pipeline
    from ..data.download import FlightDataDownloader, WeatherDataDownloader
    from ..data.preprocessing import DataPreprocessor
    from ..data.feature_engineering import FeatureEngineer

    print("Loading and preparing data...")
    flight_dl = FlightDataDownloader()
    weather_dl = WeatherDataDownloader()
    preprocessor = DataPreprocessor()
    feature_eng = FeatureEngineer()

    # Load sample data
    flights = flight_dl.load_kaggle_sample()
    weather = weather_dl.create_sample_weather_data(flights)

    # Preprocess
    processed = preprocessor.preprocess_flights(flights)
    merged = preprocessor.merge_weather_data(processed, weather)

    # Feature engineering
    featured = feature_eng.create_all_features(merged, fit=True)

    print("\nTraining models...")
    trainer = ModelTrainer()

    # Prepare data
    X, y = trainer.prepare_data(featured, target_col="is_delayed")

    # Train classifier
    clf_metrics = trainer.train_classifier(X, y, model_type="xgboost", use_smote=True)

    # Train regressor
    X_reg, y_reg = trainer.prepare_data(featured, target_col="arrival_delay_minutes")
    reg_metrics = trainer.train_regressor(X_reg, y_reg, model_type="xgboost")

    # Feature importance
    print("\nTop 10 Features:")
    importance = trainer.get_feature_importance("classifier")
    print(importance.head(10))

    # Save models
    trainer.save_models()
