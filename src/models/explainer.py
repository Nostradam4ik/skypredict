"""
Model Explainability Module for SkyPredict
===========================================

Provides SHAP-based explanations for model predictions.
Essential for understanding why a flight is predicted to be delayed.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from ..config import MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    SHAP-based model explainer for flight delay predictions.

    Features:
    - Single prediction explanations
    - Batch explanations
    - Feature importance analysis
    - Visualization generation
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize explainer with a trained model.

        Args:
            model: Trained ML model (XGBoost, LightGBM, or CatBoost)
            feature_names: List of feature names
            background_data: Sample of training data for SHAP background
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer(background_data)

    def _initialize_explainer(self, background_data: Optional[pd.DataFrame] = None):
        """Initialize SHAP explainer based on model type."""
        try:
            # Use TreeExplainer for tree-based models (faster)
            if hasattr(self.model, 'get_booster') or hasattr(self.model, 'booster_'):
                if background_data is not None:
                    # Sample background data for efficiency
                    sample_size = min(100, len(background_data))
                    background_sample = background_data.sample(n=sample_size, random_state=42)
                    self.explainer = shap.TreeExplainer(self.model, background_sample)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer")
            else:
                # Fall back to KernelExplainer for other models
                if background_data is not None:
                    sample_size = min(50, len(background_data))
                    background_sample = background_data.sample(n=sample_size, random_state=42)
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        background_sample
                    )
                    logger.info("Initialized KernelExplainer")
        except Exception as e:
            logger.error(f"Failed to initialize explainer: {e}")
            self.explainer = None

    def explain_single(
        self,
        X: pd.DataFrame,
        max_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.

        Args:
            X: Single row DataFrame with features
            max_features: Maximum number of features to include in explanation

        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}

        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multi-output (classification) vs single output
            if isinstance(shap_values, list):
                # For binary classification, use class 1 (delayed) values
                shap_values = shap_values[1]

            # Get feature values
            feature_values = X.iloc[0].to_dict() if isinstance(X, pd.DataFrame) else dict(zip(self.feature_names, X[0]))

            # Create explanation
            shap_dict = dict(zip(
                self.feature_names or X.columns.tolist(),
                shap_values[0] if len(shap_values.shape) > 1 else shap_values
            ))

            # Sort by absolute impact
            sorted_features = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:max_features]

            # Format explanation
            explanation = {
                "base_value": float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, np.ndarray) else float(self.explainer.expected_value),
                "features": []
            }

            for feature_name, shap_value in sorted_features:
                explanation["features"].append({
                    "name": feature_name,
                    "value": float(feature_values.get(feature_name, 0)),
                    "shap_value": float(shap_value),
                    "impact": "increases delay risk" if shap_value > 0 else "decreases delay risk"
                })

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {"error": str(e)}

    def explain_batch(
        self,
        X: pd.DataFrame,
        max_features: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple predictions.

        Args:
            X: DataFrame with features
            max_features: Maximum features per explanation

        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for i in range(len(X)):
            exp = self.explain_single(X.iloc[[i]], max_features)
            explanations.append(exp)
        return explanations

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP.

        Args:
            X: DataFrame with features (sample of data)

        Returns:
            DataFrame with feature importance scores
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Calculate SHAP values for all samples
        shap_values = self.explainer.shap_values(X)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            "feature": self.feature_names or X.columns.tolist(),
            "importance": importance
        }).sort_values("importance", ascending=False)

        return df

    def generate_waterfall_plot(
        self,
        X: pd.DataFrame,
        output_path: Optional[Path] = None,
        max_display: int = 10
    ) -> Optional[str]:
        """
        Generate SHAP waterfall plot for a single prediction.

        Args:
            X: Single row DataFrame
            output_path: Path to save plot (optional)
            max_display: Maximum features to display

        Returns:
            Path to saved plot or None
        """
        if self.explainer is None:
            return None

        try:
            shap_values = self.explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Create SHAP Explanation object
            exp = shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
                data=X.iloc[0].values,
                feature_names=self.feature_names or X.columns.tolist()
            )

            # Generate plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=max_display, show=False)

            if output_path:
                plt.savefig(output_path, bbox_inches="tight", dpi=150)
                plt.close()
                return str(output_path)
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error generating waterfall plot: {e}")
            return None

    def generate_summary_plot(
        self,
        X: pd.DataFrame,
        output_path: Optional[Path] = None,
        max_display: int = 20
    ) -> Optional[str]:
        """
        Generate SHAP summary plot for feature importance.

        Args:
            X: DataFrame with features
            output_path: Path to save plot
            max_display: Maximum features to display

        Returns:
            Path to saved plot or None
        """
        if self.explainer is None:
            return None

        try:
            shap_values = self.explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names or X.columns.tolist(),
                max_display=max_display,
                show=False
            )

            if output_path:
                plt.savefig(output_path, bbox_inches="tight", dpi=150)
                plt.close()
                return str(output_path)
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error generating summary plot: {e}")
            return None

    def generate_force_plot_html(
        self,
        X: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Generate interactive SHAP force plot as HTML.

        Args:
            X: Single row DataFrame
            output_path: Path to save HTML

        Returns:
            HTML string or path to saved file
        """
        if self.explainer is None:
            return None

        try:
            shap_values = self.explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Generate force plot
            force_plot = shap.force_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
                shap_values[0],
                X.iloc[0],
                feature_names=self.feature_names or X.columns.tolist(),
                matplotlib=False
            )

            if output_path:
                shap.save_html(str(output_path), force_plot)
                return str(output_path)
            else:
                return shap.getjs() + force_plot.html()

        except Exception as e:
            logger.error(f"Error generating force plot: {e}")
            return None

    def format_explanation_text(self, explanation: Dict[str, Any]) -> str:
        """
        Format explanation as human-readable text.

        Args:
            explanation: Explanation dictionary from explain_single

        Returns:
            Formatted text explanation
        """
        if "error" in explanation:
            return f"Could not generate explanation: {explanation['error']}"

        lines = ["## Flight Delay Prediction Explanation\n"]
        lines.append(f"Base delay probability: {explanation['base_value']:.1%}\n")
        lines.append("### Top Contributing Factors:\n")

        for i, feature in enumerate(explanation["features"], 1):
            direction = "↑" if feature["shap_value"] > 0 else "↓"
            impact_pct = abs(feature["shap_value"]) * 100
            lines.append(
                f"{i}. **{feature['name']}** = {feature['value']:.2f}\n"
                f"   {direction} {feature['impact']} ({impact_pct:.1f}% impact)\n"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    print("ModelExplainer module loaded successfully")
    print("To use, initialize with a trained model and call explain_single() or explain_batch()")
