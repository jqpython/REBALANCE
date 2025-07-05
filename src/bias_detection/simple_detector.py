"""
Simplified interface for bias detection.
This provides an easy-to-use API for practitioners who want quick results.
"""

from .detector import BiasDetector, BiasMetrics
import pandas as pd
from typing import Union, Optional


class SimpleBiasDetector:
    """
    A simplified interface for bias detection.

    Think of this as the "easy mode" for bias detection - it makes
    smart defaults and provides clear, actionable results.
    """

    def __init__(self):
        self.detector = BiasDetector(verbose=False)
        self.last_result = None

    def check_gender_bias(self,
                         data: pd.DataFrame,
                         target_column: str,
                         gender_column: str = None,
                         positive_outcome: any = None) -> dict:
        """
        Check for gender bias in a dataset with minimal configuration.

        This method tries to be smart about finding gender columns and
        understanding what constitutes a positive outcome.
        """
        # Auto-detect gender column if not specified
        if gender_column is None:
            gender_column = self._find_gender_column(data)
            if gender_column is None:
                raise ValueError("Could not auto-detect gender column. "
                               "Please specify using gender_column parameter.")

        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Auto-detect positive outcome if not specified
        if positive_outcome is None:
            positive_outcome = self._infer_positive_outcome(y)

        # Run bias detection
        metrics = self.detector.detect_bias(X, y, gender_column, positive_outcome)

        # Create simple result dictionary
        result = {
            'is_biased': metrics.is_biased(),
            'bias_level': metrics.get_bias_severity(),
            'disparate_impact': round(metrics.disparate_impact, 3),
            'recommendation': self._get_recommendation(metrics),
            'details': metrics
        }

        self.last_result = result
        return result

    def _find_gender_column(self, data: pd.DataFrame) -> Optional[str]:
        """
        Try to automatically find the gender column.
        Like a detective looking for clues.
        """
        # Common names for gender columns
        common_names = ['gender', 'sex', 'Gender', 'Sex', 'GENDER', 'SEX']

        for col in data.columns:
            if col in common_names:
                return col

            # Check if column contains gender-like values
            if data[col].dtype == 'object':
                unique_values = data[col].unique()
                if len(unique_values) == 2:
                    values_lower = [str(v).lower() for v in unique_values]
                    if set(values_lower) & {'male', 'female', 'm', 'f', 'man', 'woman'}:
                        return col

        return None

    def _infer_positive_outcome(self, y: pd.Series) -> any:
        """
        Try to infer what constitutes a positive outcome.
        Usually it's the less common class or the "higher" value.
        """
        if y.dtype in ['int64', 'float64']:
            # For numeric, assume higher value is positive
            return y.max()
        else:
            # For categorical, check for obvious positive indicators
            unique_values = y.unique()

            # Check for obvious positive values
            for val in unique_values:
                val_str = str(val).lower()
                if any(pos in val_str for pos in ['yes', 'true', '1', 'positive',
                                                   'high', '>50k', 'hired', 'accepted']):
                    return val

            # Otherwise, assume minority class is positive
            return y.value_counts().idxmin()

    def _get_recommendation(self, metrics: BiasMetrics) -> str:
        """
        Provide actionable recommendations based on bias level.
        Like a doctor prescribing treatment based on diagnosis.
        """
        if not metrics.is_biased():
            return "No significant bias detected. Continue monitoring."

        di = metrics.disparate_impact

        if di < 0.4:
            return ("Severe bias detected. Immediate action required. "
                   "Consider fairness-aware resampling and review data collection process.")
        elif di < 0.6:
            return ("Significant bias detected. "
                   "Recommend applying fairness-aware SMOTE or similar techniques.")
        else:
            return ("Moderate bias detected. "
                   "Consider bias mitigation techniques to ensure fair outcomes.")

    def print_summary(self):
        """Print a simple, easy-to-understand summary."""
        if self.last_result is None:
            print("No analysis performed yet. Run check_gender_bias() first.")
            return

        result = self.last_result

        print("\n🔍 GENDER BIAS ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Bias Status: {'⚠️  BIASED' if result['is_biased'] else '✅ FAIR'}")
        print(f"Severity: {result['bias_level']}")
        print(f"Disparate Impact: {result['disparate_impact']}")
        print(f"\n📋 Recommendation:")
        print(f"   {result['recommendation']}")
        print("=" * 40)
