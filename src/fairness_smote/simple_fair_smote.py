# File: src/fairness_smote/simple_fair_smote.py

"""
Simplified interface for Fairness-Aware SMOTE.
Makes it easy to use without deep understanding of all parameters.
"""

from .fair_smote import FairSMOTE
import pandas as pd
import numpy as np
from typing import Union, Tuple


class SimpleFairSMOTE:
    """
    A user-friendly interface for Fairness-Aware SMOTE.

    This class automatically detects gender columns and applies
    fairness-aware resampling with sensible defaults.
    """

    def __init__(self, k_neighbors: int = 5, random_state: int = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.fair_smote = None
        self.gender_column = None

    def fit_resample(self,
                     X: pd.DataFrame,
                     y: Union[pd.Series, np.ndarray],
                     gender_column: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply fairness-aware resampling to reduce gender bias.

        Parameters
        ----------
        X : pd.DataFrame
            Features including gender information
        y : pd.Series or np.ndarray
            Target variable (0/1 or equivalent)
        gender_column : str, optional
            Name of gender column. If None, will try to auto-detect.

        Returns
        -------
        X_resampled : pd.DataFrame
            Resampled features with synthetic samples added
        y_resampled : np.ndarray
            Resampled target variable
        """
        # Auto-detect gender column if not specified
        if gender_column is None:
            self.gender_column = self._find_gender_column(X)
            if self.gender_column is None:
                raise ValueError(
                    "Could not auto-detect gender column. "
                    "Please specify using gender_column parameter."
                )
        else:
            self.gender_column = gender_column

        print(f"Using gender column: '{self.gender_column}'")

        # Initialize FairSMOTE
        self.fair_smote = FairSMOTE(
            protected_attribute=self.gender_column,
            k_neighbors=self.k_neighbors,
            sampling_strategy='auto',
            fairness_strategy='equal_opportunity',
            random_state=self.random_state
        )

        # Apply fairness-aware resampling
        X_resampled, y_resampled = self.fair_smote.fit_resample(X, y)

        # Print summary of what happened
        self._print_resampling_summary(X, y, X_resampled, y_resampled)

        return X_resampled, y_resampled

    def _find_gender_column(self, X: pd.DataFrame) -> str:
        """Auto-detect gender column in the dataframe."""
        common_names = ['gender', 'sex', 'Gender', 'Sex', 'GENDER', 'SEX']

        for col in X.columns:
            if col in common_names:
                return col

            # Check if column contains gender values
            if X[col].dtype == 'object' and len(X[col].unique()) == 2:
                values_lower = [str(v).lower() for v in X[col].unique()]
                if set(values_lower) & {'male', 'female', 'm', 'f'}:
                    return col

        return None

    def _print_resampling_summary(self, X, y, X_resampled, y_resampled):
        """Print a summary of what the resampling accomplished."""
        print("\n" + "="*50)
        print("FAIRNESS-AWARE RESAMPLING SUMMARY")
        print("="*50)

        # Original statistics
        orig_positive = (y == 1).sum()
        orig_total = len(y)
        print(f"\nOriginal dataset:")
        print(f"  Total samples: {orig_total:,}")
        print(f"  Positive samples: {orig_positive:,} ({orig_positive/orig_total*100:.1f}%)")

        # Gender distribution in positive class (original)
        orig_positive_mask = y == 1
        if isinstance(y, pd.Series):
            orig_positive_mask = orig_positive_mask.values

        orig_gender_positive = X.loc[orig_positive_mask, self.gender_column].value_counts()
        print(f"\n  Positive class gender distribution:")
        for gender, count in orig_gender_positive.items():
            print(f"    {gender}: {count} ({count/orig_positive*100:.1f}%)")

        # Resampled statistics
        resampled_positive = (y_resampled == 1).sum()
        resampled_total = len(y_resampled)
        print(f"\nResampled dataset:")
        print(f"  Total samples: {resampled_total:,}")
        print(f"  Positive samples: {resampled_positive:,} ({resampled_positive/resampled_total*100:.1f}%)")
        print(f"  Synthetic samples created: {resampled_total - orig_total:,}")

        # Gender distribution in positive class (resampled)
        resampled_positive_mask = y_resampled == 1
        resampled_gender_positive = X_resampled.loc[resampled_positive_mask, self.gender_column].value_counts()
        print(f"\n  Positive class gender distribution:")
        for gender, count in resampled_gender_positive.items():
            print(f"    {gender}: {count} ({count/resampled_positive*100:.1f}%)")

        print("\n" + "="*50)
