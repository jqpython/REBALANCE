# File: src/fairness_smote/fair_smote.py

"""
Fairness-Aware SMOTE Implementation

This module implements an enhanced version of SMOTE that considers
protected attributes when generating synthetic samples, ensuring that
bias is reduced rather than amplified.

The key insight: Instead of finding nearest neighbors globally,
we find them within protected attribute groups.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type
from collections import Counter
import warnings


class FairSMOTE(BaseOverSampler):
    """
    Fairness-Aware Synthetic Minority Over-sampling Technique.

    This implementation modifies standard SMOTE to consider protected
    attributes when generating synthetic samples. The key innovation is
    that nearest neighbors are found within the same protected group,
    preserving group characteristics while balancing the dataset.

    Think of it as SMOTE with social awareness - it ensures that
    synthetic women look like real women, not like interpolations
    between women and men.

    Parameters
    ----------
    protected_attribute : str
        Column name of the protected attribute (e.g., 'gender', 'sex')

    k_neighbors : int, default=5
        Number of nearest neighbors to use for generating synthetic samples

    sampling_strategy : str or dict, default='auto'
        Sampling strategy to use:
        - 'auto': Balance all classes to have equal samples
        - 'minority': Resample only the minority class
        - dict: Specify the number of samples per class

    fairness_strategy : str, default='equal_opportunity'
        How to achieve fairness:
        - 'equal_opportunity': Equalize positive rates across protected groups
        - 'demographic_parity': Equalize overall representation

    random_state : int, default=None
        Random seed for reproducibility
    """

    def __init__(self,
                 protected_attribute: str,
                 k_neighbors: int = 5,
                 sampling_strategy: Union[str, dict] = 'auto',
                 fairness_strategy: str = 'equal_opportunity',
                 random_state: Optional[int] = None):

        super().__init__(sampling_strategy=sampling_strategy)
        self.protected_attribute = protected_attribute
        self.k_neighbors = k_neighbors
        self.fairness_strategy = fairness_strategy
        self.random_state = random_state

        # These will be set during fitting
        self.protected_attribute_idx_ = None
        self.encoders_ = {}
        self.feature_indices_ = {}

    def _fit_resample(self, X, y):
        """
        Resample the dataset to reduce bias.

        This is where the magic happens. We'll generate synthetic samples
        strategically to improve fairness metrics while maintaining
        data quality.
        """
        # First, understand what we're working with
        self._validate_inputs(X, y)

        # Convert to numpy arrays for easier manipulation
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        # Encode categorical variables if necessary
        X_encoded, X_was_dataframe = self._encode_categorical(X)

        # Analyze current bias situation
        bias_analysis = self._analyze_bias(X, y)

        # Determine how many samples we need to generate for each group
        sampling_plan = self._create_sampling_plan(X, y, bias_analysis)

        # Generate synthetic samples according to the plan
        X_synthetic, y_synthetic = self._generate_synthetic_samples(
            X_encoded, y_np, sampling_plan
        )

        # Combine original and synthetic data
        if X_synthetic.shape[0] > 0:
            X_resampled = np.vstack([X_encoded, X_synthetic])
            y_resampled = np.hstack([y_np, y_synthetic])
        else:
            X_resampled = X_encoded
            y_resampled = y_np

        # Decode back to original format if necessary
        if X_was_dataframe:
            X_resampled = self._decode_categorical(X_resampled, X)

        return X_resampled, y_resampled

    def _validate_inputs(self, X, y):
        """
        Ensure inputs are valid for fairness-aware resampling.
        Like a pre-flight checklist before takeoff.
        """
        if isinstance(X, pd.DataFrame):
            if self.protected_attribute not in X.columns:
                raise ValueError(
                    f"Protected attribute '{self.protected_attribute}' "
                    f"not found in DataFrame columns: {list(X.columns)}"
                )
            self.protected_attribute_idx_ = X.columns.get_loc(self.protected_attribute)
        else:
            warnings.warn(
                "X is not a pandas DataFrame. Assuming protected attribute "
                f"is in column {self.protected_attribute_idx_ or 0}"
            )
            if self.protected_attribute_idx_ is None:
                self.protected_attribute_idx_ = 0

    def _encode_categorical(self, X):
        """
        Encode categorical variables for distance calculations.
        SMOTE needs numerical data to compute distances.
        """
        if not isinstance(X, pd.DataFrame):
            return X, False

        X_encoded = X.copy()

        # Find categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns

        # Encode each categorical column
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            self.encoders_[col] = le

        return X_encoded.values, True

    def _decode_categorical(self, X_encoded, X_original):
        """
        Decode categorical variables back to original format.
        """
        X_decoded = pd.DataFrame(X_encoded, columns=X_original.columns)

        # Decode each categorical column
        for col, le in self.encoders_.items():
            X_decoded[col] = le.inverse_transform(X_decoded[col].astype(int))

        return X_decoded

    def _analyze_bias(self, X, y):
        """
        Analyze the current bias situation to inform our strategy.
        This is like a doctor's diagnosis before prescribing treatment.
        """
        # Extract protected attribute values
        if isinstance(X, pd.DataFrame):
            protected_values = X[self.protected_attribute]
        else:
            protected_values = X[:, self.protected_attribute_idx_]

        # Calculate statistics for each group
        analysis = {}

        for group in np.unique(protected_values):
            group_mask = protected_values == group
            group_y = y[group_mask]

            analysis[group] = {
                'total_count': len(group_y),
                'positive_count': (group_y == 1).sum() if len(group_y) > 0 else 0,
                'positive_rate': (group_y == 1).mean() if len(group_y) > 0 else 0,
                'mask': group_mask
            }

        # Identify privileged and unprivileged groups
        groups_by_rate = sorted(analysis.keys(),
                               key=lambda g: analysis[g]['positive_rate'],
                               reverse=True)

        analysis['privileged_group'] = groups_by_rate[0]
        analysis['unprivileged_group'] = groups_by_rate[-1]

        # Calculate disparate impact
        priv_rate = analysis[analysis['privileged_group']]['positive_rate']
        unpriv_rate = analysis[analysis['unprivileged_group']]['positive_rate']
        analysis['disparate_impact'] = unpriv_rate / priv_rate if priv_rate > 0 else 0

        return analysis

    def _create_sampling_plan(self, X, y, bias_analysis):
        """
        Determine how many synthetic samples to generate for each group.
        This is the strategic planning phase - we decide what kind of
        synthetic data will best improve fairness.
        """
        plan = {}

        if self.fairness_strategy == 'equal_opportunity':
            # Goal: Equalize positive rates across groups
            # Find the target positive rate (we'll use the privileged group's rate)
            target_positive_rate = bias_analysis[bias_analysis['privileged_group']]['positive_rate']

            for group, stats in bias_analysis.items():
                if group in ['privileged_group', 'unprivileged_group', 'disparate_impact']:
                    continue

                current_positive = stats['positive_count']
                current_total = stats['total_count']
                current_rate = stats['positive_rate']

                if current_rate < target_positive_rate:
                    # This group needs more positive samples
                    # Calculate how many positive samples we need in total
                    needed_positive = int(target_positive_rate * current_total)
                    synthetic_positive_needed = max(0, needed_positive - current_positive)

                    plan[group] = {
                        'positive_samples_needed': synthetic_positive_needed,
                        'negative_samples_needed': 0  # We only add positive samples
                    }
                else:
                    # This group doesn't need synthetic samples
                    plan[group] = {
                        'positive_samples_needed': 0,
                        'negative_samples_needed': 0
                    }

        return plan

    def _generate_synthetic_samples(self, X_encoded, y, sampling_plan):
        """
        Generate synthetic samples according to the plan.
        This is where we actually create new, fair data points.
        """
        synthetic_samples = []
        synthetic_labels = []

        # Extract protected attribute values
        protected_values = X_encoded[:, self.protected_attribute_idx_]

        for group, plan in sampling_plan.items():
            if plan['positive_samples_needed'] > 0:
                # Get positive samples from this protected group
                group_mask = (protected_values == group) & (y == 1)
                group_samples = X_encoded[group_mask]

                if len(group_samples) < 2:
                    warnings.warn(
                        f"Not enough positive samples in group {group} "
                        f"for SMOTE (found {len(group_samples)}). Skipping."
                    )
                    continue

                # Generate synthetic positive samples for this group
                synthetic = self._smote_for_group(
                    group_samples,
                    n_samples=plan['positive_samples_needed']
                )

                synthetic_samples.extend(synthetic)
                synthetic_labels.extend([1] * len(synthetic))

        # Convert to arrays
        if synthetic_samples:
            X_synthetic = np.array(synthetic_samples)
            y_synthetic = np.array(synthetic_labels)
        else:
            X_synthetic = np.empty((0, X_encoded.shape[1]))
            y_synthetic = np.empty(0)

        return X_synthetic, y_synthetic

    def _smote_for_group(self, group_samples, n_samples):
        """
        Apply SMOTE within a specific group.
        This ensures synthetic samples maintain group characteristics.
        """
        # Fit nearest neighbors within this group
        n_neighbors = min(self.k_neighbors, len(group_samples) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 for self
        nn.fit(group_samples)

        synthetic_samples = []

        for _ in range(n_samples):
            # Randomly select a sample from this group
            idx = np.random.randint(0, len(group_samples))
            sample = group_samples[idx]

            # Find its nearest neighbors
            distances, indices = nn.kneighbors([sample])
            neighbor_indices = indices[0][1:]  # Exclude self

            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = group_samples[neighbor_idx]

            # Generate synthetic sample using SMOTE formula
            # synthetic = sample + λ * (neighbor - sample)
            # where λ is random between 0 and 1
            lambda_param = np.random.random()
            synthetic = sample + lambda_param * (neighbor - sample)

            synthetic_samples.append(synthetic)

        return synthetic_samples
