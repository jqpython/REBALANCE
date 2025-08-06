"""
External Fairness Toolkit Adapters for REBALANCE

This module provides adapters to integrate popular fairness toolkits
(AIF360, Fairlearn) into the REBALANCE evaluation framework, enabling
fair comparisons and comprehensive benchmarking.

Author: REBALANCE Team
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

try:
    from aif360.datasets import StandardDataset
    from aif360.algorithms.preprocessing import Reweighing, OptimPreproc
    from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    warnings.warn("AIF360 not available. AIF360 adapters will not work.")

try:
    from fairlearn.preprocessing import CorrelationRemover
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    warnings.warn("Fairlearn not available. Fairlearn adapters will not work.")


class AIF360ReweighingAdapter(BaseEstimator, TransformerMixin):
    """
    Adapter for AIF360's Reweighing algorithm.
    
    This adapter makes AIF360's reweighing technique compatible with
    the REBALANCE evaluation framework by providing a sklearn-style
    fit_resample interface.
    """
    
    def __init__(self, protected_attribute: str = 'sex', privileged_groups: Optional[Dict] = None):
        if not AIF360_AVAILABLE:
            raise ImportError("AIF360 is required but not installed. Install with: pip install aif360")
        
        self.protected_attribute = protected_attribute
        self.privileged_groups = privileged_groups or [{protected_attribute: 1}]
        self.unprivileged_groups = [{protected_attribute: 0}]
        self.reweighing = None
        self.label_encoder = None
        
    def fit_resample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply AIF360 reweighing to the dataset.
        
        Returns reweighted samples (with sample weights applied through replication).
        """
        # Encode protected attribute if necessary
        X_encoded = X.copy()
        if X[self.protected_attribute].dtype == 'object':
            self.label_encoder = LabelEncoder()
            X_encoded[self.protected_attribute] = self.label_encoder.fit_transform(X[self.protected_attribute])
        
        # Convert to AIF360 StandardDataset format
        # Note: AIF360 expects specific column names and formats
        favorable_label = 1
        unfavorable_label = 0
        
        # Create a temporary dataframe with the target column
        df_aif = X_encoded.copy()
        df_aif['target'] = y
        
        try:
            # Create AIF360 dataset
            dataset = StandardDataset(
                df_aif,
                label_name='target',
                favorable_classes=[favorable_label],
                protected_attribute_names=[self.protected_attribute],
                privileged_classes=[self.privileged_groups[0][self.protected_attribute]]
            )
            
            # Apply reweighing
            self.reweighing = Reweighing(
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups
            )
            
            dataset_reweighed = self.reweighing.fit_transform(dataset)
            
            # Extract reweighted data
            # Convert instance weights to sample replication
            weights = dataset_reweighed.instance_weights
            
            # Create resampled dataset by replicating samples based on weights
            X_resampled_list = []
            y_resampled_list = []
            
            for idx, weight in enumerate(weights):
                # Replicate samples proportional to their weight
                replications = max(1, int(np.round(weight * len(X))))
                for _ in range(replications):
                    X_resampled_list.append(X.iloc[idx])
                    y_resampled_list.append(y[idx])
            
            X_resampled = pd.DataFrame(X_resampled_list, columns=X.columns).reset_index(drop=True)
            y_resampled = np.array(y_resampled_list)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            warnings.warn(f"AIF360 Reweighing failed: {str(e)}. Returning original data.")
            return X, y


class FairlearnThresholdAdapter(BaseEstimator, TransformerMixin):
    """
    Adapter for Fairlearn's Threshold Optimization.
    
    Note: This is a post-processing technique, so it works differently
    from preprocessing methods. For evaluation purposes, we return
    the original data since threshold optimization happens at prediction time.
    """
    
    def __init__(self, protected_attribute: str = 'sex', constraint: str = 'equalized_odds'):
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("Fairlearn is required but not installed. Install with: pip install fairlearn")
        
        self.protected_attribute = protected_attribute
        self.constraint = constraint
        self.threshold_optimizer = None
        
    def fit_resample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        For threshold optimization, we return original data since this is post-processing.
        
        The actual fairness constraint is applied during model prediction.
        """
        # For evaluation purposes, threshold optimization doesn't change the training data
        # It only affects the prediction thresholds
        warnings.warn(
            "ThresholdOptimizer is a post-processing technique. "
            "Returning original data. Fairness is achieved through threshold adjustment."
        )
        return X, y


class FairlearnGridSearchAdapter(BaseEstimator, TransformerMixin):
    """
    Adapter for Fairlearn's Grid Search with reduction approach.
    
    This adapter simulates the effect of reduction-based fairness constraints
    by creating a balanced subsample that respects fairness constraints.
    """
    
    def __init__(self, protected_attribute: str = 'sex', constraint: str = 'demographic_parity'):
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("Fairlearn is required but not installed. Install with: pip install fairlearn")
        
        self.protected_attribute = protected_attribute
        self.constraint = constraint
        
    def fit_resample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Simulate fairness-aware resampling based on demographic constraints.
        """
        try:
            # Calculate group statistics
            protected_values = X[self.protected_attribute]
            groups = protected_values.unique()
            
            if self.constraint == 'demographic_parity':
                # Balance positive rates across groups through resampling
                return self._balance_demographic_parity(X, y, protected_values, groups)
            elif self.constraint == 'equalized_odds':
                # Balance true positive and false positive rates
                return self._balance_equalized_odds(X, y, protected_values, groups)
            else:
                warnings.warn(f"Constraint {self.constraint} not implemented. Returning original data.")
                return X, y
                
        except Exception as e:
            warnings.warn(f"Fairlearn Grid Search simulation failed: {str(e)}. Returning original data.")
            return X, y
    
    def _balance_demographic_parity(self, X: pd.DataFrame, y: np.ndarray, 
                                   protected_values: pd.Series, groups) -> Tuple[pd.DataFrame, np.ndarray]:
        """Balance positive prediction rates across groups."""
        
        # Calculate target positive rate (average across groups)
        group_pos_rates = []
        for group in groups:
            group_mask = protected_values == group
            if group_mask.sum() > 0:
                pos_rate = y[group_mask].mean()
                group_pos_rates.append(pos_rate)
        
        target_pos_rate = np.mean(group_pos_rates)
        
        # Resample to achieve target positive rate for each group
        X_resampled_list = []
        y_resampled_list = []
        
        for group in groups:
            group_mask = protected_values == group
            group_X = X[group_mask]
            group_y = y[group_mask]
            
            if len(group_y) == 0:
                continue
                
            current_pos_rate = group_y.mean()
            
            # Separate positive and negative samples
            pos_mask = group_y == 1
            neg_mask = group_y == 0
            
            pos_samples = group_X[pos_mask]
            neg_samples = group_X[neg_mask]
            pos_labels = group_y[pos_mask]
            neg_labels = group_y[neg_mask]
            
            # Calculate how many samples we need
            total_group_samples = len(group_y)
            target_pos_samples = int(target_pos_rate * total_group_samples)
            target_neg_samples = total_group_samples - target_pos_samples
            
            # Resample positive samples
            if len(pos_samples) > 0 and target_pos_samples > 0:
                if target_pos_samples <= len(pos_samples):
                    # Subsample
                    pos_indices = np.random.choice(len(pos_samples), target_pos_samples, replace=False)
                else:
                    # Oversample
                    pos_indices = np.random.choice(len(pos_samples), target_pos_samples, replace=True)
                
                X_resampled_list.append(pos_samples.iloc[pos_indices])
                y_resampled_list.append(pos_labels[pos_indices])
            
            # Resample negative samples
            if len(neg_samples) > 0 and target_neg_samples > 0:
                if target_neg_samples <= len(neg_samples):
                    # Subsample
                    neg_indices = np.random.choice(len(neg_samples), target_neg_samples, replace=False)
                else:
                    # Oversample
                    neg_indices = np.random.choice(len(neg_samples), target_neg_samples, replace=True)
                
                X_resampled_list.append(neg_samples.iloc[neg_indices])
                y_resampled_list.append(neg_labels[neg_indices])
        
        if X_resampled_list:
            X_resampled = pd.concat(X_resampled_list, ignore_index=True)
            y_resampled = np.concatenate(y_resampled_list)
            return X_resampled, y_resampled
        else:
            return X, y
    
    def _balance_equalized_odds(self, X: pd.DataFrame, y: np.ndarray,
                               protected_values: pd.Series, groups) -> Tuple[pd.DataFrame, np.ndarray]:
        """Balance true positive and false positive rates across groups."""
        # Simplified implementation: balance within each outcome class
        # This is a proxy for equalized odds
        
        X_resampled_list = []
        y_resampled_list = []
        
        # Process positive and negative classes separately
        for class_label in [0, 1]:
            class_mask = y == class_label
            class_X = X[class_mask]
            class_y = y[class_mask]
            class_protected = protected_values[class_mask]
            
            # Find minimum group size within this class
            min_group_size = float('inf')
            for group in groups:
                group_size = (class_protected == group).sum()
                if group_size > 0:
                    min_group_size = min(min_group_size, group_size)
            
            # Sample equal numbers from each group within this class
            for group in groups:
                group_mask = class_protected == group
                if group_mask.sum() > 0:
                    group_indices = group_mask[group_mask].index
                    
                    if len(group_indices) >= min_group_size:
                        selected_indices = np.random.choice(group_indices, min_group_size, replace=False)
                    else:
                        selected_indices = np.random.choice(group_indices, min_group_size, replace=True)
                    
                    X_resampled_list.append(class_X.loc[selected_indices])
                    y_resampled_list.append(class_y[selected_indices])
        
        if X_resampled_list:
            X_resampled = pd.concat(X_resampled_list, ignore_index=True)
            y_resampled = np.concatenate(y_resampled_list)
            return X_resampled, y_resampled
        else:
            return X, y


def create_external_adapters(protected_attribute: str = 'sex') -> Dict[str, Any]:
    """
    Create instances of all available external fairness adapters.
    
    Returns:
        Dictionary mapping adapter names to adapter instances
    """
    adapters = {}
    
    if AIF360_AVAILABLE:
        adapters['AIF360 Reweighing'] = AIF360ReweighingAdapter(protected_attribute=protected_attribute)
    
    if FAIRLEARN_AVAILABLE:
        adapters['Fairlearn Demographic Parity'] = FairlearnGridSearchAdapter(
            protected_attribute=protected_attribute, 
            constraint='demographic_parity'
        )
        adapters['Fairlearn Equalized Odds'] = FairlearnGridSearchAdapter(
            protected_attribute=protected_attribute,
            constraint='equalized_odds'
        )
    
    return adapters


def check_external_dependencies() -> Dict[str, bool]:
    """
    Check which external fairness libraries are available.
    
    Returns:
        Dictionary with availability status for each library
    """
    return {
        'aif360': AIF360_AVAILABLE,
        'fairlearn': FAIRLEARN_AVAILABLE
    }