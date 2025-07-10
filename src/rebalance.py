"""
REBALANCE: Automated Gender Bias Resolution Toolkit
This is the main integration module that brings together bias detection
and fairness-aware mitigation into a seamless pipeline. It represents
the culmination of your work - a tool that makes fair AI accessible
to practitioners without requiring deep expertise in fairness theory.
The philosophy behind this integration is simple: detect, decide, and fix.
But the implementation is sophisticated, making intelligent decisions
based on the specific characteristics of each dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
import json
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y

# Import our components
from .bias_detection.detector import BiasDetector, BiasMetrics
from .fairness_smote.fair_smote import FairSMOTE
from .utils.visualization import create_bias_comparison_plot, create_pipeline_report

@dataclass
class RebalanceResult:
    """
    Comprehensive results from the rebalancing pipeline.
    This is like a medical report after treatment - it tells you
    what was wrong, what was done, and how successful it was.
    Each field provides important information for understanding
    and validating the bias mitigation process.
    """
    # Original data characteristics
    original_bias_metrics: BiasMetrics
    original_size: int
    # Rebalanced data
    X_rebalanced: pd.DataFrame
    y_rebalanced: np.ndarray
    # Results and improvements
    final_bias_metrics: BiasMetrics
    improvement_summary: Dict[str, float]
    # Process metadata
    method_used: str
    parameters_used: Dict[str, Any]
    processing_time: float
    recommendations: str
    # Warnings or issues encountered
    warnings: list

    def get_summary(self) -> str:
        """Generate a human-readable summary of the results."""
        return f"""
REBALANCE RESULTS SUMMARY
========================
Original Dataset:
  - Size: {self.original_size:,} samples
  - Disparate Impact: {self.original_bias_metrics.disparate_impact:.3f}
  - Bias Level: {self.original_bias_metrics.get_bias_severity()}
Rebalanced Dataset:
  - Size: {len(self.X_rebalanced):,} samples
  - Disparate Impact: {self.final_bias_metrics.disparate_impact:.3f}
  - Bias Level: {self.final_bias_metrics.get_bias_severity()}
Improvement:
  - Disparate Impact: {self.improvement_summary['disparate_impact_change']:.1f}%
  - Synthetic Samples Added: {len(self.X_rebalanced) - self.original_size:,}
Recommendation:
{self.recommendations}
Processing Time: {self.processing_time:.2f} seconds
"""

class FairRebalancer:
    """
    The main interface for the REBALANCE toolkit.
    This class orchestrates the entire bias mitigation process,
    making intelligent decisions about how to best reduce bias
    while maintaining data quality and model utility.
    Think of this as the conductor of an orchestra - it doesn't play
    any instrument itself, but it coordinates all the musicians
    (components) to create beautiful music (fair datasets).
    """
    def __init__(self,
                 protected_attribute: str = None,
                 target_fairness: float = 0.8,
                 k_neighbors: int = 5,
                 random_state: int = None,
                 verbose: bool = True):
        """
        Initialize the Fair Rebalancer.
        Parameters
        ----------
        protected_attribute : str, optional
            The column name containing the protected attribute (e.g., 'gender', 'sex').
            If None, will attempt to auto-detect.
        target_fairness : float, default=0.8
            The target disparate impact ratio. 0.8 is the legal threshold
            known as the "four-fifths rule" in employment law.
        k_neighbors : int, default=5
            Number of nearest neighbors to use in fairness-aware SMOTE.
            Think of this as how many similar examples to consider
            when creating synthetic samples.
        random_state : int, optional
            Random seed for reproducibility. Like setting a starting point
            so you can recreate the exact same results later.
        verbose : bool, default=True
            Whether to print progress information. Helpful for understanding
            what the pipeline is doing.
        """
        self.protected_attribute = protected_attribute
        self.target_fairness = target_fairness
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.verbose = verbose
        # Initialize components
        self.bias_detector = BiasDetector(verbose=False)  # We'll control verbosity
        self.fair_smote = None  # Will be initialized when needed
        # Track state
        self.last_result = None
        self._warnings = []
        self.encoders_ = {}  # Store encoders for later decoding
        self.y_encoder_ = None  # Store encoder for target variable

    def fit_transform(self,
                     X: pd.DataFrame,
                     y: Union[pd.Series, np.ndarray],
                     target_column_name: str = None) -> RebalanceResult:
        """
        The main method that performs the complete bias mitigation pipeline.
        This is where the magic happens. The method will:
        1. Detect current bias levels
        2. Decide if mitigation is needed
        3. Apply appropriate mitigation strategy
        4. Validate the results
        5. Provide recommendations

        Parameters
        ----------
        X : pd.DataFrame
            Features including the protected attribute
        y : pd.Series or np.ndarray
            Target variable (what we're predicting)
        target_column_name : str, optional
            Name of the target column (for reporting purposes)

        Returns
        -------
        RebalanceResult
            Comprehensive results including rebalanced data and metrics
        """
        start_time = time.time()
        self._warnings = []
        if self.verbose:
            print("\n" + "="*60)
            print("REBALANCE PIPELINE STARTING")
            print("="*60)
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Validate and prepare inputs
        X, y, protected_attr = self._prepare_inputs(X, y)

        # Step 2: Initial bias detection
        if self.verbose:
            print("\n📊 STEP 1: Detecting current bias levels...")
        original_metrics = self.bias_detector.detect_bias(
            X=X,
            y=y,
            protected_attribute=protected_attr,
            positive_label=self._infer_positive_label(y)
        )
        if self.verbose:
            print(f"   Disparate Impact: {original_metrics.disparate_impact:.3f}")
            print(f"   Bias Level: {original_metrics.get_bias_severity()}")

        # Step 3: Decide on mitigation strategy
        strategy = self._decide_mitigation_strategy(original_metrics)
        if strategy == "none":
            # No mitigation needed
            if self.verbose:
                print("\n✅ No significant bias detected. Dataset is already fair!")
            result = self._create_result(
                X, y, X, y, original_metrics, original_metrics,
                method_used="none", parameters_used={},
                processing_time=time.time() - start_time
            )
            self.last_result = result
            return result

        # Step 4: Apply mitigation
        if self.verbose:
            print(f"\n🔧 STEP 2: Applying {strategy} mitigation strategy...")
        X_rebalanced, y_rebalanced, method_params = self._apply_mitigation(
            X, y, protected_attr, original_metrics, strategy
        )

        # Step 5: Validate results
        if self.verbose:
            print("\n📈 STEP 3: Validating results...")
        final_metrics = self.bias_detector.detect_bias(
            X_rebalanced, y_rebalanced,
            protected_attribute=protected_attr,
            positive_label=self._infer_positive_label(y)
        )
        if self.verbose:
            print(f"   New Disparate Impact: {final_metrics.disparate_impact:.3f}")
            print(f"   New Bias Level: {final_metrics.get_bias_severity()}")

        # Step 6: Create comprehensive result
        result = self._create_result(
            X, y, X_rebalanced, y_rebalanced,
            original_metrics, final_metrics,
            method_used=strategy,
            parameters_used=method_params,
            processing_time=time.time() - start_time
        )
        self.last_result = result
        if self.verbose:
            print("\n" + "="*60)
            print("REBALANCE PIPELINE COMPLETE")
            print("="*60)
            print(result.get_summary())
        return result

    def _prepare_inputs(self, X, y):
        """
        Prepare and validate inputs for processing.
        Like a chef preparing ingredients before cooking.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Ensure y is the right length
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")

        # Auto-detect protected attribute if needed
        if self.protected_attribute is None:
            detected = self._auto_detect_protected_attribute(X)
            if detected is None:
                raise ValueError(
                    "Could not auto-detect protected attribute. "
                    "Please specify it explicitly using protected_attribute parameter."
                )
            self.protected_attribute = detected
            if self.verbose:
                print(f"\n🔍 Auto-detected protected attribute: '{self.protected_attribute}'")

        # Validate protected attribute exists
        if self.protected_attribute not in X.columns:
            raise ValueError(
                f"Protected attribute '{self.protected_attribute}' not found in DataFrame. "
                f"Available columns: {list(X.columns)}"
            )

        return X, y, self.protected_attribute

    def _auto_detect_protected_attribute(self, X):
        """
        Attempt to automatically detect gender/sex column.
        Like a detective looking for clues.
        """
        common_names = ['gender', 'sex', 'Gender', 'Sex', 'GENDER', 'SEX']
        # First, check exact matches
        for col in X.columns:
            if col in common_names:
                return col
        # Then, check column contents
        for col in X.columns:
            if X[col].dtype == 'object' and len(X[col].unique()) == 2:
                values_lower = [str(v).lower() for v in X[col].unique()]
                if set(values_lower) & {'male', 'female', 'm', 'f', 'man', 'woman'}:
                    return col
        return None

    def _infer_positive_label(self, y):
        """
        Infer what constitutes a positive outcome.
        Usually it's 1, '>50K', 'Yes', etc.
        """
        if hasattr(y, 'dtype') and y.dtype in ['int64', 'float64']:
            return y.max()  # Assume higher value is positive
        else:
            # For categorical, look for obvious positive indicators
            unique_values = np.unique(y)
            for val in unique_values:
                if pd.api.types.is_string_dtype(type(val)):
                    val_str = str(val).lower()
                    if any(pos in val_str for pos in ['yes', '1', 'true', '>50k', 'high']):
                        return val
            # If we couldn't find an obvious positive label, use the minority class
            if hasattr(y, 'value_counts'):
                counts = y.value_counts()
                return counts.idxmin()  # Return the least frequent class as positive
            else:
                # For numpy array or other types
                unique, counts = np.unique(y, return_counts=True)
                return unique[np.argmin(counts)]

    def _decide_mitigation_strategy(self, metrics: BiasMetrics) -> str:
        """
        Decide which mitigation strategy to use based on bias level.
        This is like a doctor deciding on treatment based on diagnosis.
        """
        di = metrics.disparate_impact
        if di >= self.target_fairness:
            return "none"  # Already fair
        elif di >= 0.6:
            return "moderate"  # Moderate intervention needed
        elif di >= 0.4:
            return "aggressive"  # Significant intervention needed
        else:
            return "maximum"  # Severe bias requires maximum intervention

    def _apply_mitigation(self, X, y, protected_attr, original_metrics, strategy):
        """
        Apply the chosen mitigation strategy.
        Different strategies represent different levels of intervention.
        """
        # Store original size for later reference
        original_size = len(X)

        # Convert y to binary if needed (store original for later)
        original_y = y.copy()
        y_binary = self._convert_y_to_binary(y)
        self.y_encoder_ = {'original': original_y, 'binary': y_binary}

        # Encode categorical variables for FairSMOTE
        if self.verbose:
            print("🔢 Encoding categorical variables for processing...")

        # Store the original column types for later decoding
        original_dtypes = X.dtypes.copy()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Encode categorical columns
        X_encoded = X.copy()
        self.encoders_ = {}

        for col in categorical_columns:
            # Skip the protected attribute if it's categorical (we'll handle it specially)
            if col == protected_attr:
                continue

            try:
                le = LabelEncoder()
                # Fit on the column data
                le.fit(X[col].astype(str))
                X_encoded[col] = le.transform(X[col].astype(str))
                self.encoders_[col] = le
            except Exception as e:
                self._warnings.append(f"Failed to encode column {col}: {str(e)}")
                # If encoding fails, leave as is (may cause problems later)
                pass

        # Handle the protected attribute separately
        if X_encoded[protected_attr].dtype == 'object':
            try:
                le = LabelEncoder()
                le.fit(X_encoded[protected_attr].astype(str))
                X_encoded[protected_attr] = le.transform(X_encoded[protected_attr].astype(str))
                self.encoders_[protected_attr] = le
            except Exception as e:
                self._warnings.append(f"Failed to encode protected attribute {protected_attr}: {str(e)}")
                raise ValueError(f"Could not encode protected attribute: {str(e)}")

        # Apply fairness-aware SMOTE
        try:
            if self.verbose:
                print("🛠 Applying fairness-aware SMOTE...")

            # Initialize FairSMOTE with appropriate parameters
            strategy_params = {
                "moderate": {
                    "k_neighbors": self.k_neighbors,
                    "sampling_strategy": "auto",
                    "fairness_strategy": "equal_opportunity"
                },
                "aggressive": {
                    "k_neighbors": max(3, self.k_neighbors - 2),  # Fewer neighbors = more aggressive
                    "sampling_strategy": "auto",
                    "fairness_strategy": "equal_opportunity"
                },
                "maximum": {
                    "k_neighbors": 3,  # Minimum neighbors for maximum effect
                    "sampling_strategy": "all",  # Oversample all classes
                    "fairness_strategy": "equal_opportunity"
                }
            }
            params = strategy_params.get(strategy, strategy_params["moderate"])

            # Initialize FairSMOTE
            self.fair_smote = FairSMOTE(
                protected_attribute=protected_attr,
                random_state=self.random_state,
                **params
            )

            # Apply FairSMOTE
            X_resampled_encoded, y_resampled_binary = self.fair_smote.fit_resample(
                X_encoded, y_binary
            )

            # Decode back to original format
            X_resampled = self._decode_categorical(X_resampled_encoded, X, original_dtypes)

            # Convert y back to original format
            y_resampled = self._convert_binary_to_original_y(y_resampled_binary, original_y)

            return X_resampled, y_resampled, params

        except Exception as e:
            self._warnings.append(f"FairSMOTE failed: {str(e)}. Falling back to standard approach.")
            raise

    def _decode_categorical(self, X_encoded, X_original, original_dtypes):
        """
        Decode categorical variables back to original format.
        Handles both pandas DataFrames and numpy arrays.
        """
        if isinstance(X_encoded, np.ndarray):
            # Convert to DataFrame if needed
            X_decoded = pd.DataFrame(X_encoded, columns=X_original.columns)
        else:
            X_decoded = X_encoded.copy()

        # Decode each categorical column
        for col, le in self.encoders_.items():
            if col in X_decoded.columns:
                try:
                    # Convert to int first, then decode
                    if X_decoded[col].dtype != 'int':
                        X_decoded[col] = X_decoded[col].astype(int)
                    X_decoded[col] = le.inverse_transform(X_decoded[col])
                except Exception as e:
                    self._warnings.append(f"Failed to decode column {col}: {str(e)}")
                    # If decoding fails, leave as is
                    pass

        # Ensure the output has the same dtypes as the original data where possible
        for col in X_original.columns:
            if col in X_decoded.columns:
                # For categorical columns, we've already decoded them
                if col in self.encoders_:
                    continue
                # Try to match original dtype if possible
                try:
                    if original_dtypes[col].name in ['int64', 'float64']:
                        X_decoded[col] = pd.to_numeric(X_decoded[col], errors='ignore')
                except Exception:
                    pass

        # Convert back to DataFrame if needed
        if isinstance(X_encoded, np.ndarray):
            return pd.DataFrame(X_decoded, columns=X_original.columns)
        else:
            return X_decoded

    def _convert_y_to_binary(self, y):
        """
        Convert target variable y to binary format (0/1) for SMOTE.
        Stores information needed to convert back to original format later.
        """
        # If y is already binary (0/1), just return it
        if set(np.unique(y)) <= {0, 1}:
            return y

        # Get positive label based on our inference method
        positive_label = self._infer_positive_label(y)

        # Convert to binary (1 for positive label, 0 otherwise)
        if isinstance(y, pd.Series):
            y_binary = (y == positive_label).astype(int)
        else:
            y_binary = (y == positive_label).astype(int)

        # Store the mapping information
        self.y_encoder_ = {
            'positive_label': positive_label,
            'original_values': np.unique(y),
            'binary': y_binary
        }

        return y_binary

    def _convert_binary_to_original_y(self, y_binary, original_y):
        """
        Convert binary y back to its original format
        """
        # If we don't have encoding information, return as-is
        if 'positive_label' not in self.y_encoder_:
            return y_binary

        positive_label = self.y_encoder_['positive_label']

        # Try to get the negative label from original_y
        unique_values = np.unique(original_y)
        if len(unique_values) >= 2:
            negative_label = [v for v in unique_values if v != positive_label][0]
        else:
            # If we can't determine, default to using the first non-positive label
            negative_label = 'other'  # Default fallback

        # Convert back to original format
        if isinstance(original_y, pd.Series):
            return pd.Series(
                [positive_label if val == 1 else negative_label for val in y_binary],
                index=original_y.index[:len(y_binary)] if hasattr(original_y, 'index') else None
            )
        else:
            return np.array([
                positive_label if val == 1 else negative_label
                for val in y_binary
            ])


    def _create_result(self, X_orig, y_orig, X_rebal, y_rebal,
                      orig_metrics, final_metrics, method_used,
                      parameters_used, processing_time):
        """
        Create a comprehensive result object with all relevant information.
        This is like preparing a complete report of the treatment.
        """
        # Calculate improvements
        di_improvement = ((final_metrics.disparate_impact - orig_metrics.disparate_impact)
                         / orig_metrics.disparate_impact * 100)
        spd_improvement = abs(final_metrics.statistical_parity_difference) - abs(orig_metrics.statistical_parity_difference)
        improvement_summary = {
            'disparate_impact_change': di_improvement,
            'statistical_parity_improvement': spd_improvement,
            'bias_reduced': final_metrics.disparate_impact >= self.target_fairness
        }
        # Generate recommendations
        recommendations = self._generate_recommendations(
            orig_metrics, final_metrics, improvement_summary
        )
        return RebalanceResult(
            original_bias_metrics=orig_metrics,
            original_size=len(X_orig),
            X_rebalanced=X_rebal,
            y_rebalanced=y_rebal,
            final_bias_metrics=final_metrics,
            improvement_summary=improvement_summary,
            method_used=method_used,
            parameters_used=parameters_used,
            processing_time=processing_time,
            recommendations=recommendations,
            warnings=self._warnings.copy()
        )

    def _generate_recommendations(self, orig_metrics, final_metrics, improvements):
        """
        Generate actionable recommendations based on results.
        Like a doctor's advice after treatment.
        """
        if improvements['bias_reduced']:
            if final_metrics.disparate_impact >= 0.9:
                return (
                    "✅ Excellent results! The dataset now exhibits minimal bias. "
                    "You can proceed with model training. Continue monitoring for "
                    "bias in model predictions, as model choice can reintroduce bias."
                )
            else:
                return (
                    "✅ Good results! Bias has been reduced to acceptable levels. "
                    "Consider using ensemble methods or regularization during model "
                    "training to maintain fairness."
                )
        else:
            if improvements['disparate_impact_change'] > 50:
                return (
                    "⚠️  Significant improvement achieved, but bias remains above threshold. "
                    "Consider: 1) Collecting more diverse training data, "
                    "2) Using in-processing fairness constraints during model training, "
                    "3) Applying post-processing calibration to model outputs."
                )
            else:
                return (
                    "⚠️  Limited improvement achieved. The bias in this dataset may be "
                    "too severe for pre-processing alone. Strongly consider: "
                    "1) Investigating data collection practices, "
                    "2) Using fairness-aware learning algorithms, "
                    "3) Implementing decision-level interventions."
                )
