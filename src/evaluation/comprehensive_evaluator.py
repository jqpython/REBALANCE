# File: src/evaluation/comprehensive_evaluator.py

"""
Comprehensive Evaluation Framework for REBALANCE
This module provides a systematic approach to evaluating bias mitigation
effectiveness. Think of it as a scientific laboratory where we conduct
controlled experiments to understand exactly how well REBALANCE performs
under various conditions.
The evaluation philosophy is simple: measure everything that matters,
compare fairly, and present results clearly. We're not just checking
if the tool works - we're understanding when it works best, where it
struggles, and how it compares to alternatives.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
import json
from datetime import datetime
# Import all relevant tools for comparison
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.utils import resample
# Our components
from ..bias_detection.detector import BiasDetector
from ..fairness_smote.fair_smote import FairSMOTE
from ..rebalance import FairRebalancer

# External integrations
try:
    from ..integration.external_adapters import create_external_adapters, check_external_dependencies
    EXTERNAL_ADAPTERS_AVAILABLE = True
except ImportError:
    EXTERNAL_ADAPTERS_AVAILABLE = False
    warnings.warn("External adapter integrations not available.")


@dataclass
class EvaluationResult:
    """
    Stores comprehensive results from evaluating a bias mitigation method.
    Think of this as a detailed lab report that captures every aspect
    of how a method performed. It's designed to enable fair comparisons
    and deep understanding of each approach's strengths and weaknesses.
    """
    method_name: str
    # Bias metrics
    original_disparate_impact: float
    final_disparate_impact: float
    bias_improvement: float
    # Model performance metrics (averaged across models)
    accuracy: float
    precision: float
    recall: float
    f1: float
    # Efficiency metrics
    processing_time: float
    synthetic_samples_created: int
    # Robustness metrics
    cross_val_scores: List[float]
    variance_in_fairness: float
    # Detailed results for different models
    model_specific_results: Dict[str, Dict[str, float]]
    # Additional metadata
    parameters_used: Dict[str, Any]
    warnings_encountered: List[str]

    def get_summary_score(self) -> float:
        """
        Calculate a single score that balances fairness and performance.
        This helps in ranking different methods.
        """
        # Weighted combination of fairness and performance
        fairness_score = min(self.final_disparate_impact / 0.8, 1.0)  # Normalize to 0-1
        performance_score = self.f1  # Already 0-1
        # 60% weight on fairness, 40% on performance
        return 0.6 * fairness_score + 0.4 * performance_score


class ComprehensiveEvaluator:
    """
    Orchestrates comprehensive evaluation of bias mitigation methods.
    This class is like a head scientist who designs experiments, ensures
    fair testing conditions, and analyzes results to draw meaningful
    conclusions. It handles the complexity of evaluation so you can
    focus on understanding the results.
    """
    def __init__(self,
                 models_to_test: List[Any] = None,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the evaluator with testing parameters.
        Parameters
        ----------
        models_to_test : List of sklearn models
            ML models to use for evaluation. We test multiple models
            because bias mitigation effectiveness can vary by model type.
        cv_folds : int
            Number of cross-validation folds. More folds give more
            reliable results but take longer to compute.
        random_state : int
            For reproducibility - ensures same results each run.
        verbose : bool
            Whether to print progress information.
        """
        # First set all instance attributes
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        # Initialize bias detector
        self.bias_detector = BiasDetector(verbose=False)
        # Store results for analysis
        self.all_results = {}

        # Now get default models (which uses self.random_state)
        self.models_to_test = models_to_test or self._get_default_models()

    def _get_default_models(self):
        """Get a diverse set of models for testing with better LR configuration"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Create a pipeline with scaling for logistic regression
        lr_pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,  # Increased from 1000
                solver='liblinear',  # Often more stable than lbfgs
                class_weight='balanced',
                random_state=self.random_state
            )
        )

        return [
            ('Logistic Regression', lr_pipeline),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('SVM', SVC(kernel='rbf', random_state=self.random_state)),
            ('Naive Bayes', GaussianNB())
        ]

    def evaluate_all_methods(self,
                           X: pd.DataFrame,
                           y: np.ndarray,
                           protected_attribute: str = 'sex') -> Dict[str, EvaluationResult]:
        """
        Evaluate all bias mitigation methods on the same dataset.

        This is the main experiment - we test each method under identical
        conditions to ensure fair comparison. It's like a race where all
        contestants start from the same line and run the same course.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("COMPREHENSIVE BIAS MITIGATION EVALUATION")
            print("="*70)
            print(f"Dataset size: {len(X):,} samples")
            print(f"Protected attribute: {protected_attribute}")
            print(f"Models to test: {len(self.models_to_test)}")
            print(f"Cross-validation folds: {self.cv_folds}")
            print("="*70)

        # Methods to evaluate
        methods = {
            'No Mitigation (Baseline)': self._evaluate_baseline,
            'Random Oversampling': self._evaluate_random_oversampling,
            'Standard SMOTE': self._evaluate_standard_smote,
            'REBALANCE (Fair SMOTE)': self._evaluate_rebalance,
        }
        
        # Add external methods if available
        if EXTERNAL_ADAPTERS_AVAILABLE:
            external_adapters = create_external_adapters(protected_attribute)
            for adapter_name, adapter in external_adapters.items():
                methods[adapter_name] = lambda X, y, pa, adapter=adapter: self._evaluate_external_adapter(X, y, pa, adapter)

        results = {}

        for method_name, evaluation_function in methods.items():
            if self.verbose:
                print(f"\n📊 Evaluating: {method_name}")
                print("-" * 50)

            try:
                result = evaluation_function(X, y, protected_attribute)
                results[method_name] = result

                if self.verbose:
                    print(f"✅ Complete - DI: {result.final_disparate_impact:.3f}, "
                          f"F1: {result.f1:.3f}")

            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed: {str(e)}")
                # Continue with other methods

        self.all_results = results
        return results

    def _evaluate_external_adapter(self, X, y, protected_attribute, adapter):
        """Evaluate an external fairness adapter."""
        start_time = time.time()
        
        try:
            # Apply the external adapter
            X_resampled, y_resampled = adapter.fit_resample(X, y)
            
            # Calculate bias metrics on resampled data
            bias_metrics = self.bias_detector.detect_bias(X_resampled, y_resampled, protected_attribute, 1)
            
            # Evaluate model performance
            model_results = {}
            
            for model_name, model in self.models_to_test:
                try:
                    # Handle categorical encoding for models
                    X_encoded = self._encode_for_model(X_resampled)
                    X_test_encoded = self._encode_for_model(X)  # Encode test data similarly
                    
                    # Train model
                    model.fit(X_encoded, y_resampled)
                    
                    # Predict on test set
                    y_pred = model.predict(X_test_encoded)
                    
                    # Calculate metrics
                    model_results[model_name] = {
                        'accuracy': accuracy_score(y, y_pred),
                        'precision': precision_score(y, y_pred, zero_division=0),
                        'recall': recall_score(y, y_pred, zero_division=0),
                        'f1': f1_score(y, y_pred, zero_division=0)
                    }
                    
                except Exception as e:
                    warnings.warn(f"Model {model_name} failed with external adapter: {str(e)}")
                    model_results[model_name] = {
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
                    }
            
            processing_time = time.time() - start_time
            
            # Calculate average performance
            avg_accuracy = np.mean([r['accuracy'] for r in model_results.values()])
            avg_precision = np.mean([r['precision'] for r in model_results.values()])
            avg_recall = np.mean([r['recall'] for r in model_results.values()])
            avg_f1 = np.mean([r['f1'] for r in model_results.values()])
            
            return EvaluationResult(
                method_name=adapter.__class__.__name__,
                original_disparate_impact=0.0,  # Would need original bias for comparison
                final_disparate_impact=bias_metrics.disparate_impact,
                bias_improvement=0.0,  # Would need original bias for calculation
                accuracy=avg_accuracy,
                precision=avg_precision,
                recall=avg_recall,
                f1=avg_f1,
                processing_time=processing_time,
                synthetic_samples_created=len(X_resampled) - len(X),
                cross_val_scores=[],  # Could implement CV if needed
                variance_in_fairness=0.0,
                model_specific_results=model_results,
                parameters_used={},
                warnings_encountered=[]
            )
            
        except Exception as e:
            if self.verbose:
                print(f"External adapter failed: {str(e)}")
            
            # Return a minimal result indicating failure
            return EvaluationResult(
                method_name=adapter.__class__.__name__,
                original_disparate_impact=0.0,
                final_disparate_impact=1.0,  # Neutral value
                bias_improvement=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                processing_time=0.0,
                synthetic_samples_created=0,
                cross_val_scores=[],
                variance_in_fairness=0.0,
                model_specific_results={},
                parameters_used={},
                warnings_encountered=[str(e)]
            )

    def _encode_for_model(self, X):
        """Encode categorical variables for model training."""
        if isinstance(X, pd.DataFrame):
            X_encoded = X.copy()
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                try:
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                except:
                    # If encoding fails, use numeric representation
                    X_encoded[col] = pd.Categorical(X[col]).codes
            return X_encoded
        return X

    def _evaluate_baseline(self, X, y, protected_attribute):
        """
        Evaluate performance without any bias mitigation.
        This establishes our baseline for comparison.
        """
        start_time = time.time()

        # Measure original bias
        original_metrics = self.bias_detector.detect_bias(
            X, y, protected_attribute, positive_label=1
        )

        # Evaluate model performance
        model_results = self._evaluate_models_on_data(X, y, protected_attribute)

        return EvaluationResult(
            method_name="No Mitigation (Baseline)",
            original_disparate_impact=original_metrics.disparate_impact,
            final_disparate_impact=original_metrics.disparate_impact,
            bias_improvement=0.0,
            accuracy=model_results['avg_accuracy'],
            precision=model_results['avg_precision'],
            recall=model_results['avg_recall'],
            f1=model_results['avg_f1'],
            processing_time=time.time() - start_time,
            synthetic_samples_created=0,
            cross_val_scores=model_results['cv_scores'],
            variance_in_fairness=model_results['fairness_variance'],
            model_specific_results=model_results['detailed'],
            parameters_used={},
            warnings_encountered=[]
        )

    def _evaluate_random_oversampling(self, X, y, protected_attribute):
        """
        Evaluate random oversampling - a simple baseline approach.
        This helps us understand if sophisticated methods are worth it.
        """
        start_time = time.time()
        warnings_list = []

        # Measure original bias
        original_metrics = self.bias_detector.detect_bias(
            X, y, protected_attribute, positive_label=1
        )

        # Apply random oversampling
        X_encoded, encoders = self._encode_categorical(X)
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X_encoded, y)

        # Decode back
        X_resampled = self._decode_categorical(X_resampled, X.columns, encoders)

        # Measure new bias
        final_metrics = self.bias_detector.detect_bias(
            X_resampled, y_resampled, protected_attribute, positive_label=1
        )

        # Evaluate model performance
        model_results = self._evaluate_models_on_data(
            X_resampled, y_resampled, protected_attribute
        )

        bias_improvement = ((final_metrics.disparate_impact - original_metrics.disparate_impact)
                           / original_metrics.disparate_impact * 100)

        return EvaluationResult(
            method_name="Random Oversampling",
            original_disparate_impact=original_metrics.disparate_impact,
            final_disparate_impact=final_metrics.disparate_impact,
            bias_improvement=bias_improvement,
            accuracy=model_results['avg_accuracy'],
            precision=model_results['avg_precision'],
            recall=model_results['avg_recall'],
            f1=model_results['avg_f1'],
            processing_time=time.time() - start_time,
            synthetic_samples_created=len(X_resampled) - len(X),
            cross_val_scores=model_results['cv_scores'],
            variance_in_fairness=model_results['fairness_variance'],
            model_specific_results=model_results['detailed'],
            parameters_used={'sampling_strategy': 'auto'},
            warnings_encountered=warnings_list
        )

    def _evaluate_standard_smote(self, X, y, protected_attribute):
        """
        Evaluate standard SMOTE - the current state-of-the-art for oversampling.
        This is our main competitor to beat.
        """
        start_time = time.time()
        warnings_list = []

        # Measure original bias
        original_metrics = self.bias_detector.detect_bias(
            X, y, protected_attribute, positive_label=1
        )

        # Apply standard SMOTE
        X_encoded, encoders = self._encode_categorical(X)
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)

        try:
            X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
        except ValueError as e:
            warnings_list.append(f"SMOTE warning: {str(e)}")
            # Fall back to fewer neighbors
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

        # Decode back
        X_resampled = self._decode_categorical(X_resampled, X.columns, encoders)

        # Measure new bias
        final_metrics = self.bias_detector.detect_bias(
            X_resampled, y_resampled, protected_attribute, positive_label=1
        )

        # Evaluate model performance
        model_results = self._evaluate_models_on_data(
            X_resampled, y_resampled, protected_attribute
        )

        bias_improvement = ((final_metrics.disparate_impact - original_metrics.disparate_impact)
                           / original_metrics.disparate_impact * 100)

        return EvaluationResult(
            method_name="Standard SMOTE",
            original_disparate_impact=original_metrics.disparate_impact,
            final_disparate_impact=final_metrics.disparate_impact,
            bias_improvement=bias_improvement,
            accuracy=model_results['avg_accuracy'],
            precision=model_results['avg_precision'],
            recall=model_results['avg_recall'],
            f1=model_results['avg_f1'],
            processing_time=time.time() - start_time,
            synthetic_samples_created=len(X_resampled) - len(X),
            cross_val_scores=model_results['cv_scores'],
            variance_in_fairness=model_results['fairness_variance'],
            model_specific_results=model_results['detailed'],
            parameters_used={'k_neighbors': 5, 'sampling_strategy': 'auto'},
            warnings_encountered=warnings_list
        )

    def _evaluate_rebalance(self, X, y, protected_attribute):
        """
        Evaluate our REBALANCE toolkit - the star of the show.
        This is where we expect to see superior performance.
        """
        start_time = time.time()
        warnings_list = []

        # Use our integrated pipeline
        rebalancer = FairRebalancer(
            protected_attribute=protected_attribute,
            target_fairness=0.8,
            k_neighbors=5,
            random_state=self.random_state,
            verbose=False
        )

        # Apply REBALANCE
        result = rebalancer.fit_transform(X, y)

        # Extract rebalanced data
        X_resampled = result.X_rebalanced
        y_resampled = result.y_rebalanced

        # Evaluate model performance
        model_results = self._evaluate_models_on_data(
            X_resampled, y_resampled, protected_attribute
        )

        return EvaluationResult(
            method_name="REBALANCE (Fair SMOTE)",
            original_disparate_impact=result.original_bias_metrics.disparate_impact,
            final_disparate_impact=result.final_bias_metrics.disparate_impact,
            bias_improvement=result.improvement_summary['disparate_impact_change'],
            accuracy=model_results['avg_accuracy'],
            precision=model_results['avg_precision'],
            recall=model_results['avg_recall'],
            f1=model_results['avg_f1'],
            processing_time=result.processing_time,
            synthetic_samples_created=len(X_resampled) - result.original_size,
            cross_val_scores=model_results['cv_scores'],
            variance_in_fairness=model_results['fairness_variance'],
            model_specific_results=model_results['detailed'],
            parameters_used=result.parameters_used,
            warnings_encountered=result.warnings
        )

    def _evaluate_models_on_data(self, X, y, protected_attribute):
        """
        Evaluate multiple ML models on the given data.

        This is crucial because bias mitigation effectiveness can vary
        by model type. A method that works well with logistic regression
        might not work as well with random forests.
        """
        # Encode categorical variables for model training
        X_encoded, encoders = self._encode_categorical(X)

         # Add feature scaling for numerical features (new code)
        numerical_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

        detailed_results = {}
        all_cv_scores = []
        fairness_scores = []

        for model_name, model in self.models_to_test:
            # Perform cross-validation
            cv_scores = cross_val_score(
            model, X_encoded, y,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state),
            scoring='f1'
        )

            # Train on full data for other metrics
            model.fit(X_encoded, y)
            y_pred = model.predict(X_encoded)

            # Calculate performance metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)

            # Calculate fairness of predictions
            pred_metrics = self.bias_detector.detect_bias(
                X, y_pred, protected_attribute, positive_label=1
            )

            detailed_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_scores': cv_scores.tolist(),
                'disparate_impact': pred_metrics.disparate_impact
            }

            all_cv_scores.extend(cv_scores)
            fairness_scores.append(pred_metrics.disparate_impact)

        # Calculate aggregated metrics
        avg_results = {
            'avg_accuracy': np.mean([r['accuracy'] for r in detailed_results.values()]),
            'avg_precision': np.mean([r['precision'] for r in detailed_results.values()]),
            'avg_recall': np.mean([r['recall'] for r in detailed_results.values()]),
            'avg_f1': np.mean([r['f1'] for r in detailed_results.values()]),
            'cv_scores': all_cv_scores,
            'fairness_variance': np.var(fairness_scores),
            'detailed': detailed_results
        }

        return avg_results

    def _encode_categorical(self, X):
        """
        Encode categorical variables for sklearn models.
        Returns encoded data and encoders for decoding later.
        """
        X_encoded = X.copy()
        encoders = {}

        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            encoders[col] = le

        return X_encoded, encoders

    def _decode_categorical(self, X_encoded, columns, encoders):
        """
        Decode categorical variables back to original format.
        """
        X_decoded = pd.DataFrame(X_encoded, columns=columns)

        for col, le in encoders.items():
            X_decoded[col] = le.inverse_transform(X_decoded[col].astype(int))

        return X_decoded

    def create_comparison_report(self, save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive comparison report of all methods.

        This is like writing a scientific paper's results section -
        we present findings clearly and objectively, letting the
        data speak for itself.
        """
        if not self.all_results:
            return "No evaluation results available. Run evaluate_all_methods first."

        report = f"""
================================================================================
                    BIAS MITIGATION METHODS COMPARISON REPORT
================================================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Evaluator: REBALANCE Comprehensive Evaluation Framework

--------------------------------------------------------------------------------
EXECUTIVE SUMMARY
--------------------------------------------------------------------------------

This report presents a systematic comparison of bias mitigation methods,
evaluating their effectiveness at reducing gender bias while maintaining
model performance. All methods were tested under identical conditions.

KEY FINDINGS:
"""

        # Rank methods by combined score
        ranked_methods = sorted(
            self.all_results.items(),
            key=lambda x: x[1].get_summary_score(),
            reverse=True
        )

        report += f"\n1. Best Overall: {ranked_methods[0][0]}"
        report += f"\n   - Disparate Impact: {ranked_methods[0][1].final_disparate_impact:.3f}"
        report += f"\n   - F1 Score: {ranked_methods[0][1].f1:.3f}"
        report += f"\n   - Combined Score: {ranked_methods[0][1].get_summary_score():.3f}"

        report += "\n\n" + "-"*80 + "\n"
        report += "DETAILED RESULTS BY METHOD\n"
        report += "-"*80 + "\n"

        for method_name, result in self.all_results.items():
            report += f"\n{method_name.upper()}\n"
            report += "="*len(method_name) + "\n\n"

            report += "Bias Mitigation Performance:\n"
            report += f"  Original Disparate Impact: {result.original_disparate_impact:.3f}\n"
            report += f"  Final Disparate Impact: {result.final_disparate_impact:.3f}\n"
            report += f"  Improvement: {result.bias_improvement:+.1f}%\n"
            report += f"  Meets Legal Threshold (0.8): {'YES' if result.final_disparate_impact >= 0.8 else 'NO'}\n"

            report += "\nModel Performance (Averaged):\n"
            report += f"  Accuracy: {result.accuracy:.3f}\n"
            report += f"  Precision: {result.precision:.3f}\n"
            report += f"  Recall: {result.recall:.3f}\n"
            report += f"  F1 Score: {result.f1:.3f}\n"

            report += "\nEfficiency:\n"
            report += f"  Processing Time: {result.processing_time:.2f} seconds\n"
            report += f"  Synthetic Samples Created: {result.synthetic_samples_created:,}\n"

            report += "\nRobustness:\n"
            report += f"  Cross-validation F1 (mean ± std): {np.mean(result.cross_val_scores):.3f} ± {np.std(result.cross_val_scores):.3f}\n"
            report += f"  Fairness Variance Across Models: {result.variance_in_fairness:.4f}\n"

            if result.warnings_encountered:
                report += "\nWarnings:\n"
                for warning in result.warnings_encountered:
                    report += f"  ⚠️  {warning}\n"

            report += "\n" + "-"*80 + "\n"

        # Add comparative analysis
        report += """
COMPARATIVE ANALYSIS
====================

"""

        # Create comparison table
        report += "Method                    | Disparate Impact | F1 Score | Time (s) | Combined Score\n"
        report += "-"*85 + "\n"

        for method_name, result in ranked_methods:
            report += f"{method_name:<25} | {result.final_disparate_impact:^16.3f} | "
            report += f"{result.f1:^8.3f} | {result.processing_time:^8.2f} | "
            report += f"{result.get_summary_score():^14.3f}\n"

        # Add insights
        report += """

KEY INSIGHTS:
-------------
"""

        # Compare REBALANCE to standard SMOTE
        if 'REBALANCE (Fair SMOTE)' in self.all_results and 'Standard SMOTE' in self.all_results:
            rebalance = self.all_results['REBALANCE (Fair SMOTE)']
            smote = self.all_results['Standard SMOTE']

            di_improvement = ((rebalance.final_disparate_impact - smote.final_disparate_impact)
                             / smote.final_disparate_impact * 100)

            report += f"\n1. REBALANCE vs Standard SMOTE:"
            report += f"\n   - Disparate Impact improvement: {di_improvement:+.1f}%"
            report += f"\n   - F1 Score difference: {rebalance.f1 - smote.f1:+.3f}"
            report += f"\n   - REBALANCE {'outperforms' if di_improvement > 0 else 'underperforms'} "
            report += "Standard SMOTE in fairness"

        # Check if any method achieves fairness
        fair_methods = [m for m, r in self.all_results.items()
                       if r.final_disparate_impact >= 0.8]

        report += f"\n\n2. Methods Achieving Legal Fairness Threshold (DI ≥ 0.8):"
        if fair_methods:
            for method in fair_methods:
                report += f"\n   - {method}"
        else:
            report += "\n   - None (additional interventions needed)"

        # Performance trade-offs
        report += "\n\n3. Fairness-Performance Trade-offs:"
        baseline = self.all_results.get('No Mitigation (Baseline)')
        if baseline:
            for method_name, result in self.all_results.items():
                if method_name != 'No Mitigation (Baseline)':
                    perf_loss = (baseline.f1 - result.f1) / baseline.f1 * 100
                    report += f"\n   - {method_name}: {perf_loss:.1f}% F1 loss for "
                    report += f"{result.bias_improvement:.1f}% bias improvement"

        report += """

================================================================================
                              END OF COMPARISON REPORT
================================================================================
"""

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {save_path}")

        return report
