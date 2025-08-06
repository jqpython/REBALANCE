"""
Bias Detection Module for REBALANCE Toolkit

This module provides comprehensive bias detection capabilities for
machine learning datasets, with a focus on gender bias in employment contexts.
It implements multiple fairness metrics and provides interpretable results.

Author: jqpythonai
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BiasMetrics:
    """
    Data class to store bias metrics in a structured way.

    Think of this as a medical report that contains all the test results.
    Each metric tells us something different about the bias, just like
    different blood tests tell us different things about health.
    """
    disparate_impact: float
    statistical_parity_difference: float
    demographic_parity_ratio: float
    equal_opportunity_difference: float
    group_sizes: Dict[str, int]
    group_positive_rates: Dict[str, float]

    def is_biased(self, threshold: float = 0.8) -> bool:
        """
        Determine if bias exists based on the 80% rule (four-fifths rule).
        This is a legal standard used in employment discrimination cases.
        """
        return self.disparate_impact < threshold

    def get_bias_severity(self) -> str:
        """
        Categorize bias severity to make results more interpretable.
        Like a traffic light system: green (ok), yellow (caution), red (problem).
        """
        if self.disparate_impact >= 0.8:
            return "No significant bias detected"
        elif self.disparate_impact >= 0.6:
            return "Moderate bias detected"
        elif self.disparate_impact >= 0.4:
            return "Significant bias detected"
        else:
            return "Severe bias detected"


class BiasDetector:
    """
    Main class for detecting bias in datasets.

    This is like a Swiss Army knife for bias detection - it has multiple
    tools (metrics) that each serve a different purpose in understanding bias.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the bias detector.

        Parameters:
        -----------
        verbose : bool
            If True, prints helpful information during detection process.
            Like having a guide explain what's happening during a tour.
        """
        self.verbose = verbose
        self.last_analysis = None  # Store the last analysis for reporting

    def detect_bias(self,
                   X: pd.DataFrame,
                   y: Union[pd.Series, np.ndarray],
                   protected_attribute: str = 'gender',
                   positive_label: Any = 1) -> BiasMetrics:
        """
        Main method to detect bias in a dataset.

        Think of this as running a complete diagnostic test - it performs
        all the individual tests and compiles them into a single report.

        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe containing the protected attribute
        y : pd.Series or np.ndarray
            Target variable (what we're predicting)
        protected_attribute : str
            Column name of the protected attribute (e.g., 'gender', 'sex')
        positive_label : Any
            What counts as a positive outcome (e.g., '>50K', 1, 'hired')

        Returns:
        --------
        BiasMetrics object containing all calculated metrics
        """
        if self.verbose:
            print("Starting bias detection analysis...")
            print(f"Protected attribute: {protected_attribute}")
            print(f"Positive outcome label: {positive_label}")
            print("-" * 50)

        # First, let's validate our inputs - like checking if our equipment works
        self._validate_inputs(X, y, protected_attribute)

        # Convert y to binary if needed (simplifies calculations)
        y_binary = self._prepare_target(y, positive_label)

        # Get the protected attribute values
        protected_values = X[protected_attribute]

        # Calculate group-wise statistics - the foundation of all bias metrics
        group_stats = self._calculate_group_statistics(protected_values, y_binary)

        # Now calculate each bias metric
        metrics = self._calculate_all_metrics(group_stats)

        # Store for later reporting
        self.last_analysis = {
            'metrics': metrics,
            'group_stats': group_stats,
            'protected_attribute': protected_attribute,
            'dataset_size': len(X)
        }

        if self.verbose:
            self._print_summary(metrics)

        return metrics

    def _validate_inputs(self, X: pd.DataFrame, y: Any, protected_attribute: str):
        """
        Validate inputs to ensure they're suitable for bias detection.
        Like a pre-flight checklist - making sure everything is ready.
        """
        if protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataframe. "
                           f"Available columns: {list(X.columns)}")

        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")

        # Check if protected attribute has exactly 2 unique values
        unique_values = X[protected_attribute].unique()
        if len(unique_values) != 2:
            warnings.warn(f"Protected attribute has {len(unique_values)} unique values. "
                         f"This implementation assumes binary protected attributes. "
                         f"Values found: {unique_values}")

    def _prepare_target(self, y: Any, positive_label: Any) -> np.ndarray:
        """
        Convert target variable to binary format.
        Like converting different temperature scales to Celsius for consistency.
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  # Take first column if DataFrame

        return (y == positive_label).astype(int)

    def _calculate_group_statistics(self,
                                  protected_values: pd.Series,
                                  y_binary: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each group in the protected attribute.
        This is like taking measurements of different populations.
        """
        group_stats = {}

        for group in protected_values.unique():
            # Create a mask for this group
            group_mask = protected_values == group

            # Calculate statistics
            group_size = group_mask.sum()
            group_positives = y_binary[group_mask].sum()
            group_positive_rate = group_positives / group_size if group_size > 0 else 0

            group_stats[group] = {
                'size': group_size,
                'positives': group_positives,
                'positive_rate': group_positive_rate,
                'mask': group_mask
            }

            if self.verbose:
                print(f"\nGroup '{group}':")
                print(f"  Size: {group_size:,} ({group_size/len(protected_values)*100:.1f}%)")
                print(f"  Positive outcomes: {group_positives:,}")
                print(f"  Positive rate: {group_positive_rate:.3f}")

        return group_stats

    def _calculate_all_metrics(self, group_stats: Dict[str, Dict[str, Any]]) -> BiasMetrics:
        """
        Calculate comprehensive bias metrics from group statistics.
        Each metric provides a different perspective on fairness.
        """
        # Identify privileged and unprivileged groups
        # We'll use the group with higher positive rate as privileged
        groups = list(group_stats.keys())
        if group_stats[groups[0]]['positive_rate'] > group_stats[groups[1]]['positive_rate']:
            privileged_group, unprivileged_group = groups[0], groups[1]
        else:
            privileged_group, unprivileged_group = groups[1], groups[0]

        priv_stats = group_stats[privileged_group]
        unpriv_stats = group_stats[unprivileged_group]

        # Calculate Disparate Impact (DI) with proper handling
        # DI = P(Y=1|unprivileged) / P(Y=1|privileged)
        if priv_stats['positive_rate'] == 0 and unpriv_stats['positive_rate'] == 0:
            # Both groups have zero positive rate - no meaningful DI
            disparate_impact = 1.0  # Treat as equal (no bias)
            warnings.warn("Both groups have zero positive rate. Setting DI to 1.0")
        elif priv_stats['positive_rate'] == 0:
            # Only privileged group has zero rate - this indicates reverse bias
            # Set to a large value to indicate bias favoring unprivileged group
            disparate_impact = float('inf')  # Or use a large number like 999.0
            warnings.warn(f"Privileged group ({privileged_group}) has zero positive rate. "
                        f"This indicates unusual bias favoring {unprivileged_group}")
        else:
            # Normal calculation
            disparate_impact = unpriv_stats['positive_rate'] / priv_stats['positive_rate']

        # Ensure DI is within reasonable bounds for downstream processing
        if disparate_impact == float('inf'):
            disparate_impact = 999.0  # Cap at a large but finite value

        # Calculate Statistical Parity Difference (SPD)
        # SPD = P(Y=1|unprivileged) - P(Y=1|privileged)
        # Should be close to 0 for fairness
        statistical_parity_diff = unpriv_stats['positive_rate'] - priv_stats['positive_rate']

        # Calculate Demographic Parity Ratio (DPR)
        # Similar to DI but sometimes preferred in literature
        demographic_parity_ratio = disparate_impact  # Same calculation, different name

        # Calculate Equal Opportunity Difference (EOD)
        # This requires knowing the true positives, which we'll approximate
        # For now, we'll set this to SPD (can be refined with ground truth)
        equal_opportunity_diff = statistical_parity_diff

        # Compile group information
        group_sizes = {group: stats['size'] for group, stats in group_stats.items()}
        group_positive_rates = {group: stats['positive_rate'] for group, stats in group_stats.items()}

          # Add validation check
        if disparate_impact == 0:
            warnings.warn("Disparate impact is exactly 0. This likely indicates a data issue. "
                        f"Positive rates: {group_positive_rates}")

        return BiasMetrics(
            disparate_impact=disparate_impact,
            statistical_parity_difference=statistical_parity_diff,
            demographic_parity_ratio=demographic_parity_ratio,
            equal_opportunity_difference=equal_opportunity_diff,
            group_sizes=group_sizes,
            group_positive_rates=group_positive_rates
        )

    def _print_summary(self, metrics: BiasMetrics):
        """
        Print a human-readable summary of bias metrics.
        Like a doctor explaining test results to a patient.
        """
        print("\n" + "="*50)
        print("BIAS DETECTION RESULTS")
        print("="*50)

        print(f"\nDisparate Impact: {metrics.disparate_impact:.3f}")
        print(f"  Interpretation: {metrics.get_bias_severity()}")
        print(f"  Legal threshold (80% rule): {'PASS' if metrics.disparate_impact >= 0.8 else 'FAIL'}")

        print(f"\nStatistical Parity Difference: {metrics.statistical_parity_difference:.3f}")
        print(f"  Interpretation: {abs(metrics.statistical_parity_difference)*100:.1f}% "
              f"difference in positive outcome rates")

        print(f"\nGroup Analysis:")
        for group, rate in metrics.group_positive_rates.items():
            size = metrics.group_sizes[group]
            print(f"  {group}: {rate:.1%} positive rate (n={size:,})")

    def visualize_bias(self, save_path: Optional[str] = None):
        """
        Create visualizations of the bias analysis.
        A picture is worth a thousand words - especially for stakeholders.
        """
        if self.last_analysis is None:
            raise ValueError("No analysis to visualize. Run detect_bias() first.")

        metrics = self.last_analysis['metrics']

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bias Detection Analysis', fontsize=16, fontweight='bold')

        # 1. Group Positive Rates Comparison
        ax1 = axes[0, 0]
        groups = list(metrics.group_positive_rates.keys())
        rates = list(metrics.group_positive_rates.values())
        bars = ax1.bar(groups, rates, color=['lightcoral', 'lightblue'])
        ax1.set_ylabel('Positive Outcome Rate')
        ax1.set_title('Positive Outcome Rates by Group')
        ax1.set_ylim(0, max(rates) * 1.2)

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')

        # 2. Disparate Impact Visualization
        ax2 = axes[0, 1]
        di_value = metrics.disparate_impact
        colors = ['red' if di_value < 0.8 else 'green']
        ax2.barh(['Disparate Impact'], [di_value], color=colors)
        ax2.axvline(x=0.8, color='black', linestyle='--', label='Legal Threshold (0.8)')
        ax2.set_xlim(0, 1.2)
        ax2.set_xlabel('Disparate Impact Ratio')
        ax2.set_title('Disparate Impact Analysis')
        ax2.legend()

        # Add text annotation
        ax2.text(di_value + 0.02, 0, f'{di_value:.3f}', va='center')

        # 3. Group Size Distribution
        ax3 = axes[1, 0]
        sizes = list(metrics.group_sizes.values())
        ax3.pie(sizes, labels=groups, autopct='%1.1f%%', startangle=90,
                colors=['lightcoral', 'lightblue'])
        ax3.set_title('Group Size Distribution')

        # 4. Bias Summary Text
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
        BIAS DETECTION SUMMARY

        Protected Attribute: {self.last_analysis['protected_attribute']}
        Dataset Size: {self.last_analysis['dataset_size']:,}

        Disparate Impact: {metrics.disparate_impact:.3f}
        Status: {metrics.get_bias_severity()}

        The unprivileged group is {(1-metrics.disparate_impact)*100:.1f}%
        less likely to receive positive outcomes.

        This level of bias {'requires' if metrics.is_biased() else 'may require'}
        mitigation for fair ML models.
        """

        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"\nVisualization saved to: {save_path}")

        plt.show()

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of the bias analysis.
        This is what you might send to stakeholders or include in documentation.
        """
        if self.last_analysis is None:
            raise ValueError("No analysis to report. Run detect_bias() first.")

        metrics = self.last_analysis['metrics']

        report = f"""
BIAS DETECTION REPORT
====================

Dataset Information:
- Total Samples: {self.last_analysis['dataset_size']:,}
- Protected Attribute: {self.last_analysis['protected_attribute']}

Group Distribution:
"""
        for group, size in metrics.group_sizes.items():
            percentage = (size / self.last_analysis['dataset_size']) * 100
            report += f"- {group}: {size:,} ({percentage:.1f}%)\n"

        report += f"""
Bias Metrics:
1. Disparate Impact: {metrics.disparate_impact:.3f}
   - Legal Threshold (0.8): {'PASS' if metrics.disparate_impact >= 0.8 else 'FAIL'}
   - Interpretation: {metrics.get_bias_severity()}

2. Statistical Parity Difference: {metrics.statistical_parity_difference:.3f}
   - Ideal Value: 0.000
   - Current Gap: {abs(metrics.statistical_parity_difference)*100:.1f}%

3. Group Positive Rates:
"""
        for group, rate in metrics.group_positive_rates.items():
            report += f"   - {group}: {rate:.1%}\n"

        report += f"""
Recommendations:
"""
        if metrics.is_biased():
            report += """- Significant bias detected. Consider applying bias mitigation techniques.
- Recommended approach: Fairness-aware synthetic data generation (SMOTE).
- Target: Increase disparate impact to at least 0.8."""
        else:
            report += """- No significant bias detected based on the 80% rule.
- Continue monitoring for bias in model predictions.
- Consider checking for bias in other protected attributes."""

        report += """

Note: This analysis assumes the dataset is representative of the population.
Bias in data collection or sampling may not be detected by these metrics.
"""

        return report
