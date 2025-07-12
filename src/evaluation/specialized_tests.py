# File: src/evaluation/specialized_tests.py

"""
Specialized Evaluation Scenarios for REBALANCE

While comprehensive evaluation gives us the big picture, specialized
tests help us understand the nuances of when and why REBALANCE excels.
These tests are designed to highlight specific advantages of the
fairness-aware approach.

Think of these as targeted experiments that explore edge cases and
specific scenarios where traditional methods struggle.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class SpecializedEvaluator:
    """
    Conducts specialized tests to demonstrate REBALANCE's unique capabilities.

    Each test is designed to answer a specific question about the toolkit's
    performance in challenging scenarios that arise in real-world applications.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_results = {}

    def run_all_tests(self, data_path: str) -> Dict[str, any]:
        """
        Run all specialized tests on the given dataset.

        Each test explores a different aspect of bias mitigation,
        helping us understand not just if REBALANCE works, but
        why and when it works best.
        """
        # Load the full dataset
        data = pd.read_csv(data_path)

        if self.verbose:
            print("\n" + "="*60)
            print("SPECIALIZED EVALUATION SCENARIOS")
            print("="*60)

        # Test 1: Extreme imbalance scenario
        self.test_extreme_imbalance(data)

        # Test 2: Intersectional bias (gender + another attribute)
        self.test_intersectional_bias(data)

        # Test 3: Small sample size robustness
        self.test_small_sample_robustness(data)

        # Test 4: Feature preservation quality
        self.test_feature_preservation(data)

        # Test 5: Scalability test
        self.test_scalability(data)

        return self.test_results

    def test_extreme_imbalance(self, data: pd.DataFrame):
        """
        Test how methods handle extreme gender imbalance.

        In real datasets, we might encounter situations where one gender
        is severely underrepresented in the positive class. This tests
        how well each method handles such extreme scenarios.
        """
        if self.verbose:
            print("\n📊 TEST 1: Extreme Imbalance Handling")
            print("-" * 40)

        # Create an extremely imbalanced scenario
        # Keep only 5% of high-income women
        high_income_women_mask = (data['sex'] == 'Female') & (data['income'] == '>50K')
        high_income_women = data[high_income_women_mask]

        # Sample only 5% of high-income women
        n_keep = max(10, int(len(high_income_women) * 0.05))  # Keep at least 10
        sampled_women = high_income_women.sample(n=n_keep, random_state=42)

        # Combine with all other data
        other_data = data[~high_income_women_mask]
        extreme_data = pd.concat([sampled_women, other_data])

        # Prepare for testing
        X = extreme_data.drop(['income', 'high_income', 'is_female_high_income'], axis=1)
        y = (extreme_data['income'] == '>50K').astype(int)

        # Count high-income women
        remaining_high_women = ((X['sex'] == 'Female') & (y == 1)).sum()
        total_high = y.sum()

        if self.verbose:
            print(f"Created extreme scenario:")
            print(f"  High-income women: {remaining_high_women} ({remaining_high_women/total_high*100:.1f}%)")
            print(f"  Total high-income: {total_high}")

        # Test each method's ability to handle this
        from ..bias_detection.detector import BiasDetector
        detector = BiasDetector(verbose=False)

        # Original bias
        orig_metrics = detector.detect_bias(X, y, 'sex', 1)

        # Test REBALANCE
        from ..rebalance import FairRebalancer
        rebalancer = FairRebalancer(protected_attribute='sex', verbose=False)
        result = rebalancer.fit_transform(X, y)

        # Measure improvement
        improvement = ((result.final_bias_metrics.disparate_impact -
                       orig_metrics.disparate_impact) /
                      orig_metrics.disparate_impact * 100)

        if self.verbose:
            print(f"\nResults:")
            print(f"  Original DI: {orig_metrics.disparate_impact:.3f}")
            print(f"  REBALANCE DI: {result.final_bias_metrics.disparate_impact:.3f}")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Synthetic women created: {((result.X_rebalanced['sex'] == 'Female') & (result.y_rebalanced == 1)).sum() - remaining_high_women}")

        self.test_results['extreme_imbalance'] = {
            'original_di': orig_metrics.disparate_impact,
            'final_di': result.final_bias_metrics.disparate_impact,
            'improvement': improvement,
            'handles_extreme_bias': result.final_bias_metrics.disparate_impact > 0.5
        }

    def test_intersectional_bias(self, data: pd.DataFrame):
        """
        Test handling of intersectional bias (e.g., gender + race).

        Real-world bias often involves multiple attributes. This tests
        whether our approach can handle complex, multi-dimensional bias.
        """
        if self.verbose:
            print("\n📊 TEST 2: Intersectional Bias (Gender + Race)")
            print("-" * 40)

        # Focus on a subset with clear intersectional patterns
        subset = data[data['race'].isin(['White', 'Black'])].copy()

        # Create intersectional groups
        subset['intersectional_group'] = subset['sex'] + '_' + subset['race']

        X = subset.drop(['income', 'high_income', 'is_female_high_income'], axis=1)
        y = (subset['income'] == '>50K').astype(int)

        # Analyze intersectional bias
        groups = X['intersectional_group'].unique()
        group_rates = {}

        for group in groups:
            mask = X['intersectional_group'] == group
            if mask.sum() > 0:
                rate = y[mask].mean()
                group_rates[group] = rate
                if self.verbose:
                    print(f"  {group}: {rate:.3f} ({mask.sum()} samples)")

        # Find most and least advantaged groups
        max_group = max(group_rates, key=group_rates.get)
        min_group = min(group_rates, key=group_rates.get)
        intersectional_di = group_rates[min_group] / group_rates[max_group]

        if self.verbose:
            print(f"\nIntersectional Disparate Impact: {intersectional_di:.3f}")
            print(f"Most advantaged: {max_group} ({group_rates[max_group]:.3f})")
            print(f"Least advantaged: {min_group} ({group_rates[min_group]:.3f})")

        self.test_results['intersectional_bias'] = {
            'group_rates': group_rates,
            'intersectional_di': intersectional_di,
            'most_advantaged': max_group,
            'least_advantaged': min_group
        }

    def test_small_sample_robustness(self, data: pd.DataFrame):
        """
        Test performance with small sample sizes.

        Many real-world datasets are small, especially in specialized
        domains. This tests whether REBALANCE can still work effectively
        with limited data.
        """
        if self.verbose:
            print("\n📊 TEST 3: Small Sample Size Robustness")
            print("-" * 40)

        sample_sizes = [500, 1000, 2000, 5000]
        results = {}

        for size in sample_sizes:
            # Sample data
            sampled = data.sample(n=min(size, len(data)), random_state=42)
            X = sampled.drop(['income', 'high_income', 'is_female_high_income'], axis=1)
            y = (sampled['income'] == '>50K').astype(int)

            # Check if we have enough samples of minority class
            min_class_size = y.sum()
            if min_class_size < 10:
                if self.verbose:
                    print(f"  {size}: Skipped (too few positive samples)")
                continue

            # Test REBALANCE
            from ..rebalance import FairRebalancer
            rebalancer = FairRebalancer(protected_attribute='sex', verbose=False)

            try:
                result = rebalancer.fit_transform(X, y)
                success = True
                di = result.final_bias_metrics.disparate_impact
            except Exception as e:
                success = False
                di = None
                if self.verbose:
                    print(f"  {size}: Failed - {str(e)}")

            results[size] = {
                'success': success,
                'disparate_impact': di,
                'minority_class_size': min_class_size
            }

            if success and self.verbose:
                print(f"  {size}: Success - DI = {di:.3f}")

        self.test_results['small_sample_robustness'] = results

    def test_feature_preservation(self, data: pd.DataFrame):
        """
        Test how well synthetic samples preserve feature relationships.

        A key quality metric for synthetic data is whether it maintains
        the statistical properties and relationships of the original data.
        This test measures that preservation.
        """
        if self.verbose:
            print("\n📊 TEST 4: Feature Preservation Quality")
            print("-" * 40)

        # Focus on numerical features for correlation analysis
        numerical_features = ['age', 'education-num', 'hours-per-week',
                            'capital-gain', 'capital-loss']

        subset = data[data['education'] == 'Bachelors'].copy()  # Focus on one education level
        X = subset.drop(['income', 'high_income', 'is_female_high_income'], axis=1)
        y = (subset['income'] == '>50K').astype(int)

        # Calculate original correlations for high-income women
        high_income_women_mask = (X['sex'] == 'Female') & (y == 1)
        original_data = X[high_income_women_mask][numerical_features]
        original_corr = original_data.corr()

        # Apply REBALANCE
        from ..rebalance import FairRebalancer
        rebalancer = FairRebalancer(protected_attribute='sex', verbose=False)
        result = rebalancer.fit_transform(X, y)

        # Identify synthetic samples (those after original size)
        synthetic_mask = np.zeros(len(result.X_rebalanced), dtype=bool)
        synthetic_mask[len(X):] = True

        # Get synthetic high-income women
        synthetic_women_mask = (synthetic_mask &
                               (result.X_rebalanced['sex'] == 'Female') &
                               (result.y_rebalanced == 1))

        if synthetic_women_mask.sum() > 10:  # Need enough samples for correlation
            synthetic_data = result.X_rebalanced[synthetic_women_mask][numerical_features]
            synthetic_corr = synthetic_data.corr()

            # Calculate correlation preservation (how similar are correlation matrices)
            corr_diff = np.abs(original_corr - synthetic_corr)
            avg_corr_diff = corr_diff.mean().mean()

            # Check feature distributions
            distribution_tests = {}
            for feature in numerical_features:
                orig_mean = original_data[feature].mean()
                orig_std = original_data[feature].std()
                synth_mean = synthetic_data[feature].mean()
                synth_std = synthetic_data[feature].std()

                # Calculate normalized differences
                mean_diff = abs(orig_mean - synth_mean) / (orig_mean + 1e-8)
                std_diff = abs(orig_std - synth_std) / (orig_std + 1e-8)

                distribution_tests[feature] = {
                    'mean_preservation': 1 - mean_diff,  # Closer to 1 is better
                    'std_preservation': 1 - std_diff
                }

            if self.verbose:
                print(f"Correlation matrix difference: {avg_corr_diff:.3f}")
                print(f"Feature distribution preservation:")
                for feature, scores in distribution_tests.items():
                    print(f"  {feature}: mean={scores['mean_preservation']:.3f}, "
                          f"std={scores['std_preservation']:.3f}")

            self.test_results['feature_preservation'] = {
                'correlation_difference': avg_corr_diff,
                'distribution_preservation': distribution_tests,
                'overall_quality': 1 - avg_corr_diff  # Higher is better
            }
        else:
            if self.verbose:
                print("Not enough synthetic samples for analysis")
            self.test_results['feature_preservation'] = None

    def test_scalability(self, data: pd.DataFrame):
        """
        Test how processing time scales with dataset size.

        For practical deployment, we need to understand how the toolkit
        performs as datasets grow. This test measures scalability.
        """
        if self.verbose:
            print("\n📊 TEST 5: Scalability Analysis")
            print("-" * 40)

        import time

        sizes = [1000, 5000, 10000, 20000, len(data)]
        times = []

        for size in sizes:
            if size > len(data):
                break

            # Sample data
            sampled = data.sample(n=size, random_state=42)
            X = sampled.drop(['income', 'high_income', 'is_female_high_income'], axis=1)
            y = (sampled['income'] == '>50K').astype(int)

            # Time REBALANCE
            from ..rebalance import FairRebalancer
            rebalancer = FairRebalancer(protected_attribute='sex', verbose=False)

            start_time = time.time()
            try:
                result = rebalancer.fit_transform(X, y)
                elapsed = time.time() - start_time
                times.append((size, elapsed))

                if self.verbose:
                    print(f"  {size:,} samples: {elapsed:.2f} seconds")
            except Exception as e:
                if self.verbose:
                    print(f"  {size:,} samples: Failed - {str(e)}")

        # Analyze scaling pattern
        if len(times) >= 3:
            sizes_array = np.array([t[0] for t in times])
            times_array = np.array([t[1] for t in times])

            # Fit linear relationship in log space to determine complexity
            log_sizes = np.log(sizes_array)
            log_times = np.log(times_array)

            # Simple linear regression
            slope = np.cov(log_sizes, log_times)[0, 1] / np.var(log_sizes)

            if self.verbose:
                print(f"\nScaling analysis: O(n^{slope:.2f})")
                if slope < 1.5:
                    print("✅ Good scalability (better than O(n^1.5))")
                elif slope < 2:
                    print("⚠️  Moderate scalability (between O(n^1.5) and O(n^2))")
                else:
                    print("❌ Poor scalability (worse than O(n^2))")

        self.test_results['scalability'] = {
            'measurements': times,
            'complexity_exponent': slope if len(times) >= 3 else None
        }

    def create_visualization_report(self, save_dir: str = None):
        """
        Create visualizations of the specialized test results.

        Good visualization can communicate complex results more effectively
        than tables of numbers. These plots tell the story of REBALANCE's
        capabilities.
        """
        if not self.test_results:
            print("No test results to visualize. Run tests first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('REBALANCE Specialized Evaluation Results', fontsize=16)

        # 1. Extreme Imbalance Handling
        if 'extreme_imbalance' in self.test_results:
            ax = axes[0, 0]
            result = self.test_results['extreme_imbalance']

            scenarios = ['Original', 'After REBALANCE']
            di_values = [result['original_di'], result['final_di']]
            colors = ['red' if di < 0.8 else 'green' for di in di_values]

            bars = ax.bar(scenarios, di_values, color=colors, alpha=0.7)
            ax.axhline(y=0.8, color='black', linestyle='--', label='Fair threshold')
            ax.set_ylabel('Disparate Impact')
            ax.set_title('Extreme Imbalance Test\n(5% high-income women)')
            ax.set_ylim(0, 1)

            for bar, value in zip(bars, di_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom')

        # 2. Intersectional Bias
        if 'intersectional_bias' in self.test_results:
            ax = axes[0, 1]
            result = self.test_results['intersectional_bias']

            groups = list(result['group_rates'].keys())
            rates = list(result['group_rates'].values())

            # Sort by rate for better visualization
            sorted_data = sorted(zip(groups, rates), key=lambda x: x[1])
            groups, rates = zip(*sorted_data)

            bars = ax.barh(groups, rates)
            ax.set_xlabel('Positive Outcome Rate')
            ax.set_title('Intersectional Analysis\n(Gender × Race)')

            # Color bars by gender
            for i, (bar, group) in enumerate(zip(bars, groups)):
                if 'Female' in group:
                    bar.set_color('lightpink')
                else:
                    bar.set_color('lightblue')

        # 3. Small Sample Robustness
        if 'small_sample_robustness' in self.test_results:
            ax = axes[1, 0]
            result = self.test_results['small_sample_robustness']

            sizes = []
            success_rates = []

            for size, data in sorted(result.items()):
                if data['success'] and data['disparate_impact'] is not None:
                    sizes.append(size)
                    success_rates.append(data['disparate_impact'])

            if sizes:
                ax.plot(sizes, success_rates, 'o-', color='blue', markersize=8)
                ax.axhline(y=0.8, color='red', linestyle='--', label='Fair threshold')
                ax.set_xlabel('Sample Size')
                ax.set_ylabel('Achieved Disparate Impact')
                ax.set_title('Small Sample Performance')
                ax.legend()

        # 4. Scalability
        if 'scalability' in self.test_results and self.test_results['scalability']['measurements']:
            ax = axes[1, 1]
            result = self.test_results['scalability']

            sizes, times = zip(*result['measurements'])
            ax.loglog(sizes, times, 'o-', color='green', markersize=8)
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel('Processing Time (seconds)')
            ax.set_title('Scalability Analysis')
            ax.grid(True, alpha=0.3)

            if result['complexity_exponent']:
                ax.text(0.05, 0.95, f"Complexity: O(n^{result['complexity_exponent']:.2f})",
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/specialized_tests.png", dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_dir}/specialized_tests.png")

        plt.show()
