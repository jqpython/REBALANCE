# File: src/simple_rebalance.py

"""
Simplified interface for the REBALANCE toolkit.

This provides the absolute simplest way to use REBALANCE - perfect for
practitioners who just want fair data without complexity.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .rebalance import FairRebalancer, RebalanceResult


def rebalance_dataset(data: pd.DataFrame,
                     target_column: str,
                     gender_column: str = None,
                     save_results: bool = False,
                     output_dir: str = './rebalanced_data') -> Tuple[pd.DataFrame, dict]:
    """
    The simplest possible interface to rebalance a biased dataset.

    This function is designed for maximum ease of use. Just pass your
    data and target column, and it handles everything else.

    Parameters
    ----------
    data : pd.DataFrame
        Your dataset including features, target, and gender information

    target_column : str
        Name of the column you're trying to predict (e.g., 'income', 'hired')

    gender_column : str, optional
        Name of the gender column. If None, will auto-detect.

    save_results : bool, default=False
        Whether to save the rebalanced data and report to files

    output_dir : str, default='./rebalanced_data'
        Directory to save results if save_results=True

    Returns
    -------
    rebalanced_data : pd.DataFrame
        Your data with bias reduced through fairness-aware synthetic samples

    summary : dict
        Summary of what was done and the improvements achieved

    Example
    -------
    >>> # Load your biased data
    >>> data = pd.read_csv('biased_hiring_data.csv')
    >>>
    >>> # Fix the bias
    >>> fair_data, report = rebalance_dataset(data, target_column='hired')
    >>>
    >>> # Use the fair data for model training
    >>> X = fair_data.drop('hired', axis=1)
    >>> y = fair_data['hired']
    >>> model.fit(X, y)  # Your model now learns from fair data!
    """
    print("\n🚀 REBALANCE - Automated Bias Mitigation")
    print("="*50)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Create rebalancer
    rebalancer = FairRebalancer(
        protected_attribute=gender_column,
        verbose=True
    )

    # Apply rebalancing
    result = rebalancer.fit_transform(X, y, target_column_name=target_column)

    # Combine rebalanced features and target
    rebalanced_data = result.X_rebalanced.copy()
    rebalanced_data[target_column] = result.y_rebalanced

    # Create summary
    summary = {
        'original_size': result.original_size,
        'rebalanced_size': len(rebalanced_data),
        'synthetic_samples_added': len(rebalanced_data) - result.original_size,
        'original_bias': result.original_bias_metrics.disparate_impact,
        'final_bias': result.final_bias_metrics.disparate_impact,
        'bias_improvement': result.improvement_summary['disparate_impact_change'],
        'is_fair': result.improvement_summary['bias_reduced'],
        'recommendation': result.recommendations
    }

    # Save results if requested
    if save_results:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save rebalanced data
        rebalanced_data.to_csv(f"{output_dir}/rebalanced_data.csv", index=False)

        # Save detailed report
        with open(f"{output_dir}/rebalance_report.txt", 'w') as f:
            f.write(result.get_summary())

        # Save summary as JSON
        import json
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved to {output_dir}/")

    return rebalanced_data, summary


# Convenience function for command-line usage
def main():
    """
    Command-line interface for REBALANCE.
    Makes it easy to use from terminal or scripts.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="REBALANCE - Automated Gender Bias Resolution Toolkit"
    )
    parser.add_argument('input_file', help='Path to biased dataset (CSV)')
    parser.add_argument('target_column', help='Name of target column')
    parser.add_argument('--gender-column', help='Name of gender column (auto-detect if not specified)')
    parser.add_argument('--output-dir', default='./rebalanced_data', help='Output directory')
    parser.add_argument('--no-save', action='store_true', help="Don't save results to files")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = pd.read_csv(args.input_file)

    # Apply rebalancing
    rebalanced_data, summary = rebalance_dataset(
        data=data,
        target_column=args.target_column,
        gender_column=args.gender_column,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )

    print("\n✅ Rebalancing complete!")
    print(f"   Original bias: {summary['original_bias']:.3f}")
    print(f"   Final bias: {summary['final_bias']:.3f}")
    print(f"   Improvement: {summary['bias_improvement']:.1f}%")


if __name__ == "__main__":
    main()
