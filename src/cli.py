#!/usr/bin/env python3
"""
Command Line Interface for REBALANCE Toolkit

Provides easy access to bias detection, mitigation, and recommendation
functionality through a simple command-line interface.

Author: REBALANCE Team
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

from .bias_detection.detector import BiasDetector
from .fairness_smote.fair_smote import FairSMOTE
from .recommendation.advisor import RecommendationAdvisor, PerformanceRequirement


def detect_bias_command(args):
    """Handle bias detection command."""
    print("🔍 Running bias detection...")
    
    # Load data
    data = pd.read_csv(args.input)
    if args.target_column not in data.columns:
        print(f"Error: Target column '{args.target_column}' not found in data")
        return 1
    
    if args.protected_attribute not in data.columns:
        print(f"Error: Protected attribute '{args.protected_attribute}' not found in data")
        return 1
    
    X = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    
    # Convert target to binary if needed
    if args.positive_label:
        y = (y == args.positive_label).astype(int)
    
    # Detect bias
    detector = BiasDetector(verbose=True)
    metrics = detector.detect_bias(X, y, args.protected_attribute, 1)
    
    print(f"\n📊 BIAS DETECTION RESULTS")
    print("=" * 50)
    print(f"Disparate Impact: {metrics.disparate_impact:.3f}")
    print(f"Bias Severity: {metrics.get_bias_severity()}")
    print(f"Statistical Parity Difference: {metrics.statistical_parity_difference:.3f}")
    print(f"Is Biased (80% rule): {'Yes' if metrics.is_biased() else 'No'}")
    
    # Save results if requested
    if args.output:
        results = {
            'disparate_impact': metrics.disparate_impact,
            'statistical_parity_difference': metrics.statistical_parity_difference,
            'bias_severity': metrics.get_bias_severity(),
            'is_biased': metrics.is_biased(),
            'group_positive_rates': metrics.group_positive_rates
        }
        
        if args.output.endswith('.json'):
            import json
            with open(args.output, 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = {}
                for k, v in results.items():
                    if isinstance(v, dict):
                        json_results[k] = {str(key): float(val) for key, val in v.items()}
                    elif isinstance(v, (np.integer, np.floating)):
                        json_results[k] = float(v)
                    else:
                        json_results[k] = v
                json.dump(json_results, f, indent=2)
        else:
            # Save as text
            with open(args.output, 'w') as f:
                f.write(f"Bias Detection Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Disparate Impact: {metrics.disparate_impact:.3f}\n")
                f.write(f"Bias Severity: {metrics.get_bias_severity()}\n")
                f.write(f"Statistical Parity Difference: {metrics.statistical_parity_difference:.3f}\n")
                f.write(f"Is Biased (80% rule): {'Yes' if metrics.is_biased() else 'No'}\n")
        
        print(f"\n💾 Results saved to: {args.output}")
    
    return 0


def mitigate_bias_command(args):
    """Handle bias mitigation command."""
    print("⚖️ Running bias mitigation...")
    
    # Load data
    data = pd.read_csv(args.input)
    X = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    
    # Convert target to binary if needed
    if args.positive_label:
        y = (y == args.positive_label).astype(int)
    
    # Apply Fair SMOTE
    fair_smote = FairSMOTE(
        protected_attribute=args.protected_attribute,
        k_neighbors=args.k_neighbors,
        random_state=42
    )
    
    print(f"Original dataset size: {len(X):,}")
    X_resampled, y_resampled = fair_smote.fit_resample(X, y)
    print(f"Resampled dataset size: {len(X_resampled):,}")
    print(f"Synthetic samples generated: {len(X_resampled) - len(X):,}")
    
    # Check improved bias
    detector = BiasDetector(verbose=False)
    improved_metrics = detector.detect_bias(X_resampled, y_resampled, args.protected_attribute, 1)
    
    print(f"\n📈 BIAS MITIGATION RESULTS")
    print("=" * 50)
    print(f"Final Disparate Impact: {improved_metrics.disparate_impact:.3f}")
    print(f"Final Bias Severity: {improved_metrics.get_bias_severity()}")
    
    # Save resampled data
    if args.output:
        resampled_data = X_resampled.copy()
        resampled_data[args.target_column] = y_resampled
        resampled_data.to_csv(args.output, index=False)
        print(f"\n💾 Resampled data saved to: {args.output}")
    
    return 0


def recommend_command(args):
    """Handle recommendation command."""
    print("🎯 Generating technique recommendations...")
    
    # Load data
    data = pd.read_csv(args.input)
    X = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    
    # Convert target to binary if needed
    if args.positive_label:
        y = (y == args.positive_label).astype(int)
    
    # Get performance requirement
    perf_req_map = {
        'fairness': PerformanceRequirement.FAIRNESS_FIRST,
        'balanced': PerformanceRequirement.BALANCED,
        'accuracy': PerformanceRequirement.ACCURACY_FIRST
    }
    performance_req = perf_req_map.get(args.priority, PerformanceRequirement.BALANCED)
    
    # Generate recommendations
    advisor = RecommendationAdvisor(verbose=True)
    recommendation = advisor.get_quick_recommendation(X, y, args.protected_attribute, performance_req)
    
    # Generate full report if requested
    if args.output:
        report = advisor.generate_recommendation_report(X, y, args.protected_attribute)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📋 Full recommendation report saved to: {args.output}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="REBALANCE: Automated Gender Bias Resolution Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect bias in employment data
  rebalance detect --input data.csv --target income --protected-attr sex --positive-label '>50K'
  
  # Mitigate bias using Fair SMOTE
  rebalance mitigate --input data.csv --target income --protected-attr sex --output rebalanced.csv
  
  # Get technique recommendations
  rebalance recommend --input data.csv --target income --protected-attr sex --priority balanced
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect bias command
    detect_parser = subparsers.add_parser('detect', help='Detect bias in dataset')
    detect_parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    detect_parser.add_argument('--target-column', '-t', required=True, help='Target column name')
    detect_parser.add_argument('--protected-attribute', '-p', default='sex', help='Protected attribute column')
    detect_parser.add_argument('--positive-label', help='Positive class label (if not binary)')
    detect_parser.add_argument('--output', '-o', help='Output file for results')
    
    # Mitigate bias command
    mitigate_parser = subparsers.add_parser('mitigate', help='Mitigate bias using Fair SMOTE')
    mitigate_parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    mitigate_parser.add_argument('--target-column', '-t', required=True, help='Target column name')
    mitigate_parser.add_argument('--protected-attribute', '-p', default='sex', help='Protected attribute column')
    mitigate_parser.add_argument('--positive-label', help='Positive class label (if not binary)')
    mitigate_parser.add_argument('--k-neighbors', '-k', type=int, default=5, help='Number of neighbors for SMOTE')
    mitigate_parser.add_argument('--output', '-o', help='Output CSV file for resampled data')
    
    # Recommend technique command
    recommend_parser = subparsers.add_parser('recommend', help='Get technique recommendations')
    recommend_parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    recommend_parser.add_argument('--target-column', '-t', required=True, help='Target column name')
    recommend_parser.add_argument('--protected-attribute', '-p', default='sex', help='Protected attribute column')
    recommend_parser.add_argument('--positive-label', help='Positive class label (if not binary)')
    recommend_parser.add_argument('--priority', choices=['fairness', 'balanced', 'accuracy'], 
                                 default='balanced', help='Performance priority')
    recommend_parser.add_argument('--output', '-o', help='Output file for full report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'detect':
            return detect_bias_command(args)
        elif args.command == 'mitigate':
            return mitigate_bias_command(args)
        elif args.command == 'recommend':
            return recommend_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())