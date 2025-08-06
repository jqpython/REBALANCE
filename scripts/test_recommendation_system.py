#!/usr/bin/env python3
"""
Test the REBALANCE Recommendation System

This script demonstrates the intelligent recommendation advisor
that guides practitioners to the best bias mitigation technique
based on their dataset characteristics and requirements.

Author: REBALANCE Team
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.recommendation.advisor import RecommendationAdvisor, PerformanceRequirement

def load_sample_data():
    """Load the Adult dataset for testing recommendations."""
    print("Loading Adult dataset for recommendation testing...")
    
    data_path = 'data/processed/adult_with_labels.csv'
    data = pd.read_csv(data_path).head(5000)  # Use subset for quick testing
    data = data.replace(['?', ' ?'], np.nan)
    
    # Fill missing values
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().any():
            mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
            data[col] = data[col].fillna(mode_val)
    
    X = data.drop(['income', 'high_income', 'is_female_high_income'], axis=1, errors='ignore')
    y = (data['income'] == '>50K').astype(int)
    
    return X, y

def test_recommendation_system():
    """Test the recommendation system with different scenarios."""
    print("="*80)
    print("REBALANCE RECOMMENDATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Load data
    X, y = load_sample_data()
    
    # Initialize advisor
    advisor = RecommendationAdvisor(verbose=True)
    
    print(f"\nDataset Overview:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive Rate: {y.mean():.3f}")
    
    # Test 1: Dataset Analysis
    print(f"\n{'='*60}")
    print("STEP 1: DATASET ANALYSIS")
    print("="*60)
    
    profile = advisor.analyze_dataset(X, y, 'sex')
    
    print(f"Dataset Profile:")
    print(f"  Disparate Impact: {profile.disparate_impact:.3f}")
    print(f"  Bias Level: {profile.bias_level.value.title()}")
    print(f"  Complexity: {profile.complexity_level.value.title()}")
    print(f"  Categorical Features: {profile.categorical_feature_ratio:.1%}")
    print(f"  Minority Positive Samples: {profile.minority_positive_samples}")
    
    # Test 2: Quick Recommendation
    print(f"\n{'='*60}")
    print("STEP 2: QUICK RECOMMENDATION")
    print("="*60)
    
    quick_rec = advisor.get_quick_recommendation(X, y, 'sex', PerformanceRequirement.BALANCED)
    
    # Test 3: Recommendations for Different Priorities
    print(f"\n{'='*60}")
    print("STEP 3: RECOMMENDATIONS BY PRIORITY")
    print("="*60)
    
    priorities = [
        (PerformanceRequirement.FAIRNESS_FIRST, "Fairness First"),
        (PerformanceRequirement.BALANCED, "Balanced Approach"),
        (PerformanceRequirement.ACCURACY_FIRST, "Accuracy First")
    ]
    
    for priority, label in priorities:
        print(f"\n{label}:")
        print("-" * 30)
        recommendations = advisor.recommend_techniques(profile, priority)
        
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. {rec.technique_name}")
            print(f"   Confidence: {rec.confidence_score:.2f}")
            print(f"   Expected DI Improvement: {rec.expected_di_improvement:.1f}%")
            print(f"   Expected Accuracy Change: {rec.expected_accuracy_change:+.1f}%")
            print(f"   Cost: {rec.computational_cost.title()}")
            
            if rec.reasoning:
                print(f"   Key Reasons: {', '.join(rec.reasoning[:2])}")
            
            if rec.warnings and i == 1:  # Show warnings for top recommendation
                print(f"   ⚠️  Warnings: {', '.join(rec.warnings[:1])}")
            print()
    
    # Test 4: Full Report Generation
    print(f"\n{'='*60}")
    print("STEP 4: COMPREHENSIVE RECOMMENDATION REPORT")
    print("="*60)
    
    report = advisor.generate_recommendation_report(X, y, 'sex')
    
    # Save report to file
    os.makedirs('results/recommendations', exist_ok=True)
    with open('results/recommendations/recommendation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📋 Comprehensive report saved to: results/recommendations/recommendation_report.txt")
    print("\nReport preview:")
    print("-" * 40)
    # Show first part of report
    print(report[:1000] + "...")
    
    # Test 5: Different Dataset Scenarios
    print(f"\n{'='*60}")
    print("STEP 5: SCENARIO TESTING")
    print("="*60)
    
    # Simulate different dataset characteristics
    scenarios = [
        ("Small dataset with severe bias", {"total_samples": 1000, "disparate_impact": 0.3}),
        ("Large dataset with mild bias", {"total_samples": 100000, "disparate_impact": 0.75}),
        ("High categorical features", {"categorical_feature_ratio": 0.8}),
        ("Very few minority positives", {"minority_positive_samples": 3})
    ]
    
    for scenario_name, modifications in scenarios:
        print(f"\nScenario: {scenario_name}")
        print("-" * 30)
        
        # Create modified profile
        modified_profile = profile
        for attr, value in modifications.items():
            setattr(modified_profile, attr, value)
        
        # Get recommendation for this scenario
        recommendations = advisor.recommend_techniques(modified_profile, PerformanceRequirement.BALANCED)
        
        if recommendations:
            top_rec = recommendations[0]
            print(f"Recommended: {top_rec.technique_name}")
            print(f"Confidence: {top_rec.confidence_score:.2f}")
            if top_rec.reasoning:
                print(f"Main Reason: {top_rec.reasoning[0]}")
        else:
            print("No suitable recommendations found")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION SYSTEM TEST COMPLETED")
    print("="*80)
    print("✅ Dataset analysis completed")
    print("✅ Quick recommendations provided")
    print("✅ Priority-based recommendations generated")
    print("✅ Comprehensive report created")
    print("✅ Scenario testing completed")
    print("\nThe recommendation system is ready to guide practitioners!")

if __name__ == "__main__":
    test_recommendation_system()