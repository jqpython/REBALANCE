"""
Test Employment-Specific Recommendations

This script tests that the RecommendationAdvisor correctly recommends
the employment-optimized version for employment datasets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.recommendation.advisor import RecommendationAdvisor


def test_employment_recommendations():
    """Test that employment-specific recommendations work correctly."""
    print("TESTING EMPLOYMENT-SPECIFIC RECOMMENDATIONS")
    print("="*60)
    
    # Create sample employment data
    np.random.seed(42)
    employment_data = pd.DataFrame({
        'age': np.random.randint(25, 65, 200),
        'sex': np.random.choice(['Male', 'Female'], 200),
        'education': np.random.choice(['High School', 'Bachelor', 'Master'], 200),
        'occupation': np.random.choice(['Engineer', 'Manager', 'Analyst'], 200),
        'experience_years': np.random.randint(0, 40, 200),
        'workclass': np.random.choice(['Private', 'Government'], 200),
        'hours_per_week': np.random.randint(20, 60, 200)
    })
    
    # Create biased outcomes
    y = np.random.choice([0, 1], 200, p=[0.7, 0.3])
    # Introduce bias against females
    female_mask = employment_data['sex'] == 'Female'
    y[female_mask] = np.random.choice([0, 1], np.sum(female_mask), p=[0.85, 0.15])
    
    print(f"Test Dataset: {len(employment_data)} samples")
    print(f"Features: {list(employment_data.columns)}")
    print(f"Outcome distribution: {np.bincount(y)}")
    
    # Test recommendations
    advisor = RecommendationAdvisor(verbose=False)
    
    # Test quick recommendation
    recommendation = advisor.get_quick_recommendation(employment_data, y, 'sex')
    
    print(f"\n🎯 QUICK RECOMMENDATION:")
    print(f"Technique: {recommendation.technique_name}")
    print(f"Confidence: {recommendation.confidence_score:.3f}")
    print(f"Expected DI Improvement: {recommendation.expected_di_improvement:.1f}%")
    print(f"Reasoning: {recommendation.reasoning[:2]}")  # First 2 reasons
    
    # Test priority-based recommendations
    from src.recommendation.advisor import PerformanceRequirement
    
    priorities = [
        (PerformanceRequirement.FAIRNESS_FIRST, "Fairness First"),
        (PerformanceRequirement.BALANCED, "Balanced"),
        (PerformanceRequirement.ACCURACY_FIRST, "Accuracy First")
    ]
    
    print(f"\n📊 PRIORITY-BASED RECOMMENDATIONS:")
    for priority, name in priorities:
        profile = advisor.analyze_dataset(employment_data, y, 'sex')
        recommendations = advisor.recommend_techniques(profile, priority)
        
        print(f"\n{name}:")
        for i, rec in enumerate(recommendations[:3]):  # Top 3
            print(f"  {i+1}. {rec.technique_name} (confidence: {rec.confidence_score:.3f})")
    
    # Check if employment-optimized version is available
    available_techniques = list(advisor.evidence_base.keys())
    print(f"\n🔧 AVAILABLE TECHNIQUES:")
    for technique in available_techniques:
        print(f"  • {technique}")
    
    if 'REBALANCE Employment-Optimized' in available_techniques:
        print(f"\n✅ Employment-Optimized version is available!")
        
        # Get specific recommendation for employment-optimized
        profile = advisor.analyze_dataset(employment_data, y, 'sex')
        recommendations = advisor.recommend_techniques(profile, PerformanceRequirement.BALANCED)
        
        employment_rec = None
        for rec in recommendations:
            if 'Employment-Optimized' in rec.technique_name:
                employment_rec = rec
                break
        
        if employment_rec:
            print(f"\n📋 EMPLOYMENT-OPTIMIZED DETAILS:")
            print(f"Confidence Score: {employment_rec.confidence_score:.3f}")
            print(f"Expected Parameters:")
            for param, value in employment_rec.parameters.items():
                print(f"  {param}: {value}")
        else:
            print(f"⚠️  Employment-Optimized not in top recommendations")
    else:
        print(f"❌ Employment-Optimized version not found in evidence base")
    
    print(f"\n✅ Employment recommendation testing completed!")


if __name__ == '__main__':
    test_employment_recommendations()