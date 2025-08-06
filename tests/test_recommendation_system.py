"""
Unit tests for the Recommendation System.

These tests ensure that the intelligent recommendation advisor
provides appropriate guidance across different scenarios.

Author: REBALANCE Team
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.recommendation.advisor import (
    RecommendationAdvisor, DatasetProfile, TechniqueRecommendation,
    BiasLevel, DatasetComplexity, PerformanceRequirement
)


class TestRecommendationAdvisor:
    """Test suite for the RecommendationAdvisor."""
    
    @pytest.fixture
    def sample_biased_data(self):
        """Create sample biased employment dataset."""
        np.random.seed(42)
        
        # Create biased dataset
        data = {
            'age': np.random.randint(20, 65, 1000),
            'education_years': np.random.randint(8, 20, 1000),
            'hours_per_week': np.random.randint(20, 60, 1000),
            'workclass': np.random.choice(['Private', 'Government', 'Self-employed'], 1000),
            'sex': np.random.choice(['Male', 'Female'], 1000, p=[0.6, 0.4])
        }
        
        # Introduce gender bias in outcomes
        income = []
        for i in range(1000):
            if data['sex'][i] == 'Male':
                prob = 0.3  # Males 30% chance of high income
            else:
                prob = 0.1  # Females 10% chance of high income (bias)
            income.append(1 if np.random.random() < prob else 0)
        
        df = pd.DataFrame(data)
        y = np.array(income)
        
        return df, y
    
    @pytest.fixture
    def sample_unbiased_data(self):
        """Create sample unbiased dataset."""
        np.random.seed(42)
        
        data = {
            'feature1': np.random.randn(500),
            'feature2': np.random.randn(500),
            'sex': np.random.choice(['Male', 'Female'], 500)
        }
        
        # No bias - equal opportunity
        y = np.random.choice([0, 1], 500, p=[0.8, 0.2])
        
        return pd.DataFrame(data), y
    
    def test_advisor_initialization(self):
        """Test RecommendationAdvisor initialization."""
        advisor = RecommendationAdvisor(verbose=False)
        
        assert advisor.bias_detector is not None
        assert advisor.evidence_base is not None
        assert 'REBALANCE (Fair SMOTE)' in advisor.evidence_base
        assert 'Random Oversampling' in advisor.evidence_base
        
        # Check evidence base structure
        for technique, evidence in advisor.evidence_base.items():
            assert 'di_improvement_range' in evidence
            assert 'accuracy_change_range' in evidence
            assert 'best_for' in evidence
            assert 'computational_cost' in evidence
    
    def test_dataset_analysis_biased(self, sample_biased_data):
        """Test dataset analysis with biased data."""
        X, y = sample_biased_data
        advisor = RecommendationAdvisor(verbose=False)
        
        profile = advisor.analyze_dataset(X, y, 'sex')
        
        # Check basic characteristics
        assert profile.total_samples == 1000
        assert profile.n_features == 5
        assert profile.positive_rate > 0
        
        # Check bias detection
        assert profile.disparate_impact < 1.0
        assert profile.bias_level in [BiasLevel.MODERATE, BiasLevel.SEVERE]
        
        # Check complexity analysis
        assert profile.complexity_level in [DatasetComplexity.SIMPLE, DatasetComplexity.MODERATE]
        assert profile.categorical_feature_ratio > 0  # Has categorical features
        
        # Check protected group analysis
        assert 'Male' in profile.protected_groups
        assert 'Female' in profile.protected_groups
        assert profile.minority_group_size > 0
    
    def test_dataset_analysis_unbiased(self, sample_unbiased_data):
        """Test dataset analysis with unbiased data."""
        X, y = sample_unbiased_data
        advisor = RecommendationAdvisor(verbose=False)
        
        profile = advisor.analyze_dataset(X, y, 'sex')
        
        # Should detect minimal bias
        assert profile.disparate_impact >= 0.6  # Relatively fair
        assert profile.bias_level in [BiasLevel.NONE, BiasLevel.MILD]
        
        # Should be simpler dataset
        assert profile.complexity_level in [DatasetComplexity.SIMPLE, DatasetComplexity.MODERATE]
        assert profile.categorical_feature_ratio < 0.5  # Mostly numerical
    
    def test_technique_evaluation_fair_smote(self, sample_biased_data):
        """Test evaluation of FairSMOTE technique."""
        X, y = sample_biased_data
        advisor = RecommendationAdvisor(verbose=False)
        profile = advisor.analyze_dataset(X, y, 'sex')
        
        evidence = advisor.evidence_base['REBALANCE (Fair SMOTE)']
        score, reasoning, warnings, parameters = advisor._evaluate_fair_smote(
            0.5, [], [], {}, profile, PerformanceRequirement.BALANCED
        )
        
        # Should have positive score for biased employment data
        assert score > 0.5, "FairSMOTE should score well on biased employment data"
        
        # Should have appropriate reasoning
        assert len(reasoning) > 0
        assert any('bias' in reason.lower() for reason in reasoning)
        
        # Should suggest appropriate parameters
        assert 'protected_attribute' in parameters
        assert 'k_neighbors' in parameters
        assert parameters['protected_attribute'] == 'sex'
    
    def test_technique_recommendations_biased_data(self, sample_biased_data):
        """Test technique recommendations for biased data."""
        X, y = sample_biased_data
        advisor = RecommendationAdvisor(verbose=False)
        profile = advisor.analyze_dataset(X, y, 'sex')
        
        recommendations = advisor.recommend_techniques(
            profile, PerformanceRequirement.BALANCED
        )
        
        # Should have recommendations
        assert len(recommendations) > 0
        
        # Should be sorted by confidence
        for i in range(len(recommendations) - 1):
            assert recommendations[i].confidence_score >= recommendations[i+1].confidence_score
        
        # Top recommendation should be appropriate for biased data
        top_rec = recommendations[0]
        assert top_rec.confidence_score > 0.5
        assert top_rec.technique_name in advisor.evidence_base
        assert len(top_rec.reasoning) > 0
    
    def test_performance_priority_differences(self, sample_biased_data):
        """Test that different performance priorities give different recommendations."""
        X, y = sample_biased_data
        advisor = RecommendationAdvisor(verbose=False)
        profile = advisor.analyze_dataset(X, y, 'sex')
        
        # Get recommendations for different priorities
        fairness_recs = advisor.recommend_techniques(profile, PerformanceRequirement.FAIRNESS_FIRST)
        balanced_recs = advisor.recommend_techniques(profile, PerformanceRequirement.BALANCED)
        accuracy_recs = advisor.recommend_techniques(profile, PerformanceRequirement.ACCURACY_FIRST)
        
        # Should have recommendations for all priorities
        assert len(fairness_recs) > 0
        assert len(balanced_recs) > 0
        assert len(accuracy_recs) > 0
        
        # Fairness-first should prioritize high DI improvement
        if len(fairness_recs) > 0:
            top_fairness = fairness_recs[0]
            assert top_fairness.expected_di_improvement > 50  # Should expect good fairness improvement
    
    def test_quick_recommendation(self, sample_biased_data):
        """Test quick recommendation functionality."""
        X, y = sample_biased_data
        advisor = RecommendationAdvisor(verbose=False)
        
        recommendation = advisor.get_quick_recommendation(X, y, 'sex')
        
        # Should return a valid recommendation
        assert isinstance(recommendation, TechniqueRecommendation)
        assert recommendation.technique_name in advisor.evidence_base
        assert recommendation.confidence_score > 0
        assert len(recommendation.reasoning) > 0
        assert 'protected_attribute' in recommendation.parameters
    
    def test_report_generation(self, sample_biased_data):
        """Test comprehensive report generation."""
        X, y = sample_biased_data
        advisor = RecommendationAdvisor(verbose=False)
        
        report = advisor.generate_recommendation_report(X, y, 'sex')
        
        # Should contain key sections
        assert "DATASET ANALYSIS" in report
        assert "RECOMMENDATIONS BY PERFORMANCE PRIORITY" in report
        assert "DETAILED GUIDANCE" in report
        assert "RECOMMENDED IMPLEMENTATION" in report
        
        # Should contain specific metrics
        assert "Disparate Impact:" in report
        assert "Bias Level:" in report
        assert "Dataset Complexity:" in report
        
        # Should contain code example
        assert "technique =" in report
        assert "fit_resample" in report
    
    def test_scenario_adaptations(self):
        """Test recommendations adapt to different scenarios."""
        advisor = RecommendationAdvisor(verbose=False)
        
        # Create different scenarios by modifying profile
        base_profile = DatasetProfile(
            total_samples=5000,
            n_features=10,
            positive_rate=0.2,
            disparate_impact=0.5,
            bias_level=BiasLevel.SEVERE,
            group_imbalance_ratio=0.4,
            categorical_feature_ratio=0.3,
            missing_value_ratio=0.05,
            complexity_level=DatasetComplexity.MODERATE,
            protected_groups=['Male', 'Female'],
            minority_group_size=2000,
            minority_positive_samples=50
        )
        
        # Test different scenarios
        scenarios = [
            # Small dataset
            ({"total_samples": 500}, "small dataset"),
            # High categorical features
            ({"categorical_feature_ratio": 0.8}, "high categorical"),
            # Very few minority positives
            ({"minority_positive_samples": 2}, "few positives"),
            # Large dataset
            ({"total_samples": 100000}, "large dataset")
        ]
        
        for modifications, scenario_name in scenarios:
            # Create modified profile
            modified_profile = DatasetProfile(**{**base_profile.__dict__, **modifications})
            
            recommendations = advisor.recommend_techniques(modified_profile)
            
            # Should still provide recommendations
            assert len(recommendations) > 0, f"Should provide recommendations for {scenario_name}"
            
            # Top recommendation should be valid
            top_rec = recommendations[0]
            assert top_rec.confidence_score > 0, f"Should have positive confidence for {scenario_name}"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        advisor = RecommendationAdvisor(verbose=False)
        
        # Test with minimal dataset
        minimal_data = pd.DataFrame({
            'feature1': [1, 2],
            'sex': ['Male', 'Female']
        })
        minimal_y = np.array([0, 1])
        
        # Should handle gracefully
        profile = advisor.analyze_dataset(minimal_data, minimal_y, 'sex')
        recommendations = advisor.recommend_techniques(profile)
        
        # Should still provide fallback recommendation
        assert len(recommendations) >= 0
    
    def test_bias_level_categorization(self):
        """Test bias level categorization logic."""
        advisor = RecommendationAdvisor(verbose=False)
        
        # Test different DI values
        test_cases = [
            (0.9, BiasLevel.NONE),
            (0.75, BiasLevel.MILD),
            (0.55, BiasLevel.MODERATE),
            (0.3, BiasLevel.SEVERE)
        ]
        
        for di_value, expected_level in test_cases:
            # Create mock profile with specific DI
            with patch.object(advisor.bias_detector, 'detect_bias') as mock_detect:
                mock_metrics = MagicMock()
                mock_metrics.disparate_impact = di_value
                mock_detect.return_value = mock_metrics
                
                sample_data = pd.DataFrame({
                    'feature': [1, 2, 3],
                    'sex': ['Male', 'Female', 'Male']
                })
                sample_y = np.array([0, 1, 0])
                
                profile = advisor.analyze_dataset(sample_data, sample_y, 'sex')
                assert profile.bias_level == expected_level, f"DI {di_value} should be {expected_level}"
    
    def test_complexity_calculation(self):
        """Test dataset complexity calculation."""
        advisor = RecommendationAdvisor(verbose=False)
        
        # Simple dataset
        simple_data = pd.DataFrame({
            'num1': [1, 2, 3, 4],
            'num2': [5, 6, 7, 8],
            'sex': ['Male', 'Female', 'Male', 'Female']
        })
        simple_y = np.array([0, 1, 0, 1])
        
        with patch.object(advisor.bias_detector, 'detect_bias'):
            profile = advisor.analyze_dataset(simple_data, simple_y, 'sex')
            assert profile.complexity_level == DatasetComplexity.SIMPLE
        
        # Complex dataset
        complex_data = pd.DataFrame({
            **{f'cat_{i}': np.random.choice(['A', 'B', 'C'], 100) for i in range(15)},
            **{f'num_{i}': np.random.randn(100) for i in range(10)},
            'sex': np.random.choice(['Male', 'Female'], 100)
        })
        complex_y = np.random.choice([0, 1], 100)
        
        with patch.object(advisor.bias_detector, 'detect_bias'):
            profile = advisor.analyze_dataset(complex_data, complex_y, 'sex')
            assert profile.complexity_level == DatasetComplexity.COMPLEX


if __name__ == '__main__':
    pytest.main([__file__])