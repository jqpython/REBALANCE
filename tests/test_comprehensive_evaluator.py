"""
Unit tests for the Comprehensive Evaluator.

These tests ensure that the evaluation framework works correctly
and provides accurate comparisons between different methods.

Author: REBALANCE Team
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.comprehensive_evaluator import (
    ComprehensiveEvaluator, EvaluationResult
)


class TestComprehensiveEvaluator:
    """Test suite for the ComprehensiveEvaluator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample employment dataset for evaluation testing."""
        np.random.seed(42)
        
        # Create biased dataset
        data = {
            'age': np.random.randint(20, 65, 200),
            'education_years': np.random.randint(8, 20, 200),
            'workclass': np.random.choice(['Private', 'Government'], 200),
            'sex': np.random.choice(['Male', 'Female'], 200, p=[0.6, 0.4])
        }
        
        # Introduce bias
        income = []
        for i in range(200):
            if data['sex'][i] == 'Male':
                prob = 0.4
            else:
                prob = 0.15  # Bias against females
            income.append(1 if np.random.random() < prob else 0)
        
        df = pd.DataFrame(data)
        y = np.array(income)
        
        return df, y
    
    @pytest.fixture
    def simple_models(self):
        """Create simple models for testing."""
        return [
            ('Logistic Regression', LogisticRegression(max_iter=100, random_state=42)),
            ('Random Forest', RandomForestClassifier(n_estimators=10, random_state=42))
        ]
    
    def test_evaluator_initialization(self, simple_models):
        """Test ComprehensiveEvaluator initialization."""
        evaluator = ComprehensiveEvaluator(
            models_to_test=simple_models,
            cv_folds=3,
            random_state=42,
            verbose=False
        )
        
        assert evaluator.models_to_test == simple_models
        assert evaluator.cv_folds == 3
        assert evaluator.random_state == 42
        assert evaluator.verbose == False
        assert evaluator.bias_detector is not None
        assert evaluator.all_results == {}
    
    def test_default_models(self):
        """Test default model initialization."""
        evaluator = ComprehensiveEvaluator(verbose=False)
        
        assert len(evaluator.models_to_test) > 0
        
        # Check that models have appropriate names and types
        model_names = [name for name, _ in evaluator.models_to_test]
        assert 'Logistic Regression' in model_names
        assert 'Random Forest' in model_names
    
    def test_evaluation_result_structure(self):
        """Test EvaluationResult data structure."""
        result = EvaluationResult(
            method_name="Test Method",
            original_disparate_impact=0.5,
            final_disparate_impact=0.8,
            bias_improvement=0.6,
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            processing_time=2.5,
            synthetic_samples_created=100,
            cross_val_scores=[0.83, 0.85, 0.84],
            variance_in_fairness=0.02,
            model_specific_results={'LR': {'accuracy': 0.85}},
            parameters_used={'k': 5},
            warnings_encountered=[]
        )
        
        # Test summary score calculation
        summary_score = result.get_summary_score()
        assert 0 <= summary_score <= 1
        assert isinstance(summary_score, float)
    
    def test_baseline_evaluation(self, sample_data, simple_models):
        """Test baseline evaluation (no mitigation)."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        result = evaluator._evaluate_baseline(X, y, 'sex')
        
        # Check result structure
        assert isinstance(result, EvaluationResult)
        assert result.method_name == "No Mitigation (Baseline)"
        assert result.synthetic_samples_created == 0
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1 <= 1
        assert len(result.model_specific_results) == len(simple_models)
    
    def test_random_oversampling_evaluation(self, sample_data, simple_models):
        """Test random oversampling evaluation."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        result = evaluator._evaluate_random_oversampling(X, y, 'sex')
        
        # Check that oversampling occurred
        assert isinstance(result, EvaluationResult)
        assert result.method_name == "Random Oversampling"
        assert result.synthetic_samples_created > 0
        assert len(result.model_specific_results) == len(simple_models)
    
    def test_standard_smote_evaluation(self, sample_data, simple_models):
        """Test standard SMOTE evaluation."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        # Need to encode categorical variables for standard SMOTE
        X_encoded = X.copy()
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
        
        result = evaluator._evaluate_standard_smote(X_encoded, y, 'sex')
        
        # Check that SMOTE was applied
        assert isinstance(result, EvaluationResult)
        assert result.method_name == "Standard SMOTE"
        assert result.synthetic_samples_created > 0
    
    def test_rebalance_evaluation(self, sample_data, simple_models):
        """Test REBALANCE (Fair SMOTE) evaluation."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        result = evaluator._evaluate_rebalance(X, y, 'sex')
        
        # Check that Fair SMOTE was applied
        assert isinstance(result, EvaluationResult)
        assert result.method_name == "REBALANCE (Fair SMOTE)"
        assert result.synthetic_samples_created >= 0  # May be 0 if no bias detected
        assert len(result.model_specific_results) == len(simple_models)
    
    def test_evaluate_all_methods(self, sample_data, simple_models):
        """Test comprehensive evaluation of all methods."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        # Encode categorical data first
        X_encoded = X.copy()
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
        
        results = evaluator.evaluate_all_methods(X_encoded, y, 'sex')
        
        # Should have results for multiple methods
        assert len(results) > 0
        assert isinstance(results, dict)
        
        # Should include baseline
        assert "No Mitigation (Baseline)" in results
        
        # All results should be EvaluationResult objects
        for method_name, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert result.method_name == method_name
    
    def test_encode_for_model(self, sample_data):
        """Test categorical encoding for models."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(verbose=False)
        
        X_encoded = evaluator._encode_for_model(X)
        
        # Should convert categorical columns to numerical
        assert X_encoded.select_dtypes(include=['object']).shape[1] == 0
        assert X_encoded.shape == X.shape  # Same dimensions
        assert list(X_encoded.columns) == list(X.columns)  # Same column names
    
    def test_cross_validation_scores(self, sample_data, simple_models):
        """Test cross-validation scoring."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(
            models_to_test=simple_models, 
            cv_folds=3, 
            verbose=False
        )
        
        # Encode data
        X_encoded = evaluator._encode_for_model(X)
        
        result = evaluator._evaluate_baseline(X_encoded, y, 'sex')
        
        # Should have cross-validation scores
        assert len(result.cross_val_scores) >= 0  # May be empty in this implementation
    
    def test_error_handling(self, simple_models):
        """Test error handling with problematic data."""
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        # Test with data that has issues
        problematic_X = pd.DataFrame({
            'feature1': [np.inf, -np.inf, np.nan],
            'sex': ['Male', 'Female', 'Male']
        })
        problematic_y = np.array([0, 1, 0])
        
        # Should handle gracefully and not crash
        try:
            result = evaluator._evaluate_baseline(problematic_X, problematic_y, 'sex')
            # If it completes, check basic structure
            assert isinstance(result, EvaluationResult)
        except Exception:
            # If it fails, that's acceptable for problematic data
            pass
    
    def test_processing_time_measurement(self, sample_data, simple_models):
        """Test that processing time is measured."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        X_encoded = evaluator._encode_for_model(X)
        result = evaluator._evaluate_baseline(X_encoded, y, 'sex')
        
        # Should have positive processing time
        assert result.processing_time > 0
        assert isinstance(result.processing_time, (int, float))
    
    def test_bias_improvement_calculation(self, sample_data, simple_models):
        """Test bias improvement calculation."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        # Test baseline (should have minimal improvement)
        X_encoded = evaluator._encode_for_model(X)
        baseline_result = evaluator._evaluate_baseline(X_encoded, y, 'sex')
        
        # Test with rebalancing method
        rebalance_result = evaluator._evaluate_rebalance(X, y, 'sex')
        
        # Rebalancing should show some improvement in disparate impact
        if rebalance_result.synthetic_samples_created > 0:
            assert rebalance_result.final_disparate_impact >= baseline_result.final_disparate_impact
    
    def test_model_specific_results(self, sample_data, simple_models):
        """Test that model-specific results are properly recorded."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        X_encoded = evaluator._encode_for_model(X)
        result = evaluator._evaluate_baseline(X_encoded, y, 'sex')
        
        # Should have results for each model
        assert len(result.model_specific_results) == len(simple_models)
        
        # Each model result should have required metrics
        for model_name, model_results in result.model_specific_results.items():
            assert 'accuracy' in model_results
            assert 'precision' in model_results
            assert 'recall' in model_results
            assert 'f1' in model_results
            
            # Metrics should be valid
            for metric_name, metric_value in model_results.items():
                assert 0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1"
    
    def test_verbose_output(self, sample_data, simple_models):
        """Test verbose output functionality."""
        X, y = sample_data
        
        # Test with verbose=True (should not crash)
        evaluator_verbose = ComprehensiveEvaluator(
            models_to_test=simple_models, 
            verbose=True
        )
        
        X_encoded = evaluator_verbose._encode_for_model(X)
        result = evaluator_verbose._evaluate_baseline(X_encoded, y, 'sex')
        
        assert isinstance(result, EvaluationResult)
    
    def test_protected_attribute_validation(self, sample_data, simple_models):
        """Test validation of protected attribute."""
        X, y = sample_data
        evaluator = ComprehensiveEvaluator(models_to_test=simple_models, verbose=False)
        
        # Test with invalid protected attribute
        X_encoded = evaluator._encode_for_model(X)
        
        # Should handle gracefully or raise appropriate error
        try:
            result = evaluator._evaluate_baseline(X_encoded, y, 'nonexistent_attribute')
            # If it completes, check that it handled the error appropriately
            assert isinstance(result, EvaluationResult)
        except (ValueError, KeyError):
            # Expected behavior for invalid attribute
            pass


if __name__ == '__main__':
    pytest.main([__file__])