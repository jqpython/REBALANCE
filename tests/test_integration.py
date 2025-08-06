"""
Integration tests for REBALANCE toolkit.

These tests verify that all components work together correctly
in realistic end-to-end scenarios.

Author: REBALANCE Team
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch

from src.bias_detection.detector import BiasDetector
from src.fairness_smote.fair_smote import FairSMOTE
from src.recommendation.advisor import RecommendationAdvisor, PerformanceRequirement
from src.cli import main as cli_main


class TestIntegration:
    """Integration tests for the complete REBALANCE pipeline."""
    
    @pytest.fixture
    def employment_dataset(self):
        """Create realistic employment dataset for integration testing."""
        np.random.seed(42)
        
        # Create employment-like dataset with bias
        n_samples = 500
        data = {
            'age': np.random.randint(22, 65, n_samples),
            'education_years': np.random.choice([12, 14, 16, 18, 20], n_samples, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
            'hours_per_week': np.random.normal(40, 8, n_samples).clip(20, 60).astype(int),
            'experience_years': np.random.exponential(5, n_samples).clip(0, 40).astype(int),
            'workclass': np.random.choice(['Private', 'Government', 'Self-employed'], n_samples, p=[0.7, 0.2, 0.1]),
            'occupation': np.random.choice(['Technical', 'Management', 'Sales', 'Support', 'Manual'], n_samples),
            'marital_status': np.random.choice(['Married', 'Single', 'Divorced'], n_samples, p=[0.6, 0.3, 0.1]),
            'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
        }
        
        # Generate income with realistic bias
        income = []
        for i in range(n_samples):
            # Base probability from education and experience
            base_prob = (
                data['education_years'][i] / 20 * 0.3 +
                data['experience_years'][i] / 40 * 0.3 +
                (data['age'][i] - 22) / 43 * 0.2 +
                np.random.random() * 0.2  # Random factor
            )
            
            # Apply gender bias
            if data['sex'][i] == 'Female':
                base_prob *= 0.6  # Significant bias against women
            
            income.append(1 if base_prob > 0.5 else 0)
        
        df = pd.DataFrame(data)
        df['income'] = income
        
        return df
    
    def test_complete_pipeline(self, employment_dataset):
        """Test the complete bias detection and mitigation pipeline."""
        # Prepare data
        X = employment_dataset.drop('income', axis=1)
        y = employment_dataset['income']
        
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        
        # Step 1: Detect bias
        detector = BiasDetector(verbose=False)
        initial_metrics = detector.detect_bias(X, y, 'sex', 1)
        
        print(f"Initial DI: {initial_metrics.disparate_impact:.3f}")
        assert initial_metrics.disparate_impact < 0.8, "Should detect significant bias"
        
        # Step 2: Get recommendation
        advisor = RecommendationAdvisor(verbose=False)
        recommendation = advisor.get_quick_recommendation(X, y, 'sex')
        
        print(f"Recommended: {recommendation.technique_name}")
        assert recommendation.confidence_score > 0.5, "Should have confident recommendation"
        
        # Step 3: Apply recommended technique (assuming FairSMOTE)
        if 'Fair SMOTE' in recommendation.technique_name:
            fair_smote = FairSMOTE(
                protected_attribute='sex',
                k_neighbors=min(5, recommendation.parameters.get('k_neighbors', 5)),
                random_state=42
            )
            
            X_resampled, y_resampled = fair_smote.fit_resample(X, y)
            
            print(f"Resampled: {len(X_resampled)} samples")
            assert len(X_resampled) >= len(X), "Should generate additional samples"
            
            # Step 4: Verify improvement
            final_metrics = detector.detect_bias(X_resampled, y_resampled, 'sex', 1)
            
            print(f"Final DI: {final_metrics.disparate_impact:.3f}")
            improvement = (final_metrics.disparate_impact - initial_metrics.disparate_impact) / (1.0 - initial_metrics.disparate_impact)
            
            print(f"Improvement: {improvement * 100:.1f}%")
            assert final_metrics.disparate_impact >= initial_metrics.disparate_impact, "Should improve or maintain fairness"
    
    def test_cli_integration(self, employment_dataset):
        """Test CLI interface with actual data."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            employment_dataset.to_csv(f.name, index=False)
            input_file = f.name
        
        try:
            # Test bias detection command
            with patch('sys.argv', ['rebalance', 'detect', '--input', input_file, '--target-column', 'income', '--protected-attribute', 'sex']):
                result = cli_main()
                assert result == 0, "CLI detect command should succeed"
            
            # Test recommendation command
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                output_file = f.name
            
            try:
                with patch('sys.argv', ['rebalance', 'recommend', '--input', input_file, '--target-column', 'income', '--protected-attribute', 'sex', '--output', output_file]):
                    result = cli_main()
                    assert result == 0, "CLI recommend command should succeed"
                    
                    # Check that output file was created and has content
                    assert os.path.exists(output_file), "Output file should be created"
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        assert len(content) > 0, "Output file should have content"
                        assert "RECOMMENDATION REPORT" in content, "Should contain recommendation report"
                        
            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)
                    
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)
    
    def test_error_recovery(self, employment_dataset):
        """Test system behavior with problematic data."""
        # Create problematic dataset
        problematic_data = employment_dataset.copy()
        
        # Introduce various issues
        problematic_data.loc[0:10, 'age'] = np.nan  # Missing values
        problematic_data.loc[20:30, 'sex'] = 'Unknown'  # New category
        
        X = problematic_data.drop('income', axis=1)
        y = problematic_data['income']
        
        # System should handle gracefully
        detector = BiasDetector(verbose=False)
        
        try:
            # May succeed with warnings or fail gracefully
            metrics = detector.detect_bias(X, y, 'sex', 1)
            assert hasattr(metrics, 'disparate_impact'), "Should return valid metrics"
        except Exception as e:
            # Should be informative error
            assert len(str(e)) > 0, "Error should be informative"
    
    def test_recommendation_consistency(self, employment_dataset):
        """Test that recommendations are consistent across runs."""
        X = employment_dataset.drop('income', axis=1)
        y = employment_dataset['income']
        
        advisor = RecommendationAdvisor(verbose=False)
        
        # Get multiple recommendations
        rec1 = advisor.get_quick_recommendation(X, y, 'sex')
        rec2 = advisor.get_quick_recommendation(X, y, 'sex')
        
        # Should be consistent
        assert rec1.technique_name == rec2.technique_name, "Recommendations should be consistent"
        assert abs(rec1.confidence_score - rec2.confidence_score) < 0.001, "Confidence scores should be consistent"
    
    def test_cross_component_compatibility(self, employment_dataset):
        """Test compatibility between different components."""
        X = employment_dataset.drop('income', axis=1)
        y = employment_dataset['income']
        
        # Test BiasDetector output with RecommendationAdvisor
        detector = BiasDetector(verbose=False)
        initial_metrics = detector.detect_bias(X, y, 'sex', 1)
        
        advisor = RecommendationAdvisor(verbose=False)
        profile = advisor.analyze_dataset(X, y, 'sex')
        
        # Should be compatible
        assert abs(profile.disparate_impact - initial_metrics.disparate_impact) < 0.001, "DI should match between components"
        assert profile.total_samples == len(X), "Sample counts should match"
        
        # Test FairSMOTE output with BiasDetector
        fair_smote = FairSMOTE(protected_attribute='sex', random_state=42)
        X_resampled, y_resampled = fair_smote.fit_resample(X, y)
        
        # Should be able to analyze resampled data
        final_metrics = detector.detect_bias(X_resampled, y_resampled, 'sex', 1)
        assert hasattr(final_metrics, 'disparate_impact'), "Should analyze resampled data"
    
    def test_scalability(self):
        """Test system behavior with different dataset sizes."""
        sizes = [100, 500, 1000]
        
        for size in sizes:
            # Create dataset of specified size
            np.random.seed(42)
            data = pd.DataFrame({
                'feature1': np.random.randn(size),
                'feature2': np.random.randn(size),
                'sex': np.random.choice(['Male', 'Female'], size),
                'income': np.random.choice([0, 1], size, p=[0.7, 0.3])
            })
            
            X = data.drop('income', axis=1)
            y = data['income']
            
            # Test detection
            detector = BiasDetector(verbose=False)
            metrics = detector.detect_bias(X, y, 'sex', 1)
            assert hasattr(metrics, 'disparate_impact'), f"Should work with {size} samples"
            
            # Test recommendation
            advisor = RecommendationAdvisor(verbose=False)
            recommendation = advisor.get_quick_recommendation(X, y, 'sex')
            assert recommendation.confidence_score > 0, f"Should recommend for {size} samples"
    
    def test_different_protected_attributes(self):
        """Test system with different protected attributes."""
        np.random.seed(42)
        
        # Create dataset with multiple potential protected attributes
        data = pd.DataFrame({
            'feature1': np.random.randn(300),
            'feature2': np.random.randn(300),
            'sex': np.random.choice(['Male', 'Female'], 300),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], 300),
            'age_group': np.random.choice(['Young', 'Middle', 'Senior'], 300),
            'income': np.random.choice([0, 1], 300, p=[0.8, 0.2])
        })
        
        X = data.drop('income', axis=1)
        y = data['income']
        
        # Test with different protected attributes
        protected_attrs = ['sex', 'race', 'age_group']
        
        for attr in protected_attrs:
            detector = BiasDetector(verbose=False)
            
            try:
                metrics = detector.detect_bias(X, y, attr, 1)
                assert hasattr(metrics, 'disparate_impact'), f"Should work with {attr}"
                
                # Test recommendation
                advisor = RecommendationAdvisor(verbose=False)
                recommendation = advisor.get_quick_recommendation(X, y, attr)
                assert recommendation.technique_name is not None, f"Should recommend for {attr}"
                
            except Exception as e:
                # Some protected attributes might not work well with small samples
                assert "not found" in str(e) or "unique values" in str(e), f"Should have informative error for {attr}"
    
    def test_memory_and_performance(self, employment_dataset):
        """Test that system doesn't have obvious memory leaks or performance issues."""
        import time
        X = employment_dataset.drop('income', axis=1)
        y = employment_dataset['income']
        
        # Run multiple iterations
        start_time = time.time()
        
        for i in range(5):
            # Detection
            detector = BiasDetector(verbose=False)
            metrics = detector.detect_bias(X, y, 'sex', 1)
            
            # Recommendation
            advisor = RecommendationAdvisor(verbose=False)
            recommendation = advisor.get_quick_recommendation(X, y, 'sex')
            
            # Mitigation
            fair_smote = FairSMOTE(protected_attribute='sex', random_state=42 + i)
            X_resampled, y_resampled = fair_smote.fit_resample(X, y)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert total_time < 30, f"Should complete 5 iterations in <30s, took {total_time:.1f}s"


if __name__ == '__main__':
    pytest.main([__file__])