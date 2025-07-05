# File: tests/test_bias_detection.py
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.bias_detection.detector import BiasDetector, BiasMetrics

class TestBiasDetector:
    """Test suite for the BiasDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a sample dataset with known bias
        np.random.seed(42)
        n_samples = 1000

        # Create biased data
        # Men: 70% positive rate
        # Women: 30% positive rate
        # This gives DI = 0.3/0.7 ≈ 0.43

        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])

        # Create biased outcomes
        outcome = []
        for g in gender:
            if g == 'Male':
                outcome.append(np.random.choice([0, 1], p=[0.3, 0.7]))
            else:
                outcome.append(np.random.choice([0, 1], p=[0.7, 0.3]))

        self.X = pd.DataFrame({
            'gender': gender,
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        self.y = np.array(outcome)

        self.detector = BiasDetector(verbose=False)

    def test_detect_bias_basic(self):
        """Test basic bias detection functionality."""
        metrics = self.detector.detect_bias(self.X, self.y, 'gender', 1)

        # Check that metrics are calculated
        assert isinstance(metrics, BiasMetrics)
        assert 0 <= metrics.disparate_impact <= 1
        assert metrics.is_biased()  # Changed from 'is True'

    def test_disparate_impact_calculation(self):
        """Test that disparate impact is calculated correctly."""
        metrics = self.detector.detect_bias(self.X, self.y, 'gender', 1)

        # Manually calculate expected DI
        male_positive_rate = self.y[self.X['gender'] == 'Male'].mean()
        female_positive_rate = self.y[self.X['gender'] == 'Female'].mean()
        expected_di = female_positive_rate / male_positive_rate

        # Check within reasonable tolerance
        assert abs(metrics.disparate_impact - expected_di) < 0.05

    def test_no_bias_case(self):
        """Test detection when there's no bias."""
        # Create unbiased data
        y_fair = np.random.choice([0, 1], len(self.X), p=[0.5, 0.5])

        metrics = self.detector.detect_bias(self.X, y_fair, 'gender', 1)

        # Disparate impact should be close to 1
        assert 0.8 <= metrics.disparate_impact <= 1.2
        assert not metrics.is_biased()  # Changed from 'is False'

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with missing protected attribute
        with pytest.raises(ValueError):
            self.detector.detect_bias(self.X, self.y, 'nonexistent_column', 1)

        # Test with mismatched lengths
        with pytest.raises(ValueError):
            self.detector.detect_bias(self.X, self.y[:-10], 'gender', 1)

    def test_severity_classification(self):
        """Test bias severity classification."""
        # Create datasets with different bias levels
        bias_levels = [0.9, 0.7, 0.5, 0.3]
        expected_severities = [
            "No significant bias detected",
            "Moderate bias detected",
            "Significant bias detected",
            "Severe bias detected"
        ]

        for di, expected in zip(bias_levels, expected_severities):
            metrics = BiasMetrics(
                disparate_impact=di,
                statistical_parity_difference=0,
                demographic_parity_ratio=di,
                equal_opportunity_difference=0,
                group_sizes={},
                group_positive_rates={}
            )
            assert metrics.get_bias_severity() == expected

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
