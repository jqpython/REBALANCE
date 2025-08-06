"""
Unit tests for FairSMOTE implementation.

These tests ensure that the fairness-aware SMOTE algorithm
works correctly across various scenarios and edge cases.

Author: REBALANCE Team
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.fairness_smote.fair_smote import FairSMOTE
from src.bias_detection.detector import BiasDetector


class TestFairSMOTE:
    """Test suite for FairSMOTE algorithm."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample employment-like dataset for testing."""
        np.random.seed(42)
        
        # Create base features
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.8, 0.2],  # Imbalanced classes
            random_state=42
        )
        
        # Convert to DataFrame with meaningful column names
        feature_names = [
            'age', 'education_years', 'hours_per_week', 'experience',
            'skill_score', 'workclass_encoded', 'occupation_encoded',
            'marital_status_encoded', 'relationship_encoded', 'race_encoded'
        ]
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add protected attribute (sex) with bias
        # Create bias: males more likely to have positive outcomes
        sex_prob = np.where(y == 1, 0.3, 0.7)  # When y=1, 30% chance of being female
        sex = np.random.choice(['Male', 'Female'], size=len(y), p=[0.6, 0.4])
        # Introduce bias
        biased_sex = []
        for i, (s, label) in enumerate(zip(sex, y)):
            if label == 1 and np.random.random() < 0.7:  # 70% of positive outcomes are male
                biased_sex.append('Male')
            elif label == 0 and np.random.random() < 0.4:  # 40% of negative outcomes are male
                biased_sex.append('Male')
            else:
                biased_sex.append('Female')
        
        df['sex'] = biased_sex
        
        return df, y
    
    def test_fair_smote_initialization(self):
        """Test FairSMOTE initialization with various parameters."""
        # Test default initialization
        fair_smote = FairSMOTE(protected_attribute='sex')
        assert fair_smote.protected_attribute == 'sex'
        assert fair_smote.k_neighbors == 5
        assert fair_smote.fairness_strategy == 'equal_opportunity'
        
        # Test custom initialization
        fair_smote_custom = FairSMOTE(
            protected_attribute='race',
            k_neighbors=3,
            fairness_strategy='demographic_parity',
            random_state=123
        )
        assert fair_smote_custom.protected_attribute == 'race'
        assert fair_smote_custom.k_neighbors == 3
        assert fair_smote_custom.fairness_strategy == 'demographic_parity'
        assert fair_smote_custom.random_state == 123
    
    def test_fair_smote_basic_functionality(self, sample_data):
        """Test basic FairSMOTE functionality."""
        X, y = sample_data
        
        # Initialize FairSMOTE
        fair_smote = FairSMOTE(protected_attribute='sex', random_state=42)
        
        # Check initial bias
        detector = BiasDetector(verbose=False)
        initial_metrics = detector.detect_bias(X, y, 'sex', 1)
        
        # Apply FairSMOTE
        X_resampled, y_resampled = fair_smote.fit_resample(X, y)
        
        # Check that resampling occurred
        assert len(X_resampled) >= len(X), "Resampled data should be larger or equal"
        assert len(X_resampled) == len(y_resampled), "X and y should have same length"
        assert isinstance(X_resampled, pd.DataFrame), "Should return DataFrame"
        
        # Check that all original columns are preserved
        assert list(X_resampled.columns) == list(X.columns), "Columns should be preserved"
        
        # Check that protected attribute values are valid
        original_sex_values = set(X['sex'].unique())
        resampled_sex_values = set(X_resampled['sex'].unique())
        assert resampled_sex_values.issubset(original_sex_values), "Should not create new sex values"
    
    def test_bias_improvement(self, sample_data):
        """Test that FairSMOTE actually improves bias metrics."""
        X, y = sample_data
        
        detector = BiasDetector(verbose=False)
        
        # Measure initial bias
        initial_metrics = detector.detect_bias(X, y, 'sex', 1)
        initial_di = initial_metrics.disparate_impact
        
        # Apply FairSMOTE
        fair_smote = FairSMOTE(protected_attribute='sex', random_state=42)
        X_resampled, y_resampled = fair_smote.fit_resample(X, y)
        
        # Measure improved bias
        final_metrics = detector.detect_bias(X_resampled, y_resampled, 'sex', 1)
        final_di = final_metrics.disparate_impact
        
        # Check improvement (should be closer to 1.0)
        assert final_di >= initial_di, f"DI should improve: {initial_di:.3f} -> {final_di:.3f}"
        
        # If there was significant bias initially, there should be meaningful improvement
        if initial_di < 0.8:
            improvement = (final_di - initial_di) / (1.0 - initial_di)
            assert improvement > 0.1, f"Should see at least 10% improvement, got {improvement:.3f}"
    
    def test_categorical_data_handling(self):
        """Test FairSMOTE with categorical data."""
        # Create dataset with categorical features
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(20, 65, 500),
            'workclass': np.random.choice(['Private', 'Government', 'Self-employed'], 500),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
            'sex': np.random.choice(['Male', 'Female'], 500),
            'income': np.random.choice([0, 1], 500, p=[0.8, 0.2])
        }
        
        # Introduce bias
        for i in range(len(data['income'])):
            if data['sex'][i] == 'Female' and np.random.random() < 0.7:
                data['income'][i] = 0  # Females less likely to have high income
        
        df = pd.DataFrame(data)
        X = df.drop('income', axis=1)
        y = df['income']
        
        # Apply FairSMOTE
        fair_smote = FairSMOTE(protected_attribute='sex', random_state=42)
        X_resampled, y_resampled = fair_smote.fit_resample(X, y)
        
        # Check that categorical values are preserved
        for col in ['workclass', 'education', 'sex']:
            original_values = set(X[col].unique())
            resampled_values = set(X_resampled[col].unique())
            assert resampled_values.issubset(original_values), f"New values created in {col}"
    
    def test_edge_cases(self):
        """Test FairSMOTE with edge cases."""
        # Test with very small dataset
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'sex': ['Male', 'Female', 'Male', 'Female', 'Male']
        })
        small_y = np.array([0, 1, 0, 0, 1])
        
        fair_smote = FairSMOTE(protected_attribute='sex', k_neighbors=2, random_state=42)
        X_resampled, y_resampled = fair_smote.fit_resample(small_data, small_y)
        
        # Should handle small dataset gracefully
        assert len(X_resampled) >= len(small_data)
        assert len(X_resampled) == len(y_resampled)
    
    def test_no_bias_scenario(self):
        """Test FairSMOTE when there's no bias (should make minimal changes)."""
        np.random.seed(42)
        
        # Create unbiased dataset
        X_unbiased = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'sex': np.random.choice(['Male', 'Female'], 200)
        })
        
        # Create balanced outcomes
        y_unbiased = np.random.choice([0, 1], 200, p=[0.7, 0.3])
        
        # Apply FairSMOTE
        fair_smote = FairSMOTE(protected_attribute='sex', random_state=42)
        X_resampled, y_resampled = fair_smote.fit_resample(X_unbiased, y_unbiased)
        
        # Check that minimal changes were made
        samples_added = len(X_resampled) - len(X_unbiased)
        assert samples_added >= 0, "Should not remove samples"
        
        # With no bias, should add relatively few samples
        if samples_added > 0:
            addition_ratio = samples_added / len(X_unbiased)
            assert addition_ratio < 0.5, f"Should add minimal samples when no bias, got {addition_ratio:.3f}"
    
    def test_parameter_validation(self):
        """Test parameter validation and error handling."""
        # Test invalid protected attribute
        X_invalid = pd.DataFrame({
            'feature1': [1, 2, 3],
            'other_attr': ['A', 'B', 'C']
        })
        y_invalid = np.array([0, 1, 0])
        
        fair_smote = FairSMOTE(protected_attribute='nonexistent')
        
        with pytest.raises(ValueError, match="Protected attribute"):
            fair_smote.fit_resample(X_invalid, y_invalid)
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random state."""
        X, y = sample_data
        
        # Run FairSMOTE twice with same random state
        fair_smote1 = FairSMOTE(protected_attribute='sex', random_state=42)
        X_resampled1, y_resampled1 = fair_smote1.fit_resample(X, y)
        
        fair_smote2 = FairSMOTE(protected_attribute='sex', random_state=42)
        X_resampled2, y_resampled2 = fair_smote2.fit_resample(X, y)
        
        # Results should be identical
        assert len(X_resampled1) == len(X_resampled2), "Length should be identical"
        assert len(y_resampled1) == len(y_resampled2), "Label length should be identical"
        
        # Check that the same samples were generated
        np.testing.assert_array_equal(y_resampled1, y_resampled2, "Labels should be identical")
    
    def test_k_neighbors_parameter(self, sample_data):
        """Test different k_neighbors values."""
        X, y = sample_data
        
        # Test with different k values
        for k in [1, 3, 5, 7]:
            fair_smote = FairSMOTE(protected_attribute='sex', k_neighbors=k, random_state=42)
            X_resampled, y_resampled = fair_smote.fit_resample(X, y)
            
            assert len(X_resampled) >= len(X), f"Should resample with k={k}"
            assert len(X_resampled) == len(y_resampled), f"Lengths should match with k={k}"
    
    def test_sampling_strategy_effects(self, sample_data):
        """Test different sampling strategies."""
        X, y = sample_data
        
        strategies = ['auto', 'minority']
        
        for strategy in strategies:
            fair_smote = FairSMOTE(
                protected_attribute='sex',
                sampling_strategy=strategy,
                random_state=42
            )
            X_resampled, y_resampled = fair_smote.fit_resample(X, y)
            
            assert len(X_resampled) >= len(X), f"Should resample with strategy={strategy}"
            assert len(X_resampled) == len(y_resampled), f"Lengths should match with strategy={strategy}"


if __name__ == '__main__':
    pytest.main([__file__])