#!/usr/bin/env python3
"""
Simple test for FairSMOTE to debug the categorical encoding issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.fairness_smote.fair_smote import FairSMOTE
from src.bias_detection.detector import BiasDetector

def test_fair_smote():
    """Test FairSMOTE with a small sample of categorical data."""
    
    # Load a small sample
    data_path = 'data/processed/adult_with_labels.csv'
    data = pd.read_csv(data_path).head(1000)  # Just first 1000 rows
    
    # Clean missing values
    data = data.replace(['?', ' ?'], np.nan)
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if data[col].isnull().any():
            mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
            data[col] = data[col].fillna(mode_val)
    
    # Prepare data
    X = data.drop(['income', 'high_income', 'is_female_high_income'], axis=1, errors='ignore')
    y = (data['income'] == '>50K').astype(int)
    
    print("Original data:")
    print(f"Shape: {X.shape}")
    print(f"Columns: {list(X.columns)}")
    print(f"Sex values: {X['sex'].unique()}")
    print(f"Categorical columns: {list(X.select_dtypes(include=['object']).columns)}")
    
    # Check initial bias
    detector = BiasDetector(verbose=False)
    initial_metrics = detector.detect_bias(X, y, 'sex', 1)
    print(f"\nInitial Disparate Impact: {initial_metrics.disparate_impact:.3f}")
    
    # Apply FairSMOTE
    print("\nApplying FairSMOTE...")
    fair_smote = FairSMOTE(protected_attribute='sex', random_state=42, k_neighbors=3)
    
    try:
        X_resampled, y_resampled = fair_smote.fit_resample(X, y)
        print("✓ FairSMOTE completed successfully!")
        print(f"Resampled shape: {X_resampled.shape}")
        print(f"Original size: {len(X)}, Resampled size: {len(X_resampled)}")
        
        # Check final bias
        final_metrics = detector.detect_bias(X_resampled, y_resampled, 'sex', 1)
        print(f"Final Disparate Impact: {final_metrics.disparate_impact:.3f}")
        print(f"Improvement: {(initial_metrics.disparate_impact - final_metrics.disparate_impact) / initial_metrics.disparate_impact * 100:.1f}%")
        
    except Exception as e:
        print(f"✗ FairSMOTE failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fair_smote()