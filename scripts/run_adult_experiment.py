#!/usr/bin/env python3
"""
Comprehensive UCI Adult Dataset Validation Script for REBALANCE

This script provides reproducible validation of the REBALANCE toolkit
against the proposal objectives:
- ≥50% disparate impact reduction
- ≤5% accuracy loss
- Proper comparative evaluation

Author: REBALANCE Team
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import REBALANCE components
from src.bias_detection.detector import BiasDetector
from src.fairness_smote.fair_smote import FairSMOTE
from src.rebalance import FairRebalancer
from imblearn.over_sampling import SMOTE, RandomOverSampler

def load_adult_dataset():
    """Load and prepare the UCI Adult dataset."""
    print("Loading UCI Adult dataset...")
    
    # Check if processed data exists
    processed_path = 'data/processed/adult_with_labels.csv'
    if os.path.exists(processed_path):
        data = pd.read_csv(processed_path)
        print(f"Loaded processed dataset: {len(data):,} samples")
    else:
        # Load raw data
        features_path = 'data/raw/adult_features.csv'
        target_path = 'data/raw/adult_target.csv'
        
        if not (os.path.exists(features_path) and os.path.exists(target_path)):
            raise FileNotFoundError("Adult dataset files not found. Please ensure data is downloaded.")
        
        X_raw = pd.read_csv(features_path)
        y_raw = pd.read_csv(target_path)
        
        # Combine and process
        data = pd.concat([X_raw, y_raw], axis=1)
        print(f"Loaded raw dataset: {len(data):,} samples")
    
    # Handle missing values (represented as '?' in UCI Adult dataset)
    print("Cleaning missing values...")
    missing_before = data.isin(['?', ' ?']).sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values marked as '?'")
        
        # Replace '?' with NaN and then forward fill or mode
        data = data.replace(['?', ' ?'], np.nan)
        
        # For categorical columns, fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
                data[col] = data[col].fillna(mode_val)
                print(f"  Filled {col} missing values with: {mode_val}")
        
        # For numerical columns, fill with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                print(f"  Filled {col} missing values with median: {median_val}")
    
    # Prepare features and target
    if 'income' in data.columns:
        target_col = 'income'
        positive_label = '>50K'
    elif 'high_income' in data.columns:
        target_col = 'high_income'
        positive_label = 1
    else:
        raise ValueError("Cannot find target column (income or high_income)")
    
    X = data.drop([col for col in [target_col, 'high_income', 'is_female_high_income'] if col in data.columns], axis=1)
    y = (data[target_col] == positive_label).astype(int)
    
    # Ensure we have the protected attribute
    if 'sex' not in X.columns:
        raise ValueError("Protected attribute 'sex' not found in dataset")
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution: {y.mean():.3f} positive rate")
    print(f"Protected attribute values: {X['sex'].unique()}")
    
    # Encode categorical variables for ML models
    print("Encoding categorical variables...")
    X_encoded = X.copy()
    label_encoders = {}
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    return X_encoded, y

def calculate_bias_metrics(X, y, protected_attr='sex'):
    """Calculate comprehensive bias metrics."""
    detector = BiasDetector(verbose=False)
    
    # Handle both encoded and non-encoded data
    if protected_attr in X.columns:
        # Create a temporary dataframe with decoded sex values for bias detection
        X_temp = X.copy()
        if X[protected_attr].dtype in ['int64', 'int32'] and set(X[protected_attr].unique()) == {0, 1}:
            # Decode if it's been label encoded (0/1 -> Female/Male)
            X_temp[protected_attr] = X[protected_attr].map({0: 'Female', 1: 'Male'})
        
        metrics = detector.detect_bias(X_temp, y, protected_attr, 1)
    else:
        raise ValueError(f"Protected attribute '{protected_attr}' not found in data")
    
    return {
        'disparate_impact': metrics.disparate_impact,
        'statistical_parity_diff': metrics.statistical_parity_difference,
        'group_positive_rates': metrics.group_positive_rates,
        'bias_severity': metrics.get_bias_severity(),
        'is_biased': metrics.is_biased()
    }

def evaluate_method(X_train, y_train, X_test, y_test, method_name, resampler=None):
    """Evaluate a bias mitigation method."""
    print(f"\nEvaluating: {method_name}")
    print("-" * 40)
    
    # Apply resampling if provided
    if resampler is not None:
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        print(f"Original training size: {len(X_train):,}")
        print(f"Resampled training size: {len(X_resampled):,}")
    else:
        X_resampled, y_resampled = X_train, y_train
    
    # Calculate bias metrics on resampled data
    bias_metrics = calculate_bias_metrics(X_resampled, y_resampled)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    model_results = {}
    
    for model_name, model in models.items():
        # Handle numerical scaling for LR
        if model_name == 'Logistic Regression':
            # Identify numerical columns
            numerical_cols = X_resampled.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                scaler = StandardScaler()
                X_scaled = X_resampled.copy()
                X_scaled[numerical_cols] = scaler.fit_transform(X_scaled[numerical_cols])
                
                X_test_scaled = X_test.copy()
                X_test_scaled[numerical_cols] = scaler.transform(X_test_scaled[numerical_cols])
            else:
                X_scaled = X_resampled
                X_test_scaled = X_test
        else:
            X_scaled = X_resampled
            X_test_scaled = X_test
        
        # Train model
        model.fit(X_scaled, y_resampled)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        model_results[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        print(f"{model_name}: Acc={accuracy:.3f}, F1={f1:.3f}")
    
    return {
        'method_name': method_name,
        'bias_metrics': bias_metrics,
        'model_results': model_results,
        'samples_generated': len(X_resampled) - len(X_train) if resampler else 0
    }

def main():
    """Main validation experiment."""
    print("="*70)
    print("REBALANCE UCI ADULT DATASET VALIDATION")
    print("="*70)
    print(f"Experiment timestamp: {datetime.now().isoformat()}")
    
    # Load dataset
    X, y = load_adult_dataset()
    
    # Initial bias analysis
    print(f"\n{'='*50}")
    print("INITIAL BIAS ANALYSIS")
    print("="*50)
    initial_bias = calculate_bias_metrics(X, y)
    print(f"Disparate Impact: {initial_bias['disparate_impact']:.3f}")
    print(f"Bias Severity: {initial_bias['bias_severity']}")
    print(f"Group Positive Rates: {initial_bias['group_positive_rates']}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train):,} samples")
    print(f"Testing: {len(X_test):,} samples")
    
    # Test bias on test set for final comparison
    test_bias = calculate_bias_metrics(X_test, y_test)
    print(f"Test set disparate impact: {test_bias['disparate_impact']:.3f}")
    
    # Methods to evaluate
    # Note: For FairSMOTE, we need to create a DataFrame with the protected attribute decoded
    methods = [
        ("Baseline (No Mitigation)", None),
        ("Random Oversampling", RandomOverSampler(random_state=42)),
        ("Standard SMOTE", SMOTE(random_state=42, k_neighbors=5)),
        # We'll handle FairSMOTE separately due to its need for non-encoded protected attributes
    ]
    
    # Run evaluations
    print(f"\n{'='*50}")
    print("METHOD COMPARISON")
    print("="*50)
    
    results = []
    baseline_accuracy = None
    
    for method_name, resampler in methods:
        try:
            result = evaluate_method(X_train, y_train, X_test, y_test, method_name, resampler)
            results.append(result)
            
            # Store baseline accuracy for comparison
            if method_name == "Baseline (No Mitigation)":
                baseline_accuracy = result['model_results']['Logistic Regression']['accuracy']
                
        except Exception as e:
            print(f"Error evaluating {method_name}: {str(e)}")
            continue
    
    # Handle FairSMOTE separately (needs original categorical data)
    print(f"\nEvaluating: REBALANCE (Fair SMOTE)")
    print("-" * 40)
    try:
        # Load original data for FairSMOTE (without encoding)
        data_path = 'data/processed/adult_with_labels.csv'
        data = pd.read_csv(data_path)
        data = data.replace(['?', ' ?'], np.nan)
        
        # Handle missing values
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
                data[col] = data[col].fillna(mode_val)
        
        # Prepare data
        X_fair = data.drop(['income', 'high_income', 'is_female_high_income'], axis=1, errors='ignore')
        y_fair = (data['income'] == '>50K').astype(int)
        
        X_train_fair, X_test_fair, y_train_fair, y_test_fair = train_test_split(
            X_fair, y_fair, test_size=0.2, random_state=42, stratify=y_fair
        )
        
        # Apply FairSMOTE
        fair_smote = FairSMOTE(protected_attribute='sex', random_state=42, k_neighbors=5)
        X_resampled_fair, y_resampled_fair = fair_smote.fit_resample(X_train_fair, y_train_fair)
        
        print(f"Original training size: {len(X_train_fair):,}")
        print(f"Resampled training size: {len(X_resampled_fair):,}")
        
        # Calculate bias metrics (on original categorical data)
        bias_metrics = calculate_bias_metrics(X_resampled_fair, y_resampled_fair)
        
        # Encode all categorical columns for model training
        X_resampled_encoded = X_resampled_fair.copy()
        X_test_encoded = X_test_fair.copy()
        
        encoders = {}
        for col in X_resampled_fair.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            # Fit on all unique values from both training and test
            all_values = pd.concat([X_resampled_fair[col], X_test_fair[col]]).unique()
            le.fit(all_values)
            encoders[col] = le
            
            X_resampled_encoded[col] = le.transform(X_resampled_fair[col])
            X_test_encoded[col] = le.transform(X_test_fair[col])
        
        # Train model with proper scaling
        model = LogisticRegression(max_iter=1000, random_state=42)
        scaler = StandardScaler()
        
        # Scale all features
        X_scaled = scaler.fit_transform(X_resampled_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        
        model.fit(X_scaled, y_resampled_fair)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_fair, y_pred)
        f1 = f1_score(y_test_fair, y_pred)
        precision = precision_score(y_test_fair, y_pred)
        recall = recall_score(y_test_fair, y_pred)
        
        fair_result = {
            'method_name': 'REBALANCE (Fair SMOTE)',
            'bias_metrics': bias_metrics,
            'model_results': {
                'Logistic Regression': {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            },
            'samples_generated': len(X_resampled_fair) - len(X_train_fair)
        }
        
        results.append(fair_result)
        print(f"Logistic Regression: Acc={accuracy:.3f}, F1={f1:.3f}")
        
    except Exception as e:
        print(f"Error evaluating REBALANCE (Fair SMOTE): {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Generate comparison report
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    validation_results = {
        'experiment_timestamp': datetime.now().isoformat(),
        'initial_disparate_impact': initial_bias['disparate_impact'],
        'baseline_accuracy': baseline_accuracy,
        'methods': {},
        'validation_criteria': {
            'disparate_impact_reduction_target': 0.5,  # 50%
            'accuracy_loss_limit': 0.05  # 5%
        }
    }
    
    for result in results:
        method_name = result['method_name']
        lr_results = result['model_results']['Logistic Regression']
        bias_metrics = result['bias_metrics']
        
        # Calculate improvements
        # For DI: improvement is how much closer we got to 1.0 (perfect fairness)
        initial_di = initial_bias['disparate_impact']
        final_di = bias_metrics['disparate_impact']
        
        # Calculate how much of the gap to perfect fairness (1.0) we closed
        if initial_di < 1.0:
            di_improvement = (final_di - initial_di) / (1.0 - initial_di)
        else:
            di_improvement = 0.0  # Already at or above perfect fairness
        accuracy_change = (lr_results['accuracy'] - baseline_accuracy) / baseline_accuracy if baseline_accuracy else 0
        
        validation_results['methods'][method_name] = {
            'disparate_impact': bias_metrics['disparate_impact'],
            'di_improvement_pct': di_improvement * 100,
            'accuracy': lr_results['accuracy'],
            'accuracy_change_pct': accuracy_change * 100,
            'f1_score': lr_results['f1'],
            'samples_generated': result['samples_generated'],
            'meets_di_target': bool(di_improvement >= 0.5),
            'meets_accuracy_target': bool(abs(accuracy_change) <= 0.05)
        }
        
        print(f"\n{method_name}:")
        print(f"  Disparate Impact: {bias_metrics['disparate_impact']:.3f}")
        print(f"  DI Improvement: {di_improvement*100:+.1f}%")
        print(f"  Accuracy: {lr_results['accuracy']:.3f}")
        print(f"  Accuracy Change: {accuracy_change*100:+.1f}%")
        print(f"  F1 Score: {lr_results['f1']:.3f}")
        print(f"  Meets DI Target: {'YES' if di_improvement >= 0.5 else 'NO'}")
        print(f"  Meets Accuracy Target: {'YES' if abs(accuracy_change) <= 0.05 else 'NO'}")
    
    # Save results
    os.makedirs('results/validation', exist_ok=True)
    
    with open('results/validation/adult_experiment_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Generate summary report
    summary_report = f"""
REBALANCE UCI ADULT DATASET VALIDATION REPORT
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OBJECTIVE VALIDATION:
- Target: >=50% disparate impact reduction
- Target: <=5% accuracy loss

RESULTS:
"""
    
    rebalance_result = validation_results['methods'].get('REBALANCE (Fair SMOTE)', {})
    if rebalance_result:
        summary_report += f"""
REBALANCE Performance:
- Disparate Impact Reduction: {rebalance_result['di_improvement_pct']:.1f}%
- Accuracy Change: {rebalance_result['accuracy_change_pct']:+.1f}%
- Meets DI Target: {'YES' if rebalance_result['meets_di_target'] else 'NO'}
- Meets Accuracy Target: {'YES' if rebalance_result['meets_accuracy_target'] else 'NO'}

CONCLUSION: {'VALIDATION SUCCESSFUL' if rebalance_result['meets_di_target'] and rebalance_result['meets_accuracy_target'] else 'VALIDATION NEEDS IMPROVEMENT'}
"""
    
    with open('results/validation/validation_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print(summary_report)
    print(f"\nDetailed results saved to: results/validation/")
    
    return validation_results

if __name__ == "__main__":
    main()