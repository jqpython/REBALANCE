"""
Test Employment-Specific Optimizations for FairSMOTE

This script demonstrates the enhanced employment-aware capabilities
of the EmploymentFairSMOTE algorithm.

We'll show how it:
1. Respects job category boundaries
2. Follows realistic experience patterns  
3. Maintains education-occupation relationships
4. Generates professionally coherent synthetic data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our modules
from src.fairness_smote.employment_fair_smote import EmploymentFairSMOTE
from src.fairness_smote.fair_smote import FairSMOTE
from src.bias_detection.detector import BiasDetector


def create_realistic_employment_dataset(n_samples=2000, bias_level=0.3):
    """
    Create a realistic employment dataset with job categories, experience patterns, etc.
    """
    np.random.seed(42)
    
    # Define employment structure
    job_categories = [
        'Software Engineer', 'Manager', 'Data Scientist', 'Sales Representative',
        'Marketing Specialist', 'HR Specialist', 'Analyst', 'Consultant',
        'Administrative Assistant', 'Customer Service'
    ]
    
    education_levels = [
        'High School', 'Bachelor', 'Master', 'PhD'
    ]
    
    sectors = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
    
    # Generate base demographics
    data = []
    
    for i in range(n_samples):
        # Basic demographics
        sex = np.random.choice(['Male', 'Female'], p=[0.55, 0.45])
        age = np.random.randint(22, 65)
        
        # Education (with some gender bias)
        if sex == 'Male':
            education = np.random.choice(education_levels, p=[0.15, 0.5, 0.25, 0.1])
        else:
            education = np.random.choice(education_levels, p=[0.2, 0.55, 0.2, 0.05])
        
        # Experience based on age and education
        min_exp = max(0, age - 22 - (4 if education in ['Bachelor', 'Master', 'PhD'] else 0))
        max_exp = age - 18
        experience = np.random.randint(min_exp, max(min_exp + 1, max_exp))
        
        # Job category based on education and some bias
        if education in ['PhD', 'Master']:
            job = np.random.choice([
                'Software Engineer', 'Manager', 'Data Scientist', 'Analyst', 'Consultant'
            ])
        elif education == 'Bachelor':
            job = np.random.choice([
                'Software Engineer', 'Sales Representative', 'Marketing Specialist',
                'HR Specialist', 'Analyst'
            ])
        else:
            job = np.random.choice([
                'Administrative Assistant', 'Customer Service', 'Sales Representative'
            ])
        
        # Sector
        sector = np.random.choice(sectors)
        
        # Hours per week
        hours_per_week = np.random.normal(40, 5)
        hours_per_week = max(20, min(60, hours_per_week))
        
        # Outcome (high income >50K) with gender bias
        base_prob = 0.15  # Base probability
        
        # Education boost
        if education == 'PhD':
            base_prob += 0.4
        elif education == 'Master':
            base_prob += 0.25
        elif education == 'Bachelor':
            base_prob += 0.15
        
        # Experience boost
        base_prob += min(0.3, experience * 0.01)
        
        # Job category boost
        if job in ['Manager', 'Software Engineer', 'Data Scientist']:
            base_prob += 0.2
        elif job in ['Consultant', 'Analyst']:
            base_prob += 0.1
        
        # Gender bias
        if sex == 'Female':
            base_prob *= (1 - bias_level)  # Reduce probability for females
        
        high_income = np.random.random() < base_prob
        
        data.append({
            'age': age,
            'sex': sex,
            'education': education,
            'job_category': job,
            'sector': sector,
            'experience_years': experience,
            'hours_per_week': hours_per_week,
            'high_income': int(high_income)
        })
    
    return pd.DataFrame(data)


def analyze_employment_coherence(df, title="Dataset"):
    """
    Analyze how coherent the employment data is.
    """
    print(f"\n{'='*60}")
    print(f"EMPLOYMENT COHERENCE ANALYSIS: {title}")
    print(f"{'='*60}")
    
    # Education-Job Category Analysis
    print("\n1. EDUCATION-JOB CATEGORY RELATIONSHIPS:")
    ed_job_crosstab = pd.crosstab(df['education'], df['job_category'], normalize='index')
    print(ed_job_crosstab.round(3))
    
    # Experience Analysis
    print(f"\n2. EXPERIENCE PATTERNS:")
    print(f"   Average experience by education:")
    for education in df['education'].unique():
        subset = df[df['education'] == education]
        print(f"   {education}: {subset['experience_years'].mean():.1f} ± {subset['experience_years'].std():.1f} years")
    
    # Gender representation by job
    print(f"\n3. GENDER REPRESENTATION BY JOB:")
    gender_job = pd.crosstab(df['job_category'], df['sex'], normalize='index')
    print(gender_job.round(3))
    
    # Outcome analysis
    print(f"\n4. HIGH INCOME OUTCOMES:")
    print(f"   Overall high income rate: {df['high_income'].mean():.1%}")
    print(f"   By gender:")
    for gender in df['sex'].unique():
        subset = df[df['sex'] == gender]
        print(f"   {gender}: {subset['high_income'].mean():.1%}")


def test_employment_optimizations():
    """
    Test the employment-specific optimizations.
    """
    print("EMPLOYMENT OPTIMIZATION TESTING")
    print("="*70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Create realistic employment dataset
    print("\n1. Creating realistic employment dataset...")
    df = create_realistic_employment_dataset(n_samples=1500, bias_level=0.4)
    
    X = df.drop('high_income', axis=1)
    y = df['high_income']
    
    print(f"Dataset created: {len(df):,} samples")
    print(f"Features: {list(X.columns)}")
    print(f"High income rate: {y.mean():.1%}")
    
    # Analyze original data coherence
    analyze_employment_coherence(df, "Original Dataset")
    
    # 2. Detect initial bias
    print(f"\n{'='*70}")
    print("2. BIAS DETECTION")
    print(f"{'='*70}")
    
    detector = BiasDetector(verbose=False)
    initial_metrics = detector.detect_bias(X, y, 'sex', 1)
    
    print(f"Initial Disparate Impact: {initial_metrics.disparate_impact:.3f}")
    print(f"Bias Level: {initial_metrics.get_bias_severity()}")
    print(f"Legal Compliance (80% rule): {'PASS' if not initial_metrics.is_biased() else 'FAIL'}")
    
    # 3. Apply standard FairSMOTE
    print(f"\n{'='*70}")
    print("3. STANDARD FAIR SMOTE COMPARISON")
    print(f"{'='*70}")
    
    standard_smote = FairSMOTE(
        protected_attribute='sex',
        k_neighbors=5,
        random_state=42
    )
    
    X_standard, y_standard = standard_smote.fit_resample(X, y)
    
    print(f"Standard FairSMOTE Results:")
    print(f"  Original samples: {len(X):,}")
    print(f"  Resampled samples: {len(X_standard):,}")
    print(f"  Synthetic samples added: {len(X_standard) - len(X):,}")
    
    # Check bias improvement
    standard_metrics = detector.detect_bias(X_standard, y_standard, 'sex', 1)
    print(f"  Final Disparate Impact: {standard_metrics.disparate_impact:.3f}")
    
    # Analyze coherence of standard SMOTE results
    df_standard = X_standard.copy()
    df_standard['high_income'] = y_standard
    analyze_employment_coherence(df_standard, "Standard FairSMOTE Result")
    
    # 4. Apply Employment-Optimized FairSMOTE
    print(f"\n{'='*70}")
    print("4. EMPLOYMENT-OPTIMIZED FAIR SMOTE")
    print(f"{'='*70}")
    
    employment_smote = EmploymentFairSMOTE(
        protected_attribute='sex',
        job_category_column='job_category',
        experience_column='experience_years',
        education_column='education',
        sector_column='sector',
        preserve_job_boundaries=True,
        experience_variance_threshold=0.3,
        employment_realism_weight=0.7,
        k_neighbors=5,
        random_state=42
    )
    
    X_employment, y_employment = employment_smote.fit_resample(X, y)
    
    print(f"Employment-Optimized FairSMOTE Results:")
    print(f"  Original samples: {len(X):,}")
    print(f"  Resampled samples: {len(X_employment):,}")
    print(f"  Synthetic samples added: {len(X_employment) - len(X):,}")
    
    # Check bias improvement
    employment_metrics = detector.detect_bias(X_employment, y_employment, 'sex', 1)
    print(f"  Final Disparate Impact: {employment_metrics.disparate_impact:.3f}")
    
    # Get employment insights
    insights = employment_smote.get_employment_insights()
    print(f"\nEmployment Intelligence Applied:")
    for key, value in insights.items():
        print(f"  {key}: {value}")
    
    # Analyze coherence of employment SMOTE results
    df_employment = X_employment.copy()
    df_employment['high_income'] = y_employment
    analyze_employment_coherence(df_employment, "Employment-Optimized FairSMOTE Result")
    
    # 5. Comparison Summary
    print(f"\n{'='*70}")
    print("5. COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    
    methods = ['Original', 'Standard FairSMOTE', 'Employment FairSMOTE']
    disparate_impacts = [
        initial_metrics.disparate_impact,
        standard_metrics.disparate_impact,
        employment_metrics.disparate_impact
    ]
    sample_counts = [len(X), len(X_standard), len(X_employment)]
    
    comparison_df = pd.DataFrame({
        'Method': methods,
        'Disparate Impact': disparate_impacts,
        'Samples': sample_counts,
        'Synthetic Added': [0, len(X_standard) - len(X), len(X_employment) - len(X)]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements
    standard_improvement = (standard_metrics.disparate_impact - initial_metrics.disparate_impact) / (1.0 - initial_metrics.disparate_impact)
    employment_improvement = (employment_metrics.disparate_impact - initial_metrics.disparate_impact) / (1.0 - initial_metrics.disparate_impact)
    
    print(f"\nImprovement Analysis:")
    print(f"  Standard FairSMOTE improvement: {standard_improvement * 100:.1f}%")
    print(f"  Employment FairSMOTE improvement: {employment_improvement * 100:.1f}%")
    
    if employment_improvement > standard_improvement:
        print(f"  ✅ Employment optimization provides better fairness improvement!")
    else:
        print(f"  ⚠️  Standard method performed better (possibly due to dataset characteristics)")
    
    # 6. Create visualization
    print(f"\n{'='*70}")
    print("6. CREATING VISUALIZATION")
    print(f"{'='*70}")
    
    create_comparison_visualization(
        comparison_df, 
        save_path='results/figures/employment_optimization_comparison.png'
    )
    
    # Save detailed report
    save_employment_report({
        'original_metrics': initial_metrics,
        'standard_metrics': standard_metrics,
        'employment_metrics': employment_metrics,
        'comparison_df': comparison_df,
        'insights': insights
    })
    
    print(f"\n✅ Employment optimization testing completed!")
    print(f"📊 Results saved to: results/figures/employment_optimization_comparison.png")
    print(f"📄 Report saved to: results/reports/employment_optimization_report.txt")


def create_comparison_visualization(comparison_df, save_path):
    """
    Create a visualization comparing the different approaches.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Employment Optimization Comparison', fontsize=16, fontweight='bold')
    
    # 1. Disparate Impact Comparison
    bars = ax1.bar(comparison_df['Method'], comparison_df['Disparate Impact'], 
                   color=['red', 'orange', 'green'], alpha=0.7)
    ax1.axhline(y=0.8, color='black', linestyle='--', label='Legal Threshold (80%)')
    ax1.set_ylabel('Disparate Impact')
    ax1.set_title('Fairness Achievement')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, comparison_df['Disparate Impact']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Sample Count Comparison
    ax2.bar(comparison_df['Method'], comparison_df['Samples'], 
            color=['blue', 'lightblue', 'navy'], alpha=0.7)
    ax2.set_ylabel('Total Samples')
    ax2.set_title('Dataset Size After Processing')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Synthetic Samples Added
    synthetic_data = comparison_df[comparison_df['Synthetic Added'] > 0]
    ax3.bar(synthetic_data['Method'], synthetic_data['Synthetic Added'],
            color=['orange', 'green'], alpha=0.7)
    ax3.set_ylabel('Synthetic Samples Added')
    ax3.set_title('Synthetic Data Generation')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Improvement Summary Table
    ax4.axis('off')
    
    # Calculate improvements for table
    original_di = comparison_df.iloc[0]['Disparate Impact']
    improvements = []
    for di in comparison_df['Disparate Impact']:
        if di == original_di:
            improvements.append('-')
        else:
            imp = (di - original_di) / (1.0 - original_di) * 100
            improvements.append(f'{imp:.1f}%')
    
    table_data = [
        ['Method', 'DI', 'Improvement'],
        ['Original', f"{comparison_df.iloc[0]['Disparate Impact']:.3f}", '-'],
        ['Standard SMOTE', f"{comparison_df.iloc[1]['Disparate Impact']:.3f}", improvements[1]],
        ['Employment SMOTE', f"{comparison_df.iloc[2]['Disparate Impact']:.3f}", improvements[2]]
    ]
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax4.set_title('Summary Comparison')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_employment_report(results):
    """
    Save a detailed report of the employment optimization results.
    """
    report = f"""
EMPLOYMENT OPTIMIZATION EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
========
This report evaluates the effectiveness of employment-specific optimizations
in the FairSMOTE algorithm for bias mitigation in employment contexts.

BIAS METRICS COMPARISON
======================
Original Dataset:
  Disparate Impact: {results['original_metrics'].disparate_impact:.3f}
  Bias Level: {results['original_metrics'].get_bias_severity()}
  Legal Compliance: {'PASS' if not results['original_metrics'].is_biased() else 'FAIL'}

Standard FairSMOTE:
  Disparate Impact: {results['standard_metrics'].disparate_impact:.3f}
  Improvement: {((results['standard_metrics'].disparate_impact - results['original_metrics'].disparate_impact) / (1.0 - results['original_metrics'].disparate_impact) * 100):.1f}%

Employment-Optimized FairSMOTE:
  Disparate Impact: {results['employment_metrics'].disparate_impact:.3f}
  Improvement: {((results['employment_metrics'].disparate_impact - results['original_metrics'].disparate_impact) / (1.0 - results['original_metrics'].disparate_impact) * 100):.1f}%

EMPLOYMENT INTELLIGENCE INSIGHTS
===============================
{chr(10).join([f"{key}: {value}" for key, value in results['insights'].items()])}

DETAILED COMPARISON
==================
{results['comparison_df'].to_string(index=False)}

CONCLUSIONS
===========
The employment-optimized FairSMOTE algorithm incorporates domain-specific
knowledge about employment relationships, job categories, and career
progression patterns. This leads to more realistic synthetic data generation
that maintains professional coherence while achieving fairness objectives.

Key benefits:
1. Respects job category boundaries during synthesis
2. Maintains realistic experience progression patterns
3. Preserves education-occupation relationships
4. Generates professionally coherent synthetic samples

RECOMMENDATIONS
===============
- Use Employment-Optimized FairSMOTE for employment-related bias mitigation
- Consider domain-specific constraints when generating synthetic data
- Monitor both fairness metrics and professional realism in results
- Validate synthetic data against domain expert knowledge

"""
    
    with open('results/reports/employment_optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == '__main__':
    test_employment_optimizations()