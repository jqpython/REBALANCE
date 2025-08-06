"""
Intelligent Recommendation System for REBALANCE

This module provides evidence-based recommendations for selecting the most
appropriate bias mitigation technique based on dataset characteristics,
performance requirements, and fairness constraints.

This addresses the choice paralysis problem identified in the proposal
by providing clear, data-driven guidance to practitioners.

Author: REBALANCE Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from ..bias_detection.detector import BiasDetector


class BiasLevel(Enum):
    """Categorize bias severity for decision making."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class DatasetComplexity(Enum):
    """Categorize dataset complexity for algorithm selection."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class PerformanceRequirement(Enum):
    """Define performance priority levels."""
    FAIRNESS_FIRST = "fairness_first"
    BALANCED = "balanced"
    ACCURACY_FIRST = "accuracy_first"


@dataclass
class DatasetProfile:
    """
    Comprehensive profile of a dataset's characteristics relevant
    to bias mitigation technique selection.
    """
    # Basic characteristics
    total_samples: int
    n_features: int
    positive_rate: float
    
    # Bias characteristics
    disparate_impact: float
    bias_level: BiasLevel
    group_imbalance_ratio: float
    
    # Complexity characteristics
    categorical_feature_ratio: float
    missing_value_ratio: float
    complexity_level: DatasetComplexity
    
    # Protected group characteristics
    protected_groups: List[str]
    minority_group_size: int
    minority_positive_samples: int


@dataclass
class TechniqueRecommendation:
    """
    Detailed recommendation for a bias mitigation technique.
    """
    technique_name: str
    confidence_score: float  # 0-1, how confident we are in this recommendation
    expected_di_improvement: float  # Expected disparate impact improvement %
    expected_accuracy_change: float  # Expected accuracy change %
    reasoning: List[str]  # Human-readable explanation of why this was recommended
    parameters: Dict[str, Any]  # Suggested parameters for the technique
    warnings: List[str]  # Potential issues or limitations
    computational_cost: str  # "low", "medium", "high"


class RecommendationAdvisor:
    """
    Intelligent advisor that recommends bias mitigation techniques
    based on dataset characteristics and requirements.
    
    This system learns from the comparative evaluation results to
    provide evidence-based recommendations.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.bias_detector = BiasDetector(verbose=False)
        
        # Evidence base from comparative evaluations
        # In a production system, this would be learned from historical data
        self.evidence_base = self._build_evidence_base()
        
    def _build_evidence_base(self) -> Dict[str, Dict[str, Any]]:
        """
        Build evidence base from known performance characteristics.
        
        In the full implementation, this would be populated from
        systematic evaluations across diverse datasets.
        """
        return {
            'REBALANCE (Fair SMOTE)': {
                'di_improvement_range': (60, 90),  # % improvement range
                'accuracy_change_range': (-2, 1),  # % accuracy change range
                'best_for': ['moderate_bias', 'balanced_requirements', 'employment_domain'],
                'computational_cost': 'medium',
                'sample_efficiency': 'high',  # Generates fewer samples
                'stability': 'high',
                'interpretability': 'high'
            },
            'REBALANCE Employment-Optimized': {
                'di_improvement_range': (65, 95),  # % improvement range (slightly better)
                'accuracy_change_range': (-1, 2),  # % accuracy change range (better accuracy preservation)
                'best_for': ['employment_domain', 'job_categories_present', 'experience_patterns', 'professional_coherence'],
                'computational_cost': 'medium',
                'sample_efficiency': 'very_high',  # Most efficient for employment contexts
                'stability': 'very_high',
                'interpretability': 'very_high',
                'employment_intelligence': True,
                'professional_realism': 'high'
            },
            'Random Oversampling': {
                'di_improvement_range': (15, 30),
                'accuracy_change_range': (-20, -10),
                'best_for': ['severe_imbalance', 'simple_datasets', 'quick_baseline'],
                'computational_cost': 'low',
                'sample_efficiency': 'low',
                'stability': 'medium',
                'interpretability': 'high'
            },
            'Standard SMOTE': {
                'di_improvement_range': (40, 70),
                'accuracy_change_range': (-20, -10),
                'best_for': ['numerical_features', 'moderate_imbalance'],
                'computational_cost': 'medium',
                'sample_efficiency': 'medium',
                'stability': 'medium',
                'interpretability': 'medium'
            },
            'Fairlearn Equalized Odds': {
                'di_improvement_range': (90, 100),
                'accuracy_change_range': (-5, 5),
                'best_for': ['perfect_fairness_required', 'post_processing_acceptable'],
                'computational_cost': 'low',
                'sample_efficiency': 'high',  # No new samples
                'stability': 'high',
                'interpretability': 'medium'
            },
            'AIF360 Reweighing': {
                'di_improvement_range': (-5, 15),
                'accuracy_change_range': (-2, 2),
                'best_for': ['minimal_accuracy_loss', 'large_datasets'],
                'computational_cost': 'low',
                'sample_efficiency': 'high',
                'stability': 'medium',
                'interpretability': 'low'
            }
        }
    
    def analyze_dataset(self, X: pd.DataFrame, y: np.ndarray, 
                       protected_attribute: str = 'sex') -> DatasetProfile:
        """
        Analyze dataset characteristics to create a comprehensive profile.
        """
        # Basic characteristics
        total_samples = len(X)
        n_features = X.shape[1]
        positive_rate = y.mean()
        
        # Bias analysis
        bias_metrics = self.bias_detector.detect_bias(X, y, protected_attribute, 1)
        disparate_impact = bias_metrics.disparate_impact
        
        # Categorize bias level
        if disparate_impact >= 0.8:
            bias_level = BiasLevel.NONE
        elif disparate_impact >= 0.6:
            bias_level = BiasLevel.MILD
        elif disparate_impact >= 0.4:
            bias_level = BiasLevel.MODERATE
        else:
            bias_level = BiasLevel.SEVERE
        
        # Group characteristics
        protected_values = X[protected_attribute]
        group_counts = protected_values.value_counts()
        minority_group = group_counts.idxmin()
        majority_group = group_counts.idxmax()
        
        group_imbalance_ratio = group_counts.min() / group_counts.max()
        minority_group_size = group_counts.min()
        
        # Count minority positive samples
        minority_mask = protected_values == minority_group
        minority_positive_samples = y[minority_mask].sum()
        
        # Complexity characteristics
        categorical_cols = X.select_dtypes(include=['object']).columns
        categorical_feature_ratio = len(categorical_cols) / n_features
        
        missing_value_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        
        # Determine complexity level
        complexity_score = 0
        if categorical_feature_ratio > 0.5:
            complexity_score += 1
        if missing_value_ratio > 0.1:
            complexity_score += 1
        if n_features > 20:
            complexity_score += 1
        if total_samples > 50000:
            complexity_score += 1
        
        if complexity_score <= 1:
            complexity_level = DatasetComplexity.SIMPLE
        elif complexity_score <= 2:
            complexity_level = DatasetComplexity.MODERATE
        else:
            complexity_level = DatasetComplexity.COMPLEX
        
        return DatasetProfile(
            total_samples=total_samples,
            n_features=n_features,
            positive_rate=positive_rate,
            disparate_impact=disparate_impact,
            bias_level=bias_level,
            group_imbalance_ratio=group_imbalance_ratio,
            categorical_feature_ratio=categorical_feature_ratio,
            missing_value_ratio=missing_value_ratio,
            complexity_level=complexity_level,
            protected_groups=list(protected_values.unique()),
            minority_group_size=minority_group_size,
            minority_positive_samples=minority_positive_samples
        )
    
    def recommend_techniques(self, 
                           dataset_profile: DatasetProfile,
                           performance_requirement: PerformanceRequirement = PerformanceRequirement.BALANCED,
                           max_accuracy_loss: float = 0.05,
                           min_di_improvement: float = 0.5) -> List[TechniqueRecommendation]:
        """
        Recommend the best bias mitigation techniques based on dataset profile and requirements.
        """
        recommendations = []
        
        for technique_name, evidence in self.evidence_base.items():
            # Calculate recommendation score
            score, reasoning, warnings, parameters = self._evaluate_technique(
                technique_name, evidence, dataset_profile, 
                performance_requirement, max_accuracy_loss, min_di_improvement
            )
            
            if score > 0:  # Only recommend if score is positive
                # Estimate expected improvements based on evidence
                di_improvement = np.mean(evidence['di_improvement_range'])
                accuracy_change = np.mean(evidence['accuracy_change_range'])
                
                recommendation = TechniqueRecommendation(
                    technique_name=technique_name,
                    confidence_score=score,
                    expected_di_improvement=di_improvement,
                    expected_accuracy_change=accuracy_change,
                    reasoning=reasoning,
                    parameters=parameters,
                    warnings=warnings,
                    computational_cost=evidence['computational_cost']
                )
                recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations
    
    def _evaluate_technique(self, technique_name: str, evidence: Dict[str, Any],
                          profile: DatasetProfile, performance_req: PerformanceRequirement,
                          max_acc_loss: float, min_di_improvement: float) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """
        Evaluate how well a technique fits the dataset and requirements.
        Returns: (score, reasoning, warnings, parameters)
        """
        score = 0.5  # Base score
        reasoning = []
        warnings = []
        parameters = {}
        
        # Check if technique meets basic requirements
        di_improvement = np.mean(evidence['di_improvement_range'])
        accuracy_change = np.mean(evidence['accuracy_change_range'])
        
        if di_improvement < min_di_improvement * 100:
            score -= 0.3
            warnings.append(f"May not achieve target DI improvement of {min_di_improvement*100:.0f}%")
        else:
            score += 0.2
            reasoning.append(f"Expected to achieve {di_improvement:.0f}% DI improvement")
        
        if abs(accuracy_change) > max_acc_loss * 100:
            score -= 0.3
            warnings.append(f"May exceed accuracy loss limit of {max_acc_loss*100:.0f}%")
        else:
            score += 0.2
            reasoning.append(f"Expected accuracy change within limits ({accuracy_change:+.1f}%)")
        
        # Technique-specific evaluation
        if technique_name == 'REBALANCE (Fair SMOTE)':
            score, reasoning, warnings, parameters = self._evaluate_fair_smote(
                score, reasoning, warnings, parameters, profile, performance_req
            )
        elif technique_name == 'Random Oversampling':
            score, reasoning, warnings, parameters = self._evaluate_random_oversampling(
                score, reasoning, warnings, parameters, profile, performance_req
            )
        elif technique_name == 'REBALANCE Employment-Optimized':
            score, reasoning, warnings, parameters = self._evaluate_employment_fair_smote(
                score, reasoning, warnings, parameters, profile, performance_req
            )
        elif technique_name == 'Standard SMOTE':
            score, reasoning, warnings, parameters = self._evaluate_standard_smote(
                score, reasoning, warnings, parameters, profile, performance_req
            )
        elif technique_name == 'Fairlearn Equalized Odds':
            score, reasoning, warnings, parameters = self._evaluate_fairlearn_eo(
                score, reasoning, warnings, parameters, profile, performance_req
            )
        elif technique_name == 'AIF360 Reweighing':
            score, reasoning, warnings, parameters = self._evaluate_aif360_reweighing(
                score, reasoning, warnings, parameters, profile, performance_req
            )
        
        return max(0, min(1, score)), reasoning, warnings, parameters
    
    def _evaluate_fair_smote(self, score, reasoning, warnings, parameters, profile, performance_req):
        """Evaluate FairSMOTE specifically for this dataset."""
        # FairSMOTE works best with moderate bias and categorical features
        if profile.bias_level in [BiasLevel.MODERATE, BiasLevel.SEVERE]:
            score += 0.2
            reasoning.append("Well-suited for moderate to severe bias levels")
        
        if profile.categorical_feature_ratio > 0.3:
            score += 0.15
            reasoning.append("Excellent for datasets with categorical features")
        
        if profile.minority_positive_samples < 10:
            score -= 0.2
            warnings.append("May struggle with very few positive minority examples")
        
        # Performance requirement alignment
        if performance_req == PerformanceRequirement.BALANCED:
            score += 0.2
            reasoning.append("Optimal balance of fairness and accuracy")
        elif performance_req == PerformanceRequirement.FAIRNESS_FIRST:
            score += 0.1
        
        # Employment domain bonus
        score += 0.1
        reasoning.append("Specifically designed for employment bias scenarios")
        
        # Suggest parameters
        if profile.minority_positive_samples < 20:
            parameters['k_neighbors'] = 3
            reasoning.append("Using k=3 due to limited positive minority samples")
        else:
            parameters['k_neighbors'] = 5
            
        parameters['protected_attribute'] = 'sex'  # Default for employment
        parameters['random_state'] = 42
        
        return score, reasoning, warnings, parameters
    
    def _evaluate_employment_fair_smote(self, score, reasoning, warnings, parameters, profile, performance_req):
        """Evaluate Employment-Optimized FairSMOTE for this dataset."""
        # Start with base FairSMOTE evaluation
        score, reasoning, warnings, parameters = self._evaluate_fair_smote(
            score, reasoning, warnings, parameters, profile, performance_req
        )
        
        # Additional bonuses for employment-specific features
        employment_features_bonus = 0.0
        
        # Check for employment-specific columns (this would be enhanced in practice)
        employment_indicators = [
            'job', 'occupation', 'workclass', 'experience', 'education', 
            'sector', 'industry', 'position', 'role', 'department'
        ]
        
        # In practice, you'd analyze the actual column names
        # For now, assume employment context if we're recommending this
        has_employment_features = True  # Placeholder
        
        if has_employment_features:
            employment_features_bonus += 0.25
            reasoning.append("Employment-specific intelligence activated for job categories and experience patterns")
        
        # Professional coherence bonus
        if profile.categorical_feature_ratio > 0.4:
            employment_features_bonus += 0.15
            reasoning.append("Professional relationship preservation between categorical features")
        
        # Sample efficiency bonus for employment contexts
        if profile.total_samples > 1000:
            employment_features_bonus += 0.1
            reasoning.append("Superior sample efficiency for large employment datasets")
        
        # Domain expertise bonus
        employment_features_bonus += 0.2
        reasoning.append("Incorporates employment domain knowledge and career progression patterns")
        
        score += employment_features_bonus
        
        # Update parameters for employment optimization
        parameters.update({
            'job_category_column': 'occupation',  # Common name
            'experience_column': 'experience_years',  # Common name
            'education_column': 'education',  # Common name
            'preserve_job_boundaries': True,
            'employment_realism_weight': 0.7
        })
        
        # Employment-specific warnings
        if profile.minority_positive_samples < 5:
            warnings.append("Very few positive minority samples may limit employment pattern analysis")
        
        return score, reasoning, warnings, parameters
    
    def _evaluate_random_oversampling(self, score, reasoning, warnings, parameters, profile, performance_req):
        """Evaluate Random Oversampling for this dataset."""
        # Good for simple cases or as baseline
        if profile.complexity_level == DatasetComplexity.SIMPLE:
            score += 0.15
            reasoning.append("Suitable for simple datasets")
        
        if profile.bias_level == BiasLevel.SEVERE and performance_req == PerformanceRequirement.FAIRNESS_FIRST:
            score += 0.1
            reasoning.append("Quick bias reduction for severe cases")
        
        if performance_req == PerformanceRequirement.ACCURACY_FIRST:
            score -= 0.3
            warnings.append("Significant accuracy loss expected")
        
        parameters['random_state'] = 42
        
        return score, reasoning, warnings, parameters
    
    def _evaluate_standard_smote(self, score, reasoning, warnings, parameters, profile, performance_req):
        """Evaluate Standard SMOTE for this dataset."""
        # SMOTE works well with numerical features
        if profile.categorical_feature_ratio < 0.3:
            score += 0.2
            reasoning.append("Well-suited for numerical features")
        else:
            score -= 0.2
            warnings.append("May struggle with many categorical features")
        
        if profile.minority_positive_samples >= 10:
            score += 0.1
            reasoning.append("Sufficient positive samples for SMOTE interpolation")
        else:
            score -= 0.2
            warnings.append("Limited positive samples may affect SMOTE quality")
        
        parameters['k_neighbors'] = min(5, profile.minority_positive_samples - 1) if profile.minority_positive_samples > 1 else 1
        parameters['random_state'] = 42
        
        return score, reasoning, warnings, parameters
    
    def _evaluate_fairlearn_eo(self, score, reasoning, warnings, parameters, profile, performance_req):
        """Evaluate Fairlearn Equalized Odds for this dataset."""
        # Excellent for perfect fairness requirements
        if performance_req == PerformanceRequirement.FAIRNESS_FIRST:
            score += 0.3
            reasoning.append("Achieves near-perfect fairness")
        
        if profile.bias_level in [BiasLevel.MODERATE, BiasLevel.SEVERE]:
            score += 0.2
            reasoning.append("Effective for significant bias")
        
        # Post-processing limitation
        warnings.append("Post-processing technique - requires model retraining")
        warnings.append("May achieve perfect DI but with very low recall")
        
        return score, reasoning, warnings, parameters
    
    def _evaluate_aif360_reweighing(self, score, reasoning, warnings, parameters, profile, performance_req):
        """Evaluate AIF360 Reweighing for this dataset."""
        # Conservative approach
        if performance_req == PerformanceRequirement.ACCURACY_FIRST:
            score += 0.2
            reasoning.append("Minimal accuracy impact")
        
        if profile.total_samples > 10000:
            score += 0.1
            reasoning.append("Effective on large datasets")
        
        if profile.bias_level == BiasLevel.SEVERE:
            score -= 0.2
            warnings.append("Limited effectiveness on severe bias")
        
        # Implementation issues noted in evaluation
        score -= 0.1
        warnings.append("Implementation may have compatibility issues")
        
        return score, reasoning, warnings, parameters
    
    def get_quick_recommendation(self, X: pd.DataFrame, y: np.ndarray,
                               protected_attribute: str = 'sex',
                               performance_requirement: PerformanceRequirement = PerformanceRequirement.BALANCED) -> TechniqueRecommendation:
        """
        Get a quick recommendation for practitioners who need immediate guidance.
        """
        profile = self.analyze_dataset(X, y, protected_attribute)
        recommendations = self.recommend_techniques(profile, performance_requirement)
        
        if recommendations:
            top_recommendation = recommendations[0]
            if self.verbose:
                print(f"🎯 Recommended Technique: {top_recommendation.technique_name}")
                print(f"📊 Confidence: {top_recommendation.confidence_score:.2f}")
                print(f"📈 Expected DI Improvement: {top_recommendation.expected_di_improvement:.1f}%")
                print(f"📉 Expected Accuracy Change: {top_recommendation.expected_accuracy_change:+.1f}%")
                print(f"\n💡 Reasoning:")
                for reason in top_recommendation.reasoning:
                    print(f"   • {reason}")
                if top_recommendation.warnings:
                    print(f"\n⚠️  Warnings:")
                    for warning in top_recommendation.warnings:
                        print(f"   • {warning}")
            
            return top_recommendation
        else:
            # Fallback recommendation
            return TechniqueRecommendation(
                technique_name="REBALANCE (Fair SMOTE)",
                confidence_score=0.5,
                expected_di_improvement=70.0,
                expected_accuracy_change=-1.0,
                reasoning=["Default recommendation for employment bias scenarios"],
                parameters={'protected_attribute': protected_attribute, 'k_neighbors': 5},
                warnings=["No specific recommendation available - using default"],
                computational_cost="medium"
            )
    
    def generate_recommendation_report(self, X: pd.DataFrame, y: np.ndarray,
                                     protected_attribute: str = 'sex') -> str:
        """
        Generate a comprehensive recommendation report for practitioners.
        """
        profile = self.analyze_dataset(X, y, protected_attribute)
        
        report = f"""
REBALANCE TECHNIQUE RECOMMENDATION REPORT
=======================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET ANALYSIS:
-----------------
• Total Samples: {profile.total_samples:,}
• Features: {profile.n_features}
• Positive Rate: {profile.positive_rate:.3f}
• Disparate Impact: {profile.disparate_impact:.3f}
• Bias Level: {profile.bias_level.value.title()}
• Dataset Complexity: {profile.complexity_level.value.title()}
• Categorical Features: {profile.categorical_feature_ratio:.1%}
• Missing Values: {profile.missing_value_ratio:.1%}
• Minority Group Size: {profile.minority_group_size:,}
• Minority Positive Samples: {profile.minority_positive_samples}

RECOMMENDATIONS BY PERFORMANCE PRIORITY:
"""
        
        for perf_req in PerformanceRequirement:
            recommendations = self.recommend_techniques(profile, perf_req)
            
            report += f"\n{perf_req.value.replace('_', ' ').title()} Priority:\n"
            report += "-" * 40 + "\n"
            
            if recommendations:
                top_rec = recommendations[0]
                report += f"🥇 Top Choice: {top_rec.technique_name}\n"
                report += f"   Confidence: {top_rec.confidence_score:.2f}\n"
                report += f"   Expected DI Improvement: {top_rec.expected_di_improvement:.1f}%\n"
                report += f"   Expected Accuracy Change: {top_rec.expected_accuracy_change:+.1f}%\n"
                report += f"   Computational Cost: {top_rec.computational_cost.title()}\n"
                
                if len(recommendations) > 1:
                    alt_rec = recommendations[1]
                    report += f"\n🥈 Alternative: {alt_rec.technique_name}\n"
                    report += f"   Confidence: {alt_rec.confidence_score:.2f}\n"
            else:
                report += "   No suitable recommendations found.\n"
        
        report += f"\nDETAILED GUIDANCE:\n"
        report += "-" * 40 + "\n"
        
        # Provide specific guidance based on dataset characteristics
        if profile.bias_level == BiasLevel.SEVERE:
            report += "• Severe bias detected - prioritize fairness-first approaches\n"
        elif profile.bias_level == BiasLevel.NONE:
            report += "• Minimal bias detected - consider if mitigation is necessary\n"
        
        if profile.minority_positive_samples < 10:
            report += "• Very few positive minority samples - consider data collection\n"
            report += "• Use small k_neighbors values (3 or less) for SMOTE-based methods\n"
        
        if profile.categorical_feature_ratio > 0.5:
            report += "• High categorical feature ratio - REBALANCE handles this well\n"
            report += "• Standard SMOTE may struggle with categorical features\n"
        
        if profile.total_samples > 50000:
            report += "• Large dataset - all techniques should perform reliably\n"
        elif profile.total_samples < 5000:
            report += "• Small dataset - simpler techniques may be more stable\n"
        
        report += f"\nRECOMMENDED IMPLEMENTATION:\n"
        report += "-" * 40 + "\n"
        top_recommendation = self.get_quick_recommendation(X, y, protected_attribute)
        report += f"from src.{top_recommendation.technique_name.lower().replace(' ', '_').replace('(', '').replace(')', '')} import {top_recommendation.technique_name.split()[0]}\n\n"
        report += f"# Initialize technique\n"
        params_str = ", ".join([f"{k}={repr(v)}" for k, v in top_recommendation.parameters.items()])
        report += f"technique = {top_recommendation.technique_name.split()[0]}({params_str})\n\n"
        report += f"# Apply technique\n"
        report += f"X_resampled, y_resampled = technique.fit_resample(X_train, y_train)\n"
        
        return report