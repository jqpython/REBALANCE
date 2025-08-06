"""
REBALANCE: Automated Gender Bias Resolution Toolkit

A comprehensive toolkit for detecting and mitigating gender bias in employment
datasets using fairness-aware synthetic data generation and intelligent
bias mitigation strategies.
"""

from .bias_detection.detector import BiasDetector
from .fairness_smote.fair_smote import FairSMOTE
from .fairness_smote.employment_fair_smote import EmploymentFairSMOTE
from .evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from .recommendation.advisor import RecommendationAdvisor

__version__ = "1.0.0"
__author__ = "REBALANCE Team"

__all__ = [
    'BiasDetector',
    'FairSMOTE',
    'EmploymentFairSMOTE',
    'ComprehensiveEvaluator',
    'RecommendationAdvisor'
]
