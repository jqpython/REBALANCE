# REBALANCE: Automated Gender Bias Resolution Toolkit

## 🎯 Project Overview

REBALANCE is a Python toolkit designed to automatically detect and mitigate gender bias in machine learning datasets, with a specific focus on employment-related data. By implementing fairness-aware variants of the Synthetic Minority Over-sampling Technique (SMOTE), REBALANCE helps data scientists and ML practitioners build more equitable AI systems without requiring deep expertise in fairness theory.

### The Problem We're Solving

Machine learning models trained on historical employment data often perpetuate gender discrimination. When datasets reflect past biases—such as fewer women in high-income positions—ML models learn these patterns as predictive features. Traditional rebalancing techniques like SMOTE address numerical imbalances but ignore protected attributes, potentially creating new forms of discrimination.

Consider this real-world impact: A hiring algorithm trained on biased data might systematically rank female candidates lower, not because they're less qualified, but because the training data reflects historical discrimination. REBALANCE addresses this by ensuring that synthetic data generation considers gender as a protected attribute.

## 🚀 Key Features

**Automated Bias Detection**: Automatically calculates key fairness metrics including disparate impact ratio and demographic parity difference, providing clear visualizations of bias levels in your dataset.

**Fairness-Aware SMOTE**: An enhanced version of SMOTE that generates synthetic samples while considering protected attributes, ensuring that rebalancing doesn't inadvertently create new biases.

**Simple API Design**: Following scikit-learn conventions, REBALANCE integrates seamlessly into existing ML workflows with intuitive fit() and transform() methods.

**Comprehensive Evaluation**: Built-in tools to measure both bias reduction and model performance, helping you understand the trade-offs between fairness and accuracy.

**Evidence-Based Recommendations**: Based on extensive testing with multiple ML models, REBALANCE provides clear guidelines on when and how to apply fairness-aware rebalancing techniques.

### Prerequisites

REBALANCE requires Python 3.8 or higher. Before installation, ensure you have the following dependencies:

```bash
python>=3.8
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
