# Getting Started with REBALANCE

Welcome to REBALANCE! This guide will help you get up and running with fair machine learning for employment contexts in just a few minutes.

## Installation

### Basic Installation

```bash
pip install rebalance-toolkit
```

This installs the core REBALANCE functionality with all required dependencies.

### Full Installation (Recommended)

```bash
pip install rebalance-toolkit[external]
```

This includes external fairness libraries (AIF360, Fairlearn) for comprehensive comparisons.

### Development Installation

```bash
git clone https://github.com/rebalance-team/rebalance.git
cd rebalance
pip install -e ".[dev,external]"
```

## Your First Bias Detection

Let's start with a simple example using the classic UCI Adult dataset:

```python
import pandas as pd
import numpy as np
from rebalance import BiasDetector

# Load some sample data (you can use your own CSV file)
from sklearn.datasets import fetch_openml
data = fetch_openml("adult", version=2, as_frame=True)
X = data.data
y = (data.target == '>50K').astype(int)

# Detect bias in the dataset
detector = BiasDetector()
metrics = detector.detect_bias(X, y, protected_attribute='sex', positive_label=1)

print(f"Disparate Impact: {metrics.disparate_impact:.3f}")
print(f"Bias Level: {metrics.get_bias_severity()}")
print(f"Passes 80% Rule: {'Yes' if not metrics.is_biased() else 'No'}")
```

## Getting Recommendations

REBALANCE's recommendation system helps you choose the best bias mitigation approach:

```python
from rebalance import RecommendationAdvisor

advisor = RecommendationAdvisor()

# Get a quick recommendation
recommendation = advisor.get_quick_recommendation(X, y, 'sex')

print(f"🎯 Recommended Technique: {recommendation.technique_name}")
print(f"📊 Confidence: {recommendation.confidence_score:.2f}")
print(f"📈 Expected DI Improvement: {recommendation.expected_di_improvement:.1f}%")

# Generate a comprehensive report
report = advisor.generate_recommendation_report(X, y, 'sex')
print(report[:500] + "...")  # Preview first 500 characters
```

## Applying Bias Mitigation

Once you have a recommendation, apply fairness-aware SMOTE:

```python
from rebalance import FairSMOTE

# Apply the recommended technique
fair_smote = FairSMOTE(
    protected_attribute='sex',
    k_neighbors=5,
    random_state=42
)

# Generate fair synthetic data
X_fair, y_fair = fair_smote.fit_resample(X, y)

print(f"Original dataset: {len(X):,} samples")
print(f"Balanced dataset: {len(X_fair):,} samples")
print(f"Synthetic samples added: {len(X_fair) - len(X):,}")
```

## Verifying Improvement

Check that your intervention actually improved fairness:

```python
# Check the improvement
final_metrics = detector.detect_bias(X_fair, y_fair, 'sex', 1)

print(f"\n📊 BIAS IMPROVEMENT SUMMARY")
print(f"{'='*40}")
print(f"Before: DI = {metrics.disparate_impact:.3f}")
print(f"After:  DI = {final_metrics.disparate_impact:.3f}")

improvement = (final_metrics.disparate_impact - metrics.disparate_impact) / (1.0 - metrics.disparate_impact)
print(f"Improvement: {improvement * 100:+.1f}%")
```

## Using the Command Line

REBALANCE also provides a convenient command-line interface:

```bash
# Save your data as CSV first
# Then use the CLI

# Detect bias
rebalance detect --input adult_data.csv --target income --protected-attr sex

# Get recommendations
rebalance recommend --input adult_data.csv --target income --protected-attr sex --output recommendations.txt

# Apply mitigation
rebalance mitigate --input adult_data.csv --target income --protected-attr sex --output fair_data.csv
```

## Working with Your Own Data

### Data Requirements

Your dataset should have:
- **Target variable**: Binary outcome (0/1, 'Yes'/'No', '>50K'/'<=50K', etc.)
- **Protected attribute**: The attribute you want to ensure fairness for (e.g., 'sex', 'gender', 'race')
- **Features**: Other variables for prediction

### Data Format

```python
# Expected format
print(X.head())
#     age  education  hours_per_week  workclass  sex
# 0    25         12              40    Private  Male
# 1    35         16              50    Private  Female
# 2    45         14              40  Government  Male

print(y.value_counts())
# 0    30000  # Not hired / Low income
# 1     8000  # Hired / High income
```

### Handling Categorical Data

REBALANCE automatically handles categorical data:

```python
# No preprocessing needed!
# REBALANCE handles categorical encoding internally

fair_smote = FairSMOTE(protected_attribute='sex')
X_resampled, y_resampled = fair_smote.fit_resample(X, y)

# Categorical values are preserved in the output
print(X_resampled['workclass'].unique())
# ['Private', 'Government', 'Self-employed']
```

## Common Use Cases

### 1. Hiring Bias Detection

```python
# Dataset: applications with hiring decisions
# Target: 'hired' (1 = hired, 0 = not hired)
# Protected: 'gender' or 'sex'

detector = BiasDetector()
metrics = detector.detect_bias(applications, hired, 'gender', 1)

if metrics.is_biased():
    print("⚠️ Potential hiring bias detected!")
    print(f"Female hiring rate: {metrics.group_positive_rates['Female']:.1%}")
    print(f"Male hiring rate: {metrics.group_positive_rates['Male']:.1%}")
```

### 2. Promotion Fairness

```python
# Dataset: employee records with promotion decisions
# Target: 'promoted' (1 = promoted, 0 = not promoted)
# Protected: 'sex'

promotion_metrics = detector.detect_bias(employees, promoted, 'sex', 1)
print(f"Promotion disparate impact: {promotion_metrics.disparate_impact:.3f}")
```

### 3. Salary Bias Analysis

```python
# For salary analysis, convert to binary outcome first
high_salary = (salaries > salaries.median()).astype(int)

salary_metrics = detector.detect_bias(employee_features, high_salary, 'sex', 1)
print(f"High salary disparate impact: {salary_metrics.disparate_impact:.3f}")
```

## Next Steps

Now that you've got the basics:

1. **Explore the API**: Check out the [API Reference](api/) for detailed documentation
2. **Advanced Usage**: Learn about [custom fairness strategies](guide/advanced-usage.md)
3. **Integration**: See how to [integrate with your ML pipeline](guide/integration.md)
4. **Best Practices**: Read our [Employment-Specific Guide](employment/) for HR contexts
5. **Contributing**: Help improve REBALANCE by [contributing](../CONTRIBUTING.md)

## Need Help?

- 📚 **Documentation**: Check the [User Guide](guide/) for detailed tutorials
- 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/rebalance-team/rebalance/issues)
- 💬 **Discussions**: Ask questions in [GitHub Discussions](https://github.com/rebalance-team/rebalance/discussions)
- 📧 **Contact**: Reach out to the team for collaboration opportunities

---

Ready to build fairer AI systems? Let's dive deeper into the [User Guide](guide/) or explore specific [Employment Applications](employment/)!