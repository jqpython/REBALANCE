# Fairness in Machine Learning: A REBALANCE Primer

Understanding fairness in machine learning is crucial for building ethical AI systems. This primer explains key concepts, metrics, and considerations specifically for employment contexts.

## What is Algorithmic Bias?

**Algorithmic bias** occurs when machine learning systems systematically and unfairly discriminate against certain groups of people. In employment contexts, this manifests as:

- **Hiring algorithms** that favor one gender over another
- **Promotion systems** that systematically disadvantage certain groups
- **Performance evaluation** tools that encode historical biases
- **Salary prediction** models that perpetuate pay gaps

### Sources of Bias

1. **Historical Bias**: Training data reflects past discrimination
2. **Representation Bias**: Certain groups are underrepresented in data
3. **Measurement Bias**: Different quality of data for different groups
4. **Evaluation Bias**: Using inappropriate benchmarks or metrics

## Key Fairness Metrics

REBALANCE implements several standard fairness metrics. Here's what they mean and when to use them:

### 1. Disparate Impact (DI)

**Definition**: The ratio of positive outcome rates between unprivileged and privileged groups.

```
DI = P(positive | unprivileged) / P(positive | privileged)
```

**Interpretation**:
- DI = 1.0: Perfect fairness
- DI = 0.8: Legal threshold (80% rule)
- DI < 0.8: Potential discrimination

**Example**: If 30% of male applicants are hired but only 21% of female applicants, then DI = 21%/30% = 0.7, indicating potential bias.

**When to use**: Legal compliance, simple fairness assessment

### 2. Statistical Parity Difference

**Definition**: The difference in positive outcome rates between groups.

```
SPD = P(positive | unprivileged) - P(positive | privileged)
```

**Interpretation**:
- SPD = 0: Perfect parity
- SPD < 0: Unprivileged group disadvantaged
- SPD > 0: Privileged group disadvantaged

**Example**: If 30% of men are promoted vs 25% of women, SPD = 25% - 30% = -5%.

**When to use**: Clear difference measurements, policy analysis

### 3. Equal Opportunity Difference

**Definition**: Difference in true positive rates (sensitivity) between groups.

```
EOD = TPR(unprivileged) - TPR(privileged)
```

**Interpretation**:
- EOD = 0: Equal opportunity
- EOD < 0: Qualified unprivileged individuals less likely to get positive outcomes

**Example**: Among qualified candidates, 80% of men are hired vs 70% of women, EOD = 70% - 80% = -10%.

**When to use**: When you want to ensure qualified individuals have equal chances

### 4. Demographic Parity

**Definition**: Requires that the algorithm's decisions are independent of the protected attribute.

```
P(decision = positive | protected = group1) = P(decision = positive | protected = group2)
```

**When to use**: When outcomes should be proportional to group representation

## Understanding Trade-offs

### Fairness vs. Accuracy

There's often a tension between fairness and predictive accuracy:

- **Fairness improvements** may reduce overall accuracy
- **High accuracy** might come at the cost of fairness
- **REBALANCE aims** to minimize this trade-off

### Different Fairness Definitions

Different fairness metrics can conflict:

- **Statistical parity** vs **equal opportunity** can be mutually exclusive
- **Individual fairness** vs **group fairness** may require different approaches
- **REBALANCE provides** multiple metrics to help you choose

## Employment-Specific Considerations

### Legal Framework

Understanding the legal context is crucial:

**80% Rule (Four-Fifths Rule)**:
- Used by EEOC for discrimination assessment
- Selection rate for protected group ≥ 80% of highest group
- Built into REBALANCE's bias detection

**Disparate Treatment vs. Disparate Impact**:
- **Disparate Treatment**: Intentional discrimination
- **Disparate Impact**: Unintentional but disproportionate effects
- REBALANCE focuses on detecting and mitigating disparate impact

### Protected Attributes in Employment

Common protected attributes in employment contexts:

- **Gender/Sex**: Primary focus of REBALANCE
- **Race/Ethnicity**: Often legally protected
- **Age**: Age discrimination considerations
- **Disability Status**: Accessibility and accommodation
- **Sexual Orientation**: Increasingly protected

### Employment Decisions

Different types of employment decisions have different fairness requirements:

**Hiring**:
- Focus on equal opportunity for qualified candidates
- Consider pipeline effects and representation

**Promotion**:
- Account for tenure and performance history
- Consider intersectional effects

**Performance Evaluation**:
- Ensure consistent standards across groups
- Address measurement bias in evaluation criteria

**Compensation**:
- Control for legitimate factors (experience, education)
- Identify unexplained gaps

## Bias Mitigation Strategies

### Pre-processing (REBALANCE's Focus)

**Advantages**:
- Works with any downstream algorithm
- Transparent and interpretable
- Can be validated independently

**REBALANCE Approach**:
- Fairness-aware synthetic data generation
- Preserves data relationships
- Employment-specific optimizations

### In-processing

**Approach**: Modify the learning algorithm to incorporate fairness constraints

**Examples**:
- Fairlearn's GridSearch
- Fairness constraints in optimization

### Post-processing

**Approach**: Adjust predictions to achieve fairness

**Examples**:
- Threshold optimization
- Calibration adjustments

## Best Practices for Employment AI

### 1. Multi-Metric Evaluation

Don't rely on a single fairness metric:

```python
from rebalance import BiasDetector

detector = BiasDetector()
metrics = detector.detect_bias(X, y, 'sex', 1)

print(f"Disparate Impact: {metrics.disparate_impact:.3f}")
print(f"Statistical Parity Diff: {metrics.statistical_parity_difference:.3f}")
print(f"Bias Severity: {metrics.get_bias_severity()}")
```

### 2. Intersectional Analysis

Consider multiple protected attributes:

```python
# Analyze intersections
for race in ['White', 'Black', 'Asian']:
    for sex in ['Male', 'Female']:
        subset = data[(data['race'] == race) & (data['sex'] == sex)]
        if len(subset) > 50:  # Sufficient sample size
            metrics = detector.detect_bias(subset, y[subset.index], 'sex', 1)
            print(f"{race} {sex}: DI = {metrics.disparate_impact:.3f}")
```

### 3. Temporal Monitoring

Bias can emerge over time:

```python
# Monitor bias over time
for year in [2020, 2021, 2022, 2023]:
    year_data = data[data['year'] == year]
    metrics = detector.detect_bias(year_data, y[year_data.index], 'sex', 1)
    print(f"{year}: DI = {metrics.disparate_impact:.3f}")
```

### 4. Stakeholder Involvement

- **HR Professionals**: Understand business context
- **Legal Teams**: Ensure compliance
- **Affected Groups**: Include diverse perspectives
- **Technical Teams**: Implement solutions correctly

## Common Pitfalls and How to Avoid Them

### 1. "Fairness Through Unawareness"

**Pitfall**: Removing protected attributes from the dataset
**Problem**: Proxy variables can still encode bias
**Solution**: Use fairness-aware techniques like REBALANCE

### 2. Focusing Only on Accuracy

**Pitfall**: Optimizing only for predictive performance
**Problem**: Can perpetuate or amplify existing biases
**Solution**: Multi-objective optimization considering fairness

### 3. One-Size-Fits-All Approach

**Pitfall**: Using the same fairness definition for all contexts
**Problem**: Different situations require different fairness concepts
**Solution**: Use REBALANCE's recommendation system for context-specific guidance

### 4. Ignoring Intersectionality

**Pitfall**: Analyzing only single protected attributes
**Problem**: Intersectional groups may face unique discrimination
**Solution**: Analyze combinations of protected attributes

## Ethical Considerations

### Limits of Technical Solutions

- **Technology alone cannot solve discrimination**
- **Policy and cultural changes are essential**
- **Human oversight remains crucial**
- **Consider unintended consequences**

### Privacy and Data Protection

- **Minimize collection of sensitive attributes**
- **Use differential privacy when appropriate**
- **Consider data retention policies**
- **Respect individual autonomy**

### Transparency and Explainability

- **Document bias detection and mitigation steps**
- **Provide clear explanations of decisions**
- **Enable auditability and accountability**
- **Communicate limitations honestly**

## Getting Started with REBALANCE

### Step 1: Assess Your Current Situation

```python
from rebalance import RecommendationAdvisor

advisor = RecommendationAdvisor()
profile = advisor.analyze_dataset(X, y, 'sex')

print(f"Bias Level: {profile.bias_level}")
print(f"Dataset Complexity: {profile.complexity_level}")
print(f"Recommended Approach: {advisor.get_quick_recommendation(X, y, 'sex').technique_name}")
```

### Step 2: Apply Appropriate Techniques

```python
from rebalance import FairSMOTE

# Use recommendation
fair_smote = FairSMOTE(protected_attribute='sex')
X_fair, y_fair = fair_smote.fit_resample(X, y)
```

### Step 3: Validate and Monitor

```python
# Verify improvement
final_metrics = detector.detect_bias(X_fair, y_fair, 'sex', 1)
print(f"Improvement: {final_metrics.disparate_impact:.3f} vs {metrics.disparate_impact:.3f}")

# Set up monitoring
# (implement periodic bias checks)
```

## Further Reading

- **[REBALANCE User Guide](guide/)**: Practical implementation guidance
- **[Employment Applications](employment/)**: Specific HR use cases
- **[API Reference](api/)**: Technical documentation
- **Legal Resources**: EEOC guidelines, local anti-discrimination laws
- **Academic Literature**: Latest research in algorithmic fairness

---

**Remember**: Fairness is not a destination but an ongoing process. REBALANCE provides tools to help you on this journey, but human judgment, domain expertise, and continuous monitoring remain essential.

*Ready to apply these concepts? Check out our [Getting Started Guide](getting-started.md) or dive into specific [Employment Applications](employment/).*