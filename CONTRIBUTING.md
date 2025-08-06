# Contributing to REBALANCE

Thank you for your interest in contributing to REBALANCE! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct that promotes respect, inclusivity, and collaborative development. By participating, you are expected to uphold this code.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/rebalance.git
   cd rebalance
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,external]"
   ```

4. **Verify the installation**
   ```bash
   pytest tests/
   rebalance --help
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black src tests
   flake8 src

   # Type checking
   mypy src --ignore-missing-imports
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## Types of Contributions

### 🐛 Bug Fixes
- Report bugs using GitHub issues
- Include minimal reproducible examples
- Fix bugs with test cases

### ✨ New Features
- Discuss major features in issues first
- Focus on bias detection and mitigation improvements
- Maintain backward compatibility

### 📚 Documentation
- Improve API documentation
- Add usage examples
- Update README and guides

### 🧪 Testing
- Add test cases for edge cases
- Improve test coverage
- Add integration tests

### 🔧 Performance
- Optimize algorithms
- Improve memory usage
- Benchmark improvements

## Coding Standards

### Code Style
- Use Black for code formatting
- Follow PEP 8 guidelines
- Maximum line length: 88 characters
- Use type hints where appropriate

### Testing Requirements
- All new features must include tests
- Maintain test coverage above 80%
- Use descriptive test names
- Include edge case testing

### Documentation
- Use Google-style docstrings
- Include examples in docstrings
- Update CHANGELOG.md for user-facing changes

## Bias Mitigation Guidelines

When contributing to bias mitigation algorithms:

### 🎯 Fairness Principles
- **Transparency**: Algorithms should be interpretable
- **Robustness**: Handle edge cases gracefully
- **Effectiveness**: Demonstrate measurable bias reduction
- **Efficiency**: Consider computational costs

### 📊 Evaluation Standards
- Test on multiple datasets
- Compare against existing methods
- Measure multiple fairness metrics
- Consider intersectional bias

### ⚠️ Ethical Considerations
- Document limitations clearly
- Avoid creating new forms of bias
- Consider real-world deployment impacts
- Respect privacy and data protection

## Pull Request Guidelines

### PR Requirements
- [ ] Clear description of changes
- [ ] Tests for new functionality
- [ ] Documentation updates
- [ ] No breaking changes (or clearly marked)
- [ ] Changelog entry for user-facing changes

### PR Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (specify):

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Tested on multiple datasets

## Fairness Impact
- [ ] Improves bias detection
- [ ] Improves bias mitigation
- [ ] No fairness impact
- [ ] Potential negative impact (explain)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data in commits
```

## Issue Guidelines

### Bug Reports
Use the bug report template and include:
- Clear reproduction steps
- Environment details
- Expected vs actual behavior
- Error messages or logs

### Feature Requests
- Describe the problem you're solving
- Explain why existing solutions are insufficient
- Consider fairness implications
- Provide use case examples

## Release Process

### Version Numbering
We follow Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] GitHub release created
- [ ] PyPI package published

## Recognition

Contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md acknowledgments
- Annual contributor highlights

## Getting Help

### Questions and Discussions
- GitHub Discussions for general questions
- Issues for bug reports and feature requests
- Email for security-related concerns

### Community
- Be respectful and inclusive
- Help others learn and contribute
- Focus on constructive feedback

## Fairness Research Guidelines

### Novel Algorithm Contributions
When proposing new bias mitigation algorithms:

1. **Literature Review**: Reference existing work
2. **Theoretical Foundation**: Explain mathematical basis
3. **Empirical Validation**: Test on standard datasets
4. **Comparative Analysis**: Compare with existing methods
5. **Limitations**: Document failure cases and limitations

### Dataset Considerations
- Use diverse datasets for testing
- Consider different types of bias
- Respect data privacy and consent
- Document dataset characteristics

### Reproducibility
- Provide complete code implementations
- Use fixed random seeds where appropriate
- Document hyperparameter choices
- Make results reproducible

## Legal and Ethical

### License
By contributing, you agree that your contributions will be licensed under the MIT License.

### Intellectual Property
- Ensure you have rights to contribute your code
- Don't include copyrighted material without permission
- Respect third-party licenses

### Ethics
- Consider societal impact of changes
- Avoid introducing discriminatory bias
- Respect user privacy and data protection
- Follow institutional ethics guidelines

---

Thank you for contributing to fairness in machine learning! 🎯⚖️