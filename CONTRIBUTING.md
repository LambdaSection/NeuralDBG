# Contributing to Neural DSL

Thank you for your interest in contributing to Neural DSL! This guide will help you get started.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/Neural.git
   cd Neural
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

4. **Make Changes** and submit a pull request!

## 📦 Versioning Strategy

Neural follows **semantic versioning** with specific rules:

### Version Format: `MAJOR.MINOR.PATCH`

- **Patch (0.0.X)**: Released every **15 bugs fixed**
  - Bug fixes only
  - No new features
  - Backward compatible

- **Minor (0.X.0)**: Released when a **new feature** is added
  - New functionality
  - Backward compatible
  - May include bug fixes

- **Major (X.0.0)**: Released when **stable with no known bugs**
  - Stable release milestone
  - All known bugs resolved (check GitHub Issues)
  - May include breaking changes
  - Production-ready

### Release Process
The automated release process is documented in [DISTRIBUTION_PLAN.md](DISTRIBUTION_PLAN.md).

Quick release:
```bash
# Patch release (15 bugs fixed)
python scripts/automation/master_automation.py --release --version-type patch

# Minor release (new feature)
python scripts/automation/master_automation.py --release --version-type minor

# Major release (stable, bug-free)
python scripts/automation/master_automation.py --release --version-type major
```

## 📋 Development Setup

### Prerequisites
- Python 3.8+
- Git
- (Optional) GitHub CLI for releases

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lemniscate-SHA-256/Neural.git
   cd Neural
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install  # Set up git hooks
   ```
   
   This installs:
   - Core Neural DSL in editable mode
   - Testing tools (pytest, pytest-cov)
   - Linting tools (ruff, pylint, flake8)
   - Type checking (mypy)
   - Pre-commit hooks
   - Security auditing (pip-audit)
   
   For testing specific features, install optional dependencies:
   ```bash
   pip install -e ".[backends]"      # ML framework testing
   pip install -e ".[visualization]" # Visualization testing
   pip install -e ".[full]"          # All features
   ```

4. **Run tests to verify setup**
   ```bash
   python -m pytest tests/ -v
   ```

## 🎯 Areas to Contribute

### High Priority
1. **AI Integration Enhancements**
   - Improve natural language understanding
   - Add more layer types to AI assistant
   - Enhance multi-language support
   - See: `neural/ai/` and `docs/ai_integration_guide.md`

2. **Example Validation**
   - Add more examples
   - Fix failing examples
   - Improve example documentation
   - See: `examples/` and `scripts/automation/example_validator.py`

3. **Test Coverage**
   - Add tests for new features
   - Improve existing tests
   - Add integration tests
   - See: `tests/`

### Medium Priority
1. **Documentation**
   - Improve guides and tutorials
   - Add more examples
   - Fix typos and clarify explanations
   - See: `docs/`

2. **Bug Fixes**
   - Check GitHub Issues
   - Fix reported bugs
   - Improve error messages

3. **Feature Development**
   - Check `ROADMAP.md` for priorities
   - Implement high-impact features
   - See: `ROADMAP.md` (internal document)

## 🔧 Development Workflow

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run tests**
   ```bash
   python -m pytest tests/ -v
   python scripts/automation/master_automation.py --test --validate
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

### Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings to functions/classes
- Keep functions focused and small

### Testing

- Write tests for new features
- Ensure all tests pass before submitting
- Add tests for bug fixes
- Aim for good test coverage

## 📝 Documentation

### When to Update Documentation

- Adding new features
- Changing existing behavior
- Fixing bugs that affect usage
- Adding examples

### Documentation Locations

- **User Guides**: `docs/`
- **API Documentation**: In code docstrings
- **Examples**: `examples/`
- **Blog Posts**: `docs/blog/` (auto-generated)

## 🐛 Reporting Bugs

### Before Reporting

1. Check if the bug is already reported
2. Try to reproduce the bug
3. Check if it's fixed in the latest version

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- Python version
- Neural DSL version
- OS
- Framework (TensorFlow/PyTorch)

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Before Requesting

1. Check `ROADMAP.md` to see if it's planned
2. Check existing issues
3. Consider if it aligns with Neural's vision

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## 🤖 Using Automation

### For Contributors

The automation system can help you:

1. **Validate Examples**
   ```bash
   python scripts/automation/master_automation.py --validate
   ```

2. **Run Tests**
   ```bash
   python scripts/automation/master_automation.py --test
   ```

3. **Check Code Quality**
   ```bash
   python -m pylint neural/
   ```

### For Maintainers

See `AUTOMATION_GUIDE.md` for full automation capabilities.

## 🎨 AI Integration Contributions

### Adding New Intents

Edit `neural/ai/natural_language_processor.py`:

```python
# Add to layer_keywords
'new_layer': ['synonyms', 'for', 'new', 'layer']

# Add to _extract_add_layer_intent
elif layer_type == 'new_layer':
    # Extract parameters
    params['param'] = extract_value(...)
```

### Improving LLM Prompts

Edit `neural/ai/llm_integration.py`:

```python
def _get_system_prompt(self) -> str:
    return """Your improved prompt here..."""
```

## 📊 Testing Your Contributions

### Before Submitting

1. **Run all tests**
   ```bash
   python -m pytest tests/ -v
   ```

2. **Run linters**
   ```bash
   python -m ruff check .
   python -m pylint neural/
   ```

3. **Type check**
   ```bash
   python -m mypy neural/code_generation neural/utils
   ```

4. **Security audit**
   ```bash
   python -m pip_audit -l --progress-spinner off
   ```

5. **Test your changes**
   - Manual testing
   - Edge cases
   - Error handling

## 📦 Managing Dependencies

When contributing code that requires new dependencies:

### Adding Dependencies

1. **Categorize the dependency** - Determine which feature group it belongs to:
   - `CORE_DEPS` - Essential for basic DSL functionality
   - `BACKEND_DEPS` - ML framework support
   - `HPO_DEPS` - Hyperparameter optimization
   - `VISUALIZATION_DEPS` - Charts and diagrams
   - `DASHBOARD_DEPS` - NeuralDbg interface
   - `CLOUD_DEPS` - Cloud integrations
   - `UTILS_DEPS` - Utilities and profiling
   - `API_DEPS` - API server support
   - `ML_EXTRAS_DEPS` - Additional ML tools

2. **Update setup.py**
   ```python
   # Add to the appropriate dependency list
   VISUALIZATION_DEPS = [
       "matplotlib<3.10",
       "graphviz>=0.20",
       "your-new-package>=1.0",  # Add here
   ]
   ```

3. **Document the dependency** in:
   - `DEPENDENCY_GUIDE.md` - Usage and purpose
   - `README.md` - If it's a major feature
   - Your PR description

4. **Test minimal installation** - Ensure core functionality works without your new dependency:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install -e .  # Core only
   # Verify basic commands work
   ```

5. **Test with dependency**:
   ```bash
   pip install -e ".[your-feature-group]"
   # Run relevant tests
   ```

### Dependency Guidelines

- **Avoid adding to CORE_DEPS** unless absolutely necessary
- **Pin minimum versions** but avoid maximum versions unless required
- **Check compatibility** with Python 3.8-3.11
- **Consider size** - large dependencies should be optional
- **Check licenses** - ensure compatibility with MIT license
- **Avoid duplication** - check if functionality exists in current dependencies

### Examples

**Good**: Adding Optuna for HPO (already in HPO_DEPS)
```python
# In neural/hpo/optimizer.py
try:
    import optuna
except ImportError:
    raise ImportError("Optuna required for HPO. Install with: pip install neural-dsl[hpo]")
```

**Bad**: Adding large library to CORE_DEPS for minor feature
```python
# Don't do this
CORE_DEPS = [
    "click>=8.1.3",
    "lark>=1.1.5",
    "numpy>=1.23.0",
    "pyyaml>=6.0.1",
    "heavy-ml-framework>=2.0",  # Bad: Too heavy for core
]
```

### Updating Development Dependencies

For dev tools (linters, formatters, etc.), update `requirements-dev.txt`:

```bash
# requirements-dev.txt
pytest>=7.0.0
pytest-cov>=4.0.0
your-new-dev-tool>=1.0  # Add here
```

## 🎯 Contribution Priorities

Based on `ROADMAP.md`:

1. **Experiment Tracking** (80% user impact)
2. **Data Pipeline Integration** (70% user impact)
3. **Model Deployment** (60% user impact)
4. **Performance Optimization** (50% user impact)
5. **Model Versioning** (40% user impact)

## 📚 Resources

### Documentation
- [AI Integration Guide](docs/ai_integration_guide.md)
- [Automation Guide](AUTOMATION_GUIDE.md)
- [DSL Documentation](docs/dsl.md)
- [Examples](examples/)

### Internal Documents (Not on GitHub)
- `ROADMAP.md` - Development roadmap
- `VISION.md` - Vision and mission
- `CHECKLIST.md` - Development checklist

## 🙏 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Discord**: For real-time chat (link in README)

## 🎉 Thank You!

Your contributions make Neural DSL better for everyone. We appreciate your time and effort!

---

**Happy Contributing!** 🚀

