# Contributing to Neural DSL

Thank you for your interest in contributing to Neural DSL! This guide will help you get started.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/Neural.git
   cd Neural
   ```

2. **Install Dependencies**
   ```bash
   pip install -e .
   pip install -e ".[full]"  # For all optional dependencies
   ```

3. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

4. **Make Changes** and submit a pull request!

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

3. **Install in development mode**
   ```bash
   pip install -e .
   pip install -e ".[full]"  # For all features
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

2. **Validate examples**
   ```bash
   python scripts/automation/master_automation.py --validate
   ```

3. **Check code style**
   ```bash
   python -m pylint neural/
   ```

4. **Test your changes**
   - Manual testing
   - Edge cases
   - Error handling

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

