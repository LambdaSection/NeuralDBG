# Getting Started with Neural DSL

Welcome to Neural DSL! This guide will help you get started quickly.

## 🚀 Quick Start (5 Minutes)

### 1. Installation

```bash
pip install neural-dsl
```

### 2. Create Your First Model

**Option A: Using DSL (Traditional)**
```bash
# Create a file: my_model.neural
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}

# Compile to TensorFlow
neural compile my_model.neural --backend tensorflow --output my_model.py
```

**Option B: Using AI Assistant (NEW!)**
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST classification")
print(result['dsl_code'])

# Save to file
with open("my_model.neural", "w") as f:
    f.write(result['dsl_code'])
```

### 3. Run Your Model

```bash
python my_model.py
```

---

## 📚 Learn More

### Core Features
- [DSL Documentation](docs/dsl.md) - Learn the DSL syntax
- [Examples](examples/) - See example models
- [AI Integration Guide](docs/ai_integration_guide.md) - 🤖 Use natural language

### Advanced Features
- [Automation Guide](AUTOMATION_GUIDE.md) - 🔄 Automated releases and blog posts
- [Contributing Guide](CONTRIBUTING.md) - Contribute to Neural
- [What's New](WHATS_NEW.md) - Latest features

---

## 🎯 Common Tasks

### Build a Model
```python
# Using AI (easiest)
from neural.ai.ai_assistant import NeuralAIAssistant
assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for image classification")
```

### Visualize Architecture
```bash
neural visualize my_model.neural --format png
```

### Debug Your Model
```bash
neural debug my_model.neural
# Opens NeuralDbg dashboard
```

### Generate Documentation
```bash
neural docs my_model.neural --output model.md
```

---

## 🤖 AI-Powered Development

Neural DSL now supports AI-powered model generation!

**Example:**
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)

# Create models in natural language
result = assistant.chat("Create a CNN for MNIST classification")
result = assistant.chat("Add dropout with rate 0.5")
result = assistant.chat("Change optimizer to Adam with learning rate 0.001")

# Get final model
final_model = assistant.get_current_model()
```

**Learn More:**
- [Complete AI Guide](docs/ai_integration_guide.md)
- [AI Examples](examples/ai_examples.py)

---

## 🔄 Automation Features

Neural DSL has comprehensive automation for:
- Blog post generation
- GitHub releases
- PyPI publishing
- Example validation
- Test automation

**Quick Commands:**
```bash
# Generate blog posts
python scripts/automation/master_automation.py --blog

# Run tests
python scripts/automation/master_automation.py --test
```

**Learn More:**
- [Automation Guide](AUTOMATION_GUIDE.md)
- [Quick Start Automation](QUICK_START_AUTOMATION.md)

---

## 📖 Next Steps

1. **Try the Examples**
   - [MNIST Classifier](examples/mnist.neural)
   - [Sentiment Analysis](examples/sentiment.neural)
   - [Transformer](examples/transformer.neural)

2. **Explore Features**
   - [Shape Propagation](neural/shape_propagation/README.md)
   - [NeuralDbg](neural/dashboard/README.md)
   - [HPO](neural/hpo/README.md)

3. **Join the Community**
   - GitHub Issues for bugs/features
   - Discussions for questions
   - Discord for real-time chat

---

## 🆘 Need Help?

- **Documentation**: Check [docs/](docs/)
- **Examples**: Check [examples/](examples/)
- **Issues**: Open a GitHub issue
- **Questions**: Start a discussion

---

**Welcome to Neural DSL! Happy coding!** 🚀

