# 👋 Welcome to Neural DSL!

## 🎯 What is Neural DSL?

Neural DSL is a **domain-specific language** for neural networks that makes deep learning development easier through:

- **🤖 AI-Powered Development** - Build models using natural language
- **🔄 Complete Automation** - Automated releases, blog posts, and maintenance
- **📊 Shape Propagation** - Catch errors before runtime
- **🔍 Built-in Debugging** - NeuralDbg for real-time monitoring
- **🌐 Cross-Framework** - Generate code for TensorFlow, PyTorch, or ONNX

---

## 🚀 Quick Start

### 1. Install
```bash
pip install neural-dsl
```

### 2. Create Your First Model

**Using AI (Easiest!):**
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST classification")
print(result['dsl_code'])
```

**Using DSL:**
```bash
# Create my_model.neural
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

# Compile
neural compile my_model.neural --backend tensorflow
```

### 3. Run
```bash
python my_model_tensorflow.py
```

---

## 📚 Documentation

### For Users
- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[What's New](WHATS_NEW.md)** - Latest features
- **[AI Integration Guide](docs/ai_integration_guide.md)** - 🤖 AI features
- **[Examples](examples/)** - Example models

### For Developers
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Automation Guide](AUTOMATION_GUIDE.md)** - 🔄 Automation features
- **[Development Guide](README_DEVELOPMENT.md)** - Development setup

### Navigation
- **[Complete Index](INDEX.md)** - All files and guides
- **[README](README.md)** - Main documentation

---

## ✨ Key Features

### 🤖 AI-Powered Development
Build models using natural language in any language!

```python
assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for image classification")
# Generates complete DSL code automatically!
```

### 🔄 Fully Automated
Everything is automated - releases, blog posts, tests, maintenance.

```bash
# Generate blog posts
python scripts/automation/master_automation.py --blog

# Run tests
python scripts/automation/master_automation.py --test
```

### 📊 Shape Propagation
Catch dimension mismatches before runtime.

### 🔍 NeuralDbg
Built-in debugger with real-time monitoring.

---

## 🎯 What's Next?

1. **Try the Examples** - See `examples/` directory
2. **Read the Guides** - Check `docs/` directory
3. **Use AI Features** - See [AI Integration Guide](docs/ai_integration_guide.md)
4. **Contribute** - See [Contributing Guide](CONTRIBUTING.md)

---

## 🆘 Need Help?

- **Documentation**: Check [docs/](docs/)
- **Examples**: Check [examples/](examples/)
- **Issues**: Open a GitHub issue
- **Questions**: Start a discussion

---

**Welcome to Neural DSL! Happy coding!** 🚀

---

*For detailed information, see [README.md](README.md) or [INDEX.md](INDEX.md)*

