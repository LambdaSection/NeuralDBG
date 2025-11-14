# Neural AI Quick Start

## Test the AI Integration (No Dependencies Required)

The rule-based natural language processor works immediately without any dependencies.

### Quick Test

```python
# Test directly (bypasses neural package imports)
import sys
sys.path.insert(0, 'neural/ai')

from natural_language_processor import NaturalLanguageProcessor, DSLGenerator

# Initialize
nlp = NaturalLanguageProcessor()
generator = DSLGenerator()

# Test intent extraction
intent, params = nlp.extract_intent("Create a CNN for image classification")
print(f"Intent: {intent.value}")
print(f"Params: {params}")

# Generate DSL
dsl = generator._generate_model(params)
print(f"\nGenerated DSL:\n{dsl}")
```

### Test Commands

Try these natural language commands:

1. **Create Model**
   - "Create a CNN for image classification"
   - "Create a model named MyModel for MNIST"
   - "Build a neural network for sentiment analysis"

2. **Add Layers**
   - "Add a convolutional layer with 32 filters"
   - "Add dropout with rate 0.5"
   - "Add a dense layer with 128 units"
   - "Add max pooling"

3. **Modify Configuration**
   - "Change optimizer to Adam with learning rate 0.001"
   - "Set loss to categorical crossentropy"

### Full Integration Test

Once you have the full Neural package installed:

```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST classification")
print(result['dsl_code'])
```

## What Works Now

✅ **Rule-Based Processing** - Works immediately, no dependencies
✅ **Intent Extraction** - Understands common commands
✅ **DSL Generation** - Generates valid Neural DSL code
✅ **Layer Support** - Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Output

## What Requires Setup

⚠️ **LLM Integration** - Requires API keys or Ollama
⚠️ **Multi-Language** - Requires translation libraries
⚠️ **Full Package** - Requires lark and other dependencies

## Next Steps

1. **Test rule-based processing** (works now)
2. **Install dependencies** for full package: `pip install lark`
3. **Set up LLM** (optional) for advanced features
4. **Try examples** in `examples/ai_examples.py`

