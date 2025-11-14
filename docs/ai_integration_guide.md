# Neural AI Integration Guide

## Overview

Neural AI provides natural language to DSL conversion, making it easy to build neural networks by describing what you want in plain language.

## Quick Start

### Basic Usage (Rule-Based - No Dependencies)

```python
from neural.ai.ai_assistant import NeuralAIAssistant

# Initialize assistant (rule-based, works immediately)
assistant = NeuralAIAssistant(use_llm=False)

# Create a model from natural language
result = assistant.chat("Create a CNN for MNIST classification")
print(result['dsl_code'])
```

### With LLM (Advanced - Requires Setup)

```python
# Initialize with LLM support
assistant = NeuralAIAssistant(use_llm=True, llm_provider='ollama')

# More complex requests
result = assistant.chat("Create a ResNet50 model for ImageNet with data augmentation")
print(result['dsl_code'])
```

## Features

### 1. Natural Language to DSL

Convert natural language descriptions into Neural DSL code:

```python
assistant = NeuralAIAssistant(use_llm=False)

# Create models
result = assistant.chat("Create a CNN for image classification")
result = assistant.chat("Create a model named MyModel for MNIST")

# Add layers
result = assistant.chat("Add a convolutional layer with 32 filters")
result = assistant.chat("Add dropout with rate 0.5")
result = assistant.chat("Add a dense layer with 128 units")

# Modify configuration
result = assistant.chat("Change optimizer to Adam with learning rate 0.001")
result = assistant.chat("Set loss to categorical crossentropy")
```

### 2. Incremental Model Building

Build models step by step:

```python
assistant = NeuralAIAssistant(use_llm=False)

# Start with a model
assistant.chat("Create a model named MyCNN")

# Add layers incrementally
assistant.chat("Add Conv2D with 32 filters")
assistant.chat("Add MaxPooling2D")
assistant.chat("Add Flatten")
assistant.chat("Add Dense with 128 units")
assistant.chat("Add Output with 10 classes")

# Get final model
final_dsl = assistant.get_current_model()
print(final_dsl)
```

### 3. Multi-Language Support

Works with multiple languages (requires translation libraries):

```python
assistant = NeuralAIAssistant(use_llm=False)

# French
result = assistant.chat("Créer un CNN pour la classification d'images")

# Spanish
result = assistant.chat("Crear una CNN para clasificación de imágenes")

# Chinese
result = assistant.chat("创建一个用于图像分类的CNN")
```

### 4. LLM-Powered Generation

For more complex requests, use LLM:

```python
# Setup: Install Ollama from https://ollama.ai
# Or set OPENAI_API_KEY / ANTHROPIC_API_KEY

assistant = NeuralAIAssistant(use_llm=True, llm_provider='ollama')

# Complex requests
result = assistant.chat("Create a transformer model for NLP with attention mechanism")
result = assistant.chat("Build a GAN with generator and discriminator")
```

## Integration with Chat Interface

The AI assistant is integrated into the existing chat interface:

```python
from neural.neural_chat.neural_chat import NeuralChat

chat = NeuralChat(use_ai=True)

# Use natural language
response = chat.process_command("Create a CNN for image classification")
print(response)
```

## LLM Providers

### Option 1: Ollama (Local, Free)

1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Use in code:

```python
assistant = NeuralAIAssistant(use_llm=True, llm_provider='ollama')
```

### Option 2: OpenAI (API, Requires Key)

1. Install: `pip install openai`
2. Set environment variable: `export OPENAI_API_KEY=your_key`
3. Use in code:

```python
assistant = NeuralAIAssistant(use_llm=True, llm_provider='openai')
```

### Option 3: Anthropic Claude (API, Requires Key)

1. Install: `pip install anthropic`
2. Set environment variable: `export ANTHROPIC_API_KEY=your_key`
3. Use in code:

```python
assistant = NeuralAIAssistant(use_llm=True, llm_provider='anthropic')
```

## Response Format

The `chat()` method returns a dictionary:

```python
{
    'response': 'Text response to user',
    'dsl_code': 'Generated DSL code (if any)',
    'intent': 'Detected intent type',
    'success': True/False,
    'language': 'Detected language code'
}
```

## Examples

See `examples/ai_examples.py` for complete examples.

## Architecture

```
User Input (Natural Language)
    ↓
Language Detection & Translation (if needed)
    ↓
Intent Extraction (LLM or Rule-based)
    ↓
DSL Generation
    ↓
Validation & Response
```

## Limitations

### Current (Rule-Based)
- Works for common patterns
- Limited to predefined intents
- No context preservation between calls

### With LLM
- More flexible and intelligent
- Can handle complex requests
- Requires API key or local LLM setup

## Future Enhancements

- [ ] Context preservation (remember conversation)
- [ ] Multi-turn model building
- [ ] Automatic error correction
- [ ] Architecture suggestions
- [ ] Performance optimization recommendations

## Troubleshooting

### "LLM not available"
- Install required packages: `pip install openai` or `pip install anthropic`
- Or install Ollama: https://ollama.ai
- Or use rule-based mode: `use_llm=False`

### "Translation not working"
- Install translation library: `pip install googletrans` or `pip install deep-translator`
- Or use English input

### "Intent not recognized"
- Try more specific language
- Use LLM mode for better understanding
- Check supported commands in examples

## Contributing

To improve the AI integration:
1. Add new intent patterns in `natural_language_processor.py`
2. Enhance LLM prompts in `llm_integration.py`
3. Add language support in `multi_language.py`

