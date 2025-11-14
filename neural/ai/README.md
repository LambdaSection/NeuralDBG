# Neural AI Integration

## Overview

This module provides AI-powered features for Neural DSL, enabling natural language to DSL conversion with multi-language support.

## Status

🚧 **In Development** - Core infrastructure is implemented. LLM integration requires optional dependencies.

## Features

### 1. Natural Language to DSL Conversion
Convert natural language descriptions into Neural DSL code.

**Example**:
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant()
result = assistant.chat("Create a CNN for image classification with 3 convolutional layers")
print(result['dsl_code'])
```

### 2. Multi-Language Support
Process input in multiple languages (English, French, Spanish, Chinese, etc.)

**Example**:
```python
# French input
result = assistant.chat("Créer un réseau de neurones pour la classification d'images")
```

### 3. LLM Integration
Support for multiple LLM providers:
- **OpenAI** (GPT-4, GPT-3.5) - Requires API key
- **Anthropic** (Claude) - Requires API key
- **Ollama** (Local LLMs) - Free, runs locally

### 4. Rule-Based Fallback
Works without LLM using intelligent pattern matching.

## Architecture

```
Natural Language Input
    ↓
Multi-Language Processing (detect & translate)
    ↓
Intent Extraction (LLM or rule-based)
    ↓
DSL Generation
    ↓
Validation & Output
```

## Usage

### Basic Usage (Rule-Based)
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST")
```

### With LLM (Requires API key or Ollama)
```python
# Auto-select available provider
assistant = NeuralAIAssistant(use_llm=True)

# Or specify provider
assistant = NeuralAIAssistant(use_llm=True, llm_provider='ollama')
```

### Integration with Chat Interface
The AI assistant is already integrated into `neural/neural_chat/neural_chat.py`:

```python
from neural.neural_chat.neural_chat import NeuralChat

chat = NeuralChat(use_ai=True)
response = chat.process_command("Create a CNN for image classification")
```

## Installation

### Optional Dependencies

For LLM support, install one of:
```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Ollama (local)
# Install from https://ollama.ai
# Then install Python client:
pip install ollama

# Language detection
pip install langdetect

# Translation
pip install googletrans
# or
pip install deep-translator
```

## Future Enhancements

- [ ] Enhanced LLM prompts for better DSL generation
- [ ] Context-aware conversation (remember previous commands)
- [ ] Multi-turn model building
- [ ] Automatic error correction
- [ ] Architecture suggestions based on data
- [ ] Performance optimization recommendations

## Files

- `natural_language_processor.py` - Intent extraction and DSL generation
- `llm_integration.py` - LLM provider abstraction
- `multi_language.py` - Language detection and translation
- `ai_assistant.py` - Main AI assistant interface
