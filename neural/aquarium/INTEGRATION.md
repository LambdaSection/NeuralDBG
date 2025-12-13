# Neural Aquarium - Integration Guide

This guide explains how Neural Aquarium integrates with the Neural DSL ecosystem.

## Overview

Neural Aquarium is a web-based AI assistant that generates Neural DSL code through natural language conversation. It integrates seamlessly with existing Neural DSL tools and workflows.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Neural Aquarium                        │
│               (AI Assistant Web UI)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Generates DSL Files
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   Neural DSL Files                       │
│                  (*.neural format)                       │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Used by
                   ▼
┌─────────────────────────────────────────────────────────┐
│                 Neural DSL Tools                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │   CLI     │  │   Parser  │  │  Compiler │          │
│  └───────────┘  └───────────┘  └───────────┘          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │Visualizer │  │  Trainer  │  │ NeuralDbg │          │
│  └───────────┘  └───────────┘  └───────────┘          │
└─────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Neural AI Modules

Neural Aquarium directly uses the following modules:

#### neural/ai/ai_assistant.py
```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=True)
result = assistant.chat("Create a CNN for MNIST")
dsl_code = result['dsl_code']
```

**Features Used:**
- Natural language processing
- Intent extraction
- DSL code generation
- Conversation state management
- Multi-language support

#### neural/ai/natural_language_processor.py
```python
from neural.ai.natural_language_processor import (
    NaturalLanguageProcessor,
    DSLGenerator
)

nlp = NaturalLanguageProcessor()
intent, params = nlp.extract_intent("Add a dense layer with 128 units")

generator = DSLGenerator()
dsl = generator.generate_from_intent(intent, params)
```

**Features Used:**
- Intent detection (create_model, add_layer, etc.)
- Parameter extraction (layer sizes, activations)
- Layer type recognition
- DSL syntax generation

#### neural/ai/multi_language.py
```python
from neural.ai.multi_language import MultiLanguageSupport

multi_lang = MultiLanguageSupport()
result = multi_lang.process("Créer un CNN", target_lang='en')
translated = result['final']
```

**Features Used:**
- Language detection
- Text translation (12+ languages)
- Fallback handling

### 2. Neural DSL CLI

Generated DSL files work with all Neural CLI commands:

#### Compile
```bash
# Download DSL from Aquarium
# Save as model.neural

neural compile model.neural --backend tensorflow --output model.py
```

#### Run/Train
```bash
neural run model.neural --data dataset.yaml --epochs 10 --batch-size 32
```

#### Visualize
```bash
neural visualize model.neural --output architecture.png
```

#### Debug
```bash
neural debug model.neural --data test_data.yaml
```

### 3. Neural DSL Parser

All generated DSL is valid Neural DSL syntax:

```python
from neural.parser.dsl_parser import DSLParser

# Parse Aquarium-generated DSL
parser = DSLParser()
ast = parser.parse_file("model.neural")
```

### 4. Code Generation Backends

Generated models work with all backends:

```bash
# TensorFlow
neural compile model.neural --backend tensorflow

# PyTorch
neural compile model.neural --backend pytorch

# ONNX
neural compile model.neural --backend onnx
```

### 5. NeuralDbg Dashboard

Models can be debugged using NeuralDbg:

```bash
# Start NeuralDbg
python neural/dashboard/dashboard.py

# Debug Aquarium-generated model
neural debug model.neural --data data.yaml --dashboard
```

## Workflow Integration

### Workflow 1: Aquarium → CLI → Training

```bash
# 1. Create model in Aquarium
#    "Create a CNN for MNIST"
#    Download as model.neural

# 2. Compile with Neural CLI
neural compile model.neural --backend tensorflow --output mnist_model.py

# 3. Train the model
neural run model.neural --data mnist.yaml --epochs 10

# 4. Evaluate
python mnist_model.py evaluate --data test_data.yaml
```

### Workflow 2: Aquarium → Visualization → Refinement

```bash
# 1. Generate architecture in Aquarium
#    Download as architecture_v1.neural

# 2. Visualize
neural visualize architecture_v1.neural --output viz.png

# 3. Review visualization, then refine in Aquarium
#    "Add dropout with rate 0.5"
#    "Add batch normalization"
#    Download as architecture_v2.neural

# 4. Compare versions
diff architecture_v1.neural architecture_v2.neural
```

### Workflow 3: Aquarium → Debug → Optimize

```bash
# 1. Create model in Aquarium
#    Download as model.neural

# 2. Train and debug
neural debug model.neural --data train.yaml --dashboard

# 3. Identify bottlenecks in NeuralDbg

# 4. Refine in Aquarium based on insights
#    "Replace dense layers with fewer parameters"

# 5. Re-test
neural debug optimized_model.neural --data train.yaml
```

## API Integration

### Frontend → Backend

```typescript
// React component
import { AIService } from './services/AIService';

const aiService = new AIService('http://localhost:5000');

const result = await aiService.chat(
  "Create a CNN for CIFAR-10",
  { current_model: existingDSL }
);

console.log(result.dsl_code);
```

### Backend → AI Modules

```python
# Flask API
from neural.ai.ai_assistant import NeuralAIAssistant

@app.route('/api/ai/chat', methods=['POST'])
def chat():
    data = request.get_json()
    assistant = NeuralAIAssistant()
    result = assistant.chat(data['user_input'], data.get('context'))
    return jsonify(result)
```

## Data Flow

### Complete Flow: User Input → DSL File → Trained Model

```
1. User Input (Natural Language)
   ↓
2. Frontend (React) → AIService
   ↓
3. Backend API → POST /api/ai/chat
   ↓
4. NeuralAIAssistant.chat()
   ↓
5. NaturalLanguageProcessor.extract_intent()
   ↓
6. DSLGenerator.generate_from_intent()
   ↓
7. DSL Code (String) → Frontend
   ↓
8. User Downloads → model.neural file
   ↓
9. Neural CLI compile → model.py (TensorFlow/PyTorch)
   ↓
10. Neural CLI run → Trained Model
    ↓
11. Model Deployment
```

## File Format Compatibility

### DSL File Structure

Aquarium generates standard Neural DSL files:

```neural
network ModelName {
    input: (height, width, channels)
    layers:
        Conv2D(filters, (kernel_h, kernel_w), "activation")
        MaxPooling2D((pool_h, pool_w))
        Flatten()
        Dense(units, "activation")
        Output(num_classes, "activation")
    loss: "loss_function"
    optimizer: OptimizerName(learning_rate=value)
}
```

This format is compatible with:
- Neural DSL Parser
- All backend compilers
- Neural CLI tools
- NeuralDbg
- Visualization tools

## Environment Variables

### Frontend (.env)
```env
REACT_APP_API_URL=http://localhost:5000
```

### Backend (environment)
```bash
FLASK_ENV=development
FLASK_APP=api.py
PORT=5000
```

### Neural DSL (inherited)
All Neural DSL environment variables work:
```bash
NEURAL_BACKEND=tensorflow  # Default backend
NEURAL_CACHE_DIR=.neural_cache
NEURAL_CONFIG=config.yaml
```

## Testing Integration

### Unit Tests

```python
# Test AI assistant integration
from neural.ai.ai_assistant import NeuralAIAssistant

def test_dsl_generation():
    assistant = NeuralAIAssistant()
    result = assistant.chat("Create a CNN for MNIST")
    assert result['success']
    assert 'network' in result['dsl_code']
    assert 'Conv2D' in result['dsl_code']
```

### Integration Tests

```python
# Test full pipeline
def test_aquarium_to_cli():
    # Generate DSL
    assistant = NeuralAIAssistant()
    result = assistant.chat("Create a CNN for MNIST")
    
    # Save to file
    with open('test_model.neural', 'w') as f:
        f.write(result['dsl_code'])
    
    # Parse with Neural DSL
    parser = DSLParser()
    ast = parser.parse_file('test_model.neural')
    
    assert ast is not None
```

### End-to-End Tests

```bash
# Test complete workflow
cd neural/aquarium

# Start backend
python backend/api.py &

# Run frontend tests
npm test

# Test CLI integration
neural compile test_model.neural --backend tensorflow
```

## Extension Points

### Custom Intent Types

Add new intents in `natural_language_processor.py`:

```python
class IntentType(Enum):
    CREATE_MODEL = "create_model"
    ADD_LAYER = "add_layer"
    # Add custom intent
    CUSTOM_INTENT = "custom_intent"
```

### Custom Layer Types

Extend DSL generator:

```python
class DSLGenerator:
    def _generate_layer(self, params):
        # Add custom layer support
        if layer_type == 'custom_layer':
            return f"CustomLayer({params})\n"
```

### Custom Backends

Generated DSL works with custom backends:

```python
from neural.code_generation.custom_backend import CustomBackend

backend = CustomBackend()
code = backend.generate(ast)
```

## Best Practices

### 1. Version Control
- Commit generated DSL files
- Track model iterations
- Use semantic versioning

### 2. Documentation
- Document model architecture decisions
- Include conversation context
- Add comments to DSL files

### 3. Validation
- Parse DSL before training
- Validate layer compatibility
- Test with sample data

### 4. Optimization
- Profile generated models
- Use NeuralDbg for bottlenecks
- Iterate in Aquarium based on insights

### 5. Collaboration
- Share DSL files with team
- Use multi-language support
- Review generated code together

## Troubleshooting Integration Issues

### Issue: DSL Parse Errors

**Problem**: Generated DSL won't parse
**Solution**: 
- Check syntax in Aquarium viewer
- Validate with `neural compile --check-only`
- Report generation bugs

### Issue: Backend Compatibility

**Problem**: Model won't compile for specific backend
**Solution**:
- Check backend-specific layer support
- Use compatible layer types
- Consult backend documentation

### Issue: API Connection

**Problem**: Frontend can't reach backend
**Solution**:
- Verify backend is running: `curl http://localhost:5000/health`
- Check CORS settings
- Review firewall rules

## Future Integration Plans

1. **Direct Compilation**: Compile DSL in Aquarium UI
2. **Live Preview**: Real-time visualization of architecture
3. **Training Integration**: Train models directly from UI
4. **Model Zoo**: Save and share model templates
5. **Cloud Integration**: Deploy models to cloud platforms
6. **Collaborative Editing**: Real-time multi-user model building

## Resources

- Neural DSL Documentation: See main README.md
- API Reference: See `backend/api.py` docstrings
- Component Documentation: See `ARCHITECTURE.md`
- Examples: See `EXAMPLES.md`

## Support

For integration issues:
1. Check logs (frontend console, backend terminal)
2. Verify DSL syntax
3. Test with Neural CLI tools
4. Report issues on GitHub
