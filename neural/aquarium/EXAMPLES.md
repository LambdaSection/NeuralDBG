# Neural Aquarium - Usage Examples

## Basic Examples

### Creating a Simple CNN

**Prompt**:
```
Create a CNN for MNIST classification with 10 classes
```

**Generated DSL**:
```neural
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
```

### Creating a Custom CNN

**Conversation**:
```
User: Create a CNN for CIFAR-10
Assistant: I've created a CNN model for you...

User: Add a convolutional layer with 64 filters
Assistant: Added layer: Conv2D(64, (3, 3), "relu")

User: Add dropout with rate 0.3
Assistant: Added layer: Dropout(0.3)

User: Set optimizer to SGD with learning rate 0.01
Assistant: Optimizer set to SGD(learning_rate=0.01)
```

**Final DSL**:
```neural
network MyModel {
    input: (32, 32, 3)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Dropout(0.3)
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Sgd(learning_rate=0.01)
}
```

## Advanced Examples

### Deep CNN with Batch Normalization

**Prompts**:
1. "Create a deep CNN for image classification with 100 classes"
2. "Add a conv layer with 128 filters and 3x3 kernel"
3. "Add batch normalization"
4. "Add max pooling with pool size 2"
5. "Add another conv layer with 256 filters"
6. "Add dropout with rate 0.5"

**Generated DSL**:
```neural
network DeepCNN {
    input: (224, 224, 3)
    layers:
        Conv2D(64, (3, 3), "relu")
        BatchNormalization()
        MaxPooling2D((2, 2))
        Conv2D(128, (3, 3), "relu")
        BatchNormalization()
        MaxPooling2D((2, 2))
        Conv2D(256, (3, 3), "relu")
        BatchNormalization()
        MaxPooling2D((2, 2))
        Dropout(0.5)
        Flatten()
        Dense(512, "relu")
        Dropout(0.5)
        Dense(256, "relu")
        Output(100, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

### ResNet-style Architecture

**Prompt**:
```
Create a residual network with skip connections for image classification
```

**Generated DSL** (conceptual):
```neural
network ResNet {
    input: (224, 224, 3)
    layers:
        Conv2D(64, (7, 7), "relu", strides=(2, 2))
        MaxPooling2D((3, 3), strides=(2, 2))
        
        # Residual Block 1
        ResidualBlock(64, 3)
        ResidualBlock(64, 3)
        
        # Residual Block 2
        ResidualBlock(128, 3, downsample=True)
        ResidualBlock(128, 3)
        
        # Residual Block 3
        ResidualBlock(256, 3, downsample=True)
        ResidualBlock(256, 3)
        
        GlobalAveragePooling2D()
        Output(1000, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

### Recurrent Network (LSTM)

**Prompt**:
```
Create an LSTM network for text classification with 5 classes
```

**Generated DSL**:
```neural
network TextLSTM {
    input: (100, 300)  # sequence_length, embedding_dim
    layers:
        LSTM(128, return_sequences=True)
        Dropout(0.3)
        LSTM(64)
        Dropout(0.3)
        Dense(32, "relu")
        Output(5, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

## Multi-Language Examples

### French

**Prompt**:
```
Créer un réseau de neurones convolutif pour la classification d'images avec 10 classes
```

**Response (in French)**:
```
J'ai créé un modèle CNN pour vous...
```

**Generated DSL** (same as English):
```neural
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
```

### Spanish

**Prompt**:
```
Añadir una capa densa con 256 unidades
```

**Response (in Spanish)**:
```
Capa añadida: Dense(256, "relu")
```

### Chinese

**Prompt**:
```
创建一个用于图像分类的卷积神经网络
```

**Response (in Chinese)**:
```
我为您创建了一个CNN模型...
```

## Integration Examples

### Using Generated DSL with Neural CLI

```bash
# Save DSL to file
# (Click "Download" in DSL Code Viewer)

# Compile the model
neural compile model.neural --backend tensorflow --output model.py

# Run training
neural run model.neural --data mnist.yaml --epochs 10

# Visualize architecture
neural visualize model.neural --output architecture.png
```

### Programmatic Usage

```python
from neural.ai.ai_assistant import NeuralAIAssistant

# Initialize assistant
assistant = NeuralAIAssistant(use_llm=True)

# Chat with assistant
result = assistant.chat("Create a CNN for MNIST")

# Get DSL code
dsl_code = result['dsl_code']
print(dsl_code)

# Continue conversation
result = assistant.chat("Add a dense layer with 256 units")

# Get updated model
full_model = assistant.get_current_model()
```

### REST API Usage

```bash
# Send chat message
curl -X POST http://localhost:5000/api/ai/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Create a CNN for MNIST",
    "language": "en"
  }'

# Get current model
curl http://localhost:5000/api/ai/current-model

# Reset assistant
curl -X POST http://localhost:5000/api/ai/reset

# Translate text
curl -X POST http://localhost:5000/api/ai/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Create a neural network",
    "target_lang": "fr"
  }'
```

## Common Workflows

### Workflow 1: Quick Prototyping

1. "Create a CNN for image classification"
2. Review generated DSL
3. Click "Apply" to use in workspace
4. Download DSL file
5. Compile and train with Neural CLI

### Workflow 2: Iterative Development

1. "Create a CNN for CIFAR-10"
2. "Add a convolutional layer with 128 filters"
3. "Add batch normalization"
4. "Add dropout with rate 0.4"
5. "Set optimizer to RMSprop with learning rate 0.0005"
6. Review and edit DSL manually if needed
7. Click "Apply"

### Workflow 3: Learning & Exploration

1. Ask: "What's a good architecture for image classification?"
2. Ask: "How do I prevent overfitting?"
3. Ask: "Add dropout and batch normalization"
4. Review generated code
5. Experiment with different architectures

### Workflow 4: Multi-Language Collaboration

1. Team member in France: "Créer un CNN"
2. Team member in Spain: "Añadir capa de dropout"
3. Team member in China: "设置优化器为Adam"
4. All see same DSL code in English
5. Download and share model file

## Tips & Tricks

### Getting Better Results

1. **Be Specific**: "Add a conv layer with 64 filters and 3x3 kernel" is better than "Add a layer"

2. **Use Context**: "Add another conv layer with twice the filters" (assumes context from previous layers)

3. **Iterate**: Start simple, then refine

4. **Review & Edit**: Always review generated code, edit if needed

### Common Patterns

**Pattern**: Building block by block
```
"Create a CNN" → "Add conv layer" → "Add pooling" → "Add dropout" → etc.
```

**Pattern**: Complete specification
```
"Create a CNN with 3 conv layers (32, 64, 128 filters), max pooling after each, dropout 0.5, and dense layer with 256 units"
```

**Pattern**: Reference architectures
```
"Create a VGG-style network" or "Create a ResNet-inspired architecture"
```

### Language Selection

- Select language at top of sidebar
- Assistant responds in selected language
- DSL code always in English (standard syntax)
- Translations preserve technical accuracy

### Code Editing

- Click "Edit" to manually modify DSL
- Make changes directly in editor
- Click "Save" to apply changes
- Click "Cancel" to discard edits

## Troubleshooting

### "I didn't understand that"

**Solution**: Be more specific or use simpler language

**Instead of**: "Make it better"
**Try**: "Add dropout with rate 0.5"

### Layer not recognized

**Solution**: Use standard layer names

**Supported**: Conv2D, Dense, Dropout, MaxPooling2D, Flatten, LSTM, GRU, BatchNormalization

### Wrong input shape

**Solution**: Specify in creation prompt

"Create a CNN for images of size 224x224 with 3 channels"

### Translation issues

**Solution**: 
- Switch to English for best results
- Use technical terms in English even in other languages
- Report translation issues

## Example Sessions

### Complete Session: Image Classifier

```
User: Create a CNN for fashion item classification with 10 categories