# Rapid Prototyping Guide: From Idea to Model in Minutes

This guide shows you how to use Neural DSL for rapid prototyping - quickly testing ideas, comparing architectures, and iterating on designs.

## Why Neural DSL for Prototyping?

**Speed:** Define models in seconds, not hours  
**Flexibility:** Switch frameworks with one flag  
**Validation:** Catch errors before training  
**Comparison:** Test multiple architectures easily  

## Quick Start Templates

### Template 1: Image Classification

```neural
network ImageClassifier {
    input: (224, 224, 3)
    layers:
        Conv2D(32, (3,3), "relu") * 2
        MaxPooling2D((2,2))
        Conv2D(64, (3,3), "relu") * 2
        MaxPooling2D((2,2))
        Conv2D(128, (3,3), "relu") * 2
        MaxPooling2D((2,2))
        Flatten()
        Dense(512, "relu")
        Dropout(0.5)
        Output(1000, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

**Quick adaptations:**
- Small images: Change input to `(32, 32, 3)`
- Binary classification: Change output to `Output(1, "sigmoid")` and loss to `"binary_crossentropy"`
- More classes: Change output to `Output(N, "softmax")`

### Template 2: Text Classification

```neural
network TextClassifier {
    input: (500,)  # Sequence length
    layers:
        Embedding(10000, 128)  # vocab_size, embedding_dim
        LSTM(64, return_sequences=True)
        LSTM(64)
        Dense(64, "relu")
        Dropout(0.5)
        Output(3, "softmax")  # 3 classes
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

**Quick adaptations:**
- Longer sequences: Increase input `(1000,)`
- Different vocabulary: Change embedding `(20000, 128)`
- Binary sentiment: Change to `Output(1, "sigmoid")`

### Template 3: Time Series Forecasting

```neural
network TimeSeriesForecaster {
    input: (100, 10)  # timesteps, features
    layers:
        LSTM(64, return_sequences=True)
        LSTM(32)
        Dense(32, "relu")
        Dense(1)  # Single value prediction
    loss: "mse"
    optimizer: Adam(learning_rate=0.001)
}
```

**Quick adaptations:**
- Multivariate output: Change to `Dense(N)` for N outputs
- Add attention: Insert `Attention()` after LSTM layers
- Deeper network: Add more LSTM layers with `return_sequences=True`

### Template 4: Simple Transformer

```neural
network TransformerModel {
    input: (512,)
    layers:
        Embedding(10000, 256)
        PositionalEncoding(256)
        TransformerEncoder(num_heads=8, ff_dim=512) * 3
        GlobalAveragePooling1D()
        Dense(128, "relu")
        Output(2, "softmax")
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.0001)
}
```

## Rapid Iteration Workflow

### Step 1: Start with Template

```bash
# Copy template to your project
cp templates/image_classifier.neural my_model.neural
```

### Step 2: Quick Modifications

Edit just the essentials:
```neural
# Change these three things:
input: (your_image_size)
# Adjust layers as needed
Output(your_num_classes, "softmax")
```

### Step 3: Instant Validation

```bash
# Validates in < 1 second
neural compile my_model.neural --dry-run
```

### Step 4: Visualize

```bash
# See your architecture instantly
neural visualize my_model.neural --format html
# Opens in browser automatically
```

### Step 5: Compare Backends

```bash
# Try different frameworks
neural compile my_model.neural --backend tensorflow --output tf_model.py
neural compile my_model.neural --backend pytorch --output torch_model.py

# Compare generated code
diff tf_model.py torch_model.py
```

## A/B Testing Architectures

### Test Multiple Variants Quickly

Create variants:

```bash
# Base model
cp my_model.neural variant_a.neural

# Deeper model
cp my_model.neural variant_b.neural
# Edit: Add more layers

# Wider model
cp my_model.neural variant_c.neural
# Edit: Increase layer sizes

# With normalization
cp my_model.neural variant_d.neural
# Edit: Add BatchNormalization
```

Validate all at once:

```bash
for model in variant_*.neural; do
    echo "Testing $model..."
    neural compile $model --dry-run
    neural visualize $model --format png
done
```

Compare architectures side by side:

```bash
# Visual comparison
neural compare variant_a.neural variant_b.neural variant_c.neural
```

## HPO for Quick Exploration

Instead of manually trying hyperparameters, let Neural find them:

```neural
network AutoTunedClassifier {
    input: (28, 28, 1)
    layers:
        Conv2D(HPO(choice(16, 32, 64)), (3,3), "relu")
        MaxPooling2D((2,2))
        Conv2D(HPO(choice(32, 64, 128)), (3,3), "relu")
        MaxPooling2D((2,2))
        Flatten()
        Dense(HPO(choice(64, 128, 256)), "relu")
        Dropout(HPO(range(0.2, 0.6, step=0.1)))
        Output(10, "softmax")
    
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.01)))
    
    train {
        epochs: 10
        batch_size: HPO(choice(32, 64, 128))
        search_method: "bayesian"
    }
}
```

Run quick HPO sweep:

```bash
neural compile my_model.neural --hpo --backend tensorflow
```

This finds good hyperparameters automatically in the background.

## Prototyping Best Practices

### 1. Start Simple, Add Complexity

```neural
# Start here (simple)
network MyModel {
    input: (28, 28, 1)
    layers:
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}

# Then add (if needed)
# - Convolutional layers
# - Dropout
# - Batch normalization
# - More dense layers
```

**Why:** Simpler models train faster, debug easier, and often work surprisingly well.

### 2. Use Layer Repetition

Instead of:
```neural
Conv2D(32, (3,3), "relu")
Conv2D(32, (3,3), "relu")
Conv2D(32, (3,3), "relu")
```

Write:
```neural
Conv2D(32, (3,3), "relu") * 3
```

**Benefit:** Faster to write, easier to modify.

### 3. Template Common Blocks

Save reusable blocks as macros:

```neural
macro ConvBlock(filters) {
    Conv2D(filters, (3,3), "relu")
    BatchNormalization()
    MaxPooling2D((2,2))
}

network MyModel {
    input: (224, 224, 3)
    layers:
        ConvBlock(32)
        ConvBlock(64)
        ConvBlock(128)
        Flatten()
        Dense(512, "relu")
        Output(1000, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

### 4. Quick Shape Debugging

If uncertain about shapes:

```bash
# This shows exact shape at each layer
neural debug my_model.neural --show-shapes
```

Or add explicit assertions in DSL:

```neural
layers:
    Conv2D(32, (3,3), "relu")
    # Assert expected shape
    assert_shape: (None, 26, 26, 32)
    MaxPooling2D((2,2))
```

### 5. Framework-Agnostic Development

Develop without committing to a framework:

```neural
# This model works with ANY backend
network FlexibleModel {
    input: (224, 224, 3)
    layers:
        # Standard layers work everywhere
        Conv2D(32, (3,3), "relu")
        MaxPooling2D((2,2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

Compile to any target:

```bash
neural compile model.neural --backend tensorflow
neural compile model.neural --backend pytorch
neural compile model.neural --backend onnx
```

## Common Prototyping Patterns

### Pattern: Residual Connections

```neural
network ResNetStyle {
    input: (224, 224, 3)
    layers:
        Conv2D(64, (7,7), strides=2, "relu")
        MaxPooling2D((3,3), strides=2)
        
        # Residual block
        ResidualBlock {
            Conv2D(64, (3,3), "relu")
            Conv2D(64, (3,3))
        }
        
        GlobalAveragePooling2D()
        Output(1000, "softmax")
    loss: "categorical_crossentropy"
    optimizer: SGD(learning_rate=0.1, momentum=0.9)
}
```

### Pattern: Encoder-Decoder

```neural
network EncoderDecoder {
    input: (256, 256, 3)
    
    # Encoder (downsampling)
    layers:
        Conv2D(32, (3,3), "relu") @ encoder
        MaxPooling2D((2,2))
        Conv2D(64, (3,3), "relu") @ encoder
        MaxPooling2D((2,2))
        Conv2D(128, (3,3), "relu") @ encoder
        
        # Decoder (upsampling)
        TransposedConv2D(64, (3,3), "relu")
        TransposedConv2D(32, (3,3), "relu")
        Conv2D(3, (1,1), "sigmoid")  # Output image
    
    loss: "mse"
    optimizer: Adam(learning_rate=0.001)
}
```

### Pattern: Multi-Input Model

```neural
network MultiInputModel {
    # Two separate inputs
    input_image: (224, 224, 3)
    input_metadata: (10,)
    
    # Image branch
    branch image_features from input_image:
        Conv2D(32, (3,3), "relu")
        Flatten()
        Dense(128, "relu")
    
    # Metadata branch
    branch meta_features from input_metadata:
        Dense(64, "relu")
        Dense(64, "relu")
    
    # Combine branches
    layers:
        Concatenate([image_features, meta_features])
        Dense(256, "relu")
        Output(10, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

## Debugging Prototypes

### Quick Checks

```bash
# 1. Syntax check (instant)
neural compile model.neural --dry-run

# 2. Visualize (< 1 second)
neural visualize model.neural --format png

# 3. Shape propagation (instant)
neural debug model.neural --show-shapes

# 4. Parameter count
neural analyze model.neural --params
```

### Common Issues and Quick Fixes

**Issue:** Shape mismatch after Flatten  
**Fix:** Check Conv2D output dimensions, adjust input size

**Issue:** Too many parameters  
**Fix:** Reduce layer sizes or add more pooling

**Issue:** Not enough parameters  
**Fix:** Add layers or increase neurons per layer

**Issue:** Model too slow  
**Fix:** Reduce input size, use fewer layers, or add GPU

## Time-Saving Tools

### 1. Model Generator

Generate models from natural language:

```python
from neural.ai import generate_model

model = generate_model("""
Create a CNN for classifying chest X-rays into 3 categories.
Use transfer learning with ResNet-like architecture.
Input size: 512x512 grayscale images.
""")

with open("xray_classifier.neural", "w") as f:
    f.write(model)
```

### 2. Template Library

Access pre-built templates:

```bash
# List available templates
neural templates list

# Use a template
neural templates use resnet50 --output my_resnet.neural

# Customize template
neural templates use vgg16 --num_classes 10 --input_size 224
```

### 3. Batch Operations

Process multiple models:

```python
from neural.prototyping import BatchProcessor

processor = BatchProcessor()

# Test variants
variants = [
    "model_v1.neural",
    "model_v2.neural",
    "model_v3.neural"
]

results = processor.validate_all(variants)
processor.compare_models(variants)
processor.generate_report("comparison.html")
```

### 4. Interactive Mode

Quick experimentation in Python:

```python
from neural import ModelBuilder

# Build model interactively
model = ModelBuilder()
model.input((28, 28, 1))
model.conv2d(32, (3, 3), activation="relu")
model.maxpool2d((2, 2))
model.flatten()
model.dense(128, activation="relu")
model.output(10, activation="softmax")

# Configure training
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="Adam"
)

# Export to DSL
model.to_dsl("my_model.neural")

# Or directly to code
model.to_tensorflow("tf_model.py")
```

## Prototyping Checklist

Before moving to full training:

- [ ] Model compiles without errors
- [ ] Shapes propagate correctly (check with `neural debug`)
- [ ] Parameter count is reasonable for your data
- [ ] Architecture makes sense visually (check `neural visualize`)
- [ ] Tested on both TensorFlow and PyTorch (if targeting both)
- [ ] HPO exploration completed (if using)
- [ ] Compared with baseline/alternative architectures

## Next Steps

Once your prototype works:

1. **Scale Up:** Use full dataset and more epochs
2. **Optimize:** Fine-tune hyperparameters
3. **Deploy:** Export to production format
4. **Monitor:** Add experiment tracking

---

**Remember:** The goal of prototyping is to **fail fast and learn quickly**. Don't spend hours on a single model - try many variations and see what works!

For more examples, see the `examples/` directory and `templates/` for ready-to-use architectures.

Happy prototyping! 🚀
