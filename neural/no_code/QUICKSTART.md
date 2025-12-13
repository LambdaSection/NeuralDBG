# Quick Start Guide - Neural DSL No-Code Designer

Get started building neural networks visually in under 5 minutes!

## Installation

```bash
# Install with dashboard dependencies
pip install -e ".[dashboard]"
```

## Launch the Designer

```bash
# Modern React Flow Interface (Recommended)
python neural/no_code/app.py

# Or use the launcher
python neural/no_code/launcher.py --ui react
```

Open your browser to **http://localhost:8051**

## Your First Model (2 Minutes)

### Option 1: Use a Template

1. Click **📋 Templates** button
2. Select **"MNIST CNN"**
3. Model loads instantly with 8 pre-configured layers
4. Click **🚀 Export Code** to generate code

### Option 2: Build from Scratch

1. **Drag a layer** from the palette (left sidebar) onto the canvas
2. **Connect layers** by dragging from one layer's output to another's input
3. **Edit parameters** by clicking a layer and using the Properties tab
4. **Validate** automatically as you build (bottom panel)
5. **Export** your model when ready

## Example: Build a Simple CNN

### Step 1: Add Input Processing
Drag these layers onto canvas:
- **Conv2D** (filters: 32, kernel_size: [3,3])
- **MaxPooling2D** (pool_size: [2,2])

### Step 2: Add More Layers
- **Conv2D** (filters: 64, kernel_size: [3,3])
- **MaxPooling2D** (pool_size: [2,2])

### Step 3: Add Classification Head
- **Flatten**
- **Dense** (units: 128, activation: relu)
- **Dropout** (rate: 0.5)
- **Dense** (units: 10, activation: softmax)

### Step 4: Connect & Validate
Connect layers in sequence from top to bottom. The validation panel will show any errors.

### Step 5: Export Code
Click **🚀 Export Code** and choose:
- **Neural DSL** - Native DSL code
- **TensorFlow** - Keras model
- **PyTorch** - torch.nn model

## Interactive Tutorial

Click **📖 Tutorial** for a guided walkthrough covering:
- Layer palette navigation
- Drag and drop
- Properties editing
- Validation system
- Code export

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Delete** | Remove selected layer |
| **Escape** | Deselect all |
| **Click + Drag** | Pan canvas |
| **Scroll** | Zoom in/out |

## Common Workflows

### Load → Modify → Export
```
1. Click "📋 Templates"
2. Select "CIFAR-10 VGG"
3. Click a layer to edit parameters
4. Export code when satisfied
```

### Build → Save → Resume Later
```
1. Build your model
2. (Coming soon: Click "💾 Save")
3. Give it a name
4. Load later from saved models
```

### Validate → Fix → Export
```
1. Build model
2. Check validation panel
3. Fix any red-highlighted layers
4. Export when validation passes
```

## Layer Palette Overview

### Most Used Layers

**Convolutional**
- Conv2D: Standard 2D convolution
- SeparableConv2D: Depthwise separable convolution

**Core**
- Dense: Fully connected layer
- Flatten: Flatten multi-dimensional input
- Dropout: Regularization layer

**Normalization**
- BatchNormalization: Normalize batch activations
- LayerNormalization: Normalize layer activations

**Pooling**
- MaxPooling2D: Max pooling
- GlobalAveragePooling2D: Global average pooling

### Search Feature
Type in the search box to filter layers:
- "conv" → All convolution layers
- "pool" → All pooling layers
- "lstm" → LSTM and related layers

## Tips & Tricks

### 1. Use Templates as Starting Points
Start with a template and modify it rather than building from scratch.

### 2. Real-Time Validation
Watch the validation panel as you build. Fix errors immediately.

### 3. Shape Information
Hover over layers to see tensor shapes flowing through the network.

### 4. Parameter Editing
Click a layer → Properties tab → Edit values → Press Enter

### 5. Organize Your Canvas
Use the MiniMap (bottom-right) to navigate large models.

### 6. Connect Efficiently
Drag from output port (bottom) to input port (top) of next layer.

## Troubleshooting

### Layer won't connect
- Check shape compatibility
- Verify layer order (input → output direction)
- Look for validation errors

### Code generation fails
- Ensure all layers have valid parameters
- Check that model passes validation
- Review error messages in export modal

### Can't find a layer
- Use search box in layer palette
- Check correct category is expanded
- Scroll through the palette

## Next Steps

### Learn More
- Read the [full README](README.md) for detailed documentation
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for advanced features
- Try [example scripts](examples/) for API usage

### Build Advanced Models
- Explore all templates
- Combine multiple layer types
- Experiment with different architectures

### Export & Train
1. Export TensorFlow or PyTorch code
2. Add data loading code
3. Train your model
4. Evaluate performance

## Examples to Try

### 1. Image Classification (MNIST)
Template: "MNIST CNN"
- Quick start for digit recognition
- 8 layers, simple architecture
- Perfect for learning

### 2. Image Classification (CIFAR-10)
Template: "CIFAR-10 VGG"
- More complex, VGG-style
- 12 layers with blocks
- Good for RGB images

### 3. Text Classification
Template: "Text LSTM"
- Sequence processing
- Embedding + LSTM layers
- Sentiment analysis ready

### 4. Transformer (NLP)
Template: "Transformer Encoder"
- Attention mechanism
- Modern architecture
- Advanced NLP tasks

### 5. Image Classification (ImageNet)
Template: "ResNet Block"
- Residual connections
- 11 layers
- State-of-the-art architecture

## API Usage (Advanced)

Use the REST API programmatically:

```python
import requests

# Validate a model
response = requests.post('http://localhost:8051/api/validate', json={
    'input_shape': [None, 28, 28, 1],
    'layers': [
        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': [3, 3]}}
    ]
})
print(response.json())
```

See [examples/example_usage.py](examples/example_usage.py) for complete examples.

## Getting Help

### Resources
- **Tutorial**: Built-in interactive tutorial (📖 button)
- **Documentation**: Full README and guides
- **Examples**: Code examples in `examples/` folder
- **Issues**: GitHub issues for bugs/features

### Common Questions

**Q: Can I export to ONNX?**
A: Currently supports Neural DSL, TensorFlow, and PyTorch. ONNX support coming soon.

**Q: Can I train models in the interface?**
A: No, export code and train using TensorFlow/PyTorch. Training UI is a future feature.

**Q: Does it support custom layers?**
A: Not yet, but you can modify exported code to add custom layers.

**Q: Can multiple people edit the same model?**
A: Not currently. Collaborative editing is a planned feature.

**Q: Is my data secure?**
A: Models are saved locally. We don't collect or transmit your data.

## Support

Need help?
- 📖 Read the docs
- 💬 Ask in GitHub discussions
- 🐛 Report bugs via GitHub issues
- 📧 Email: support@example.com

## What's Next?

Future features in development:
- ✅ Undo/Redo
- ✅ Keyboard shortcuts
- ✅ Model versioning
- ✅ Collaborative editing
- ✅ Custom layers
- ✅ Training interface
- ✅ ONNX export
- ✅ Cloud integration

---

**Ready to build?** Launch the designer and start creating! 🚀

```bash
python neural/no_code/app.py
```
