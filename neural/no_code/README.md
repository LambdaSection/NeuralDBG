# Neural No-Code Interface

<p align="center">
  <img src="../../docs/images/no_code_interface.png" alt="No-Code Interface" width="800"/>
</p>

The Neural No-Code Interface provides two powerful ways to build neural networks without writing code: a modern React Flow-based visual designer and a classic Dash interface.

## Features

### 🎨 Modern React Flow Interface (New!)

- **Drag-and-Drop Visual Designer**: Build networks by dragging layers onto a canvas with real-time connections
- **Layer Palette with Categories & Search**: Browse and search 70+ layers organized by type
- **Real-Time Shape Validation**: Automatic tensor shape propagation with error highlighting
- **Interactive Properties Panel**: Edit layer parameters with instant validation feedback
- **Model Templates Gallery**: Quick-start with MNIST CNN, VGG, LSTM, Transformer, ResNet templates
- **Interactive Tutorial System**: Step-by-step guided workflows for beginners
- **Visual Flow Editor**: Connect layers with visual edges to see data flow
- **Modern UI Components**: Clean, dark-themed interface with smooth animations
- **Multi-Backend Export**: Generate Neural DSL, TensorFlow, or PyTorch code

### 📊 Classic Dash Interface

- **Intuitive Model Building**: Tab-based interface for layer selection
- **Layer Configuration**: Configure parameters with editable tables
- **Shape Propagation Visualization**: Interactive charts showing tensor shapes
- **Code Generation**: Generate code for multiple backends
- **Model Management**: Save and load models
- **NeuralDbg Integration**: Launch real-time debugger
- **Dark Theme**: Modern, eye-friendly interface

## Quick Start

### Launch Modern Interface (Recommended)

```bash
python neural/no_code/launcher.py --ui react
# or directly:
python neural/no_code/app.py
```

### Launch Classic Interface

```bash
python neural/no_code/launcher.py --ui dash
# or directly:
python neural/no_code/no_code.py
```

Then open http://localhost:8051 in your browser.

### Using the CLI (if configured)

```bash
neural no-code         # Launch default (React) interface
neural no-code --classic  # Launch classic Dash interface
```

## Building Your First Model (React Flow Interface)

### 1. Start with a Template (Optional)

Click **📋 Templates** and choose:
- **MNIST CNN**: Digit classification (28×28×1)
- **CIFAR-10 VGG**: Image classification (32×32×3)
- **Text LSTM**: Sequence processing
- **Transformer**: NLP tasks with attention
- **ResNet Block**: Image classification with residual connections

### 2. Drag and Drop Layers

1. Browse the **Layer Palette** on the left sidebar
2. Use the **search box** to find specific layers
3. **Drag** a layer from the palette onto the canvas
4. **Connect** layers by dragging from output port to input port
5. Layers auto-validate shapes as you connect them

### 3. Configure Layer Properties

1. **Click** any layer on the canvas
2. Switch to the **Properties** tab in the sidebar
3. Edit parameters (filters, kernel_size, activation, etc.)
4. Changes apply **instantly** with validation

### 4. Validate Your Architecture

- **Validation panel** at the bottom shows errors/warnings
- Invalid layers are **highlighted in red**
- Hover over layers to see tensor shapes
- Fix errors before exporting

### 5. Export Code

1. Click **🚀 Export Code**
2. Choose format:
   - **Neural DSL**: Native DSL code
   - **TensorFlow**: Keras model code
   - **PyTorch**: torch.nn model code
3. **Copy** code to clipboard or download

### 6. Interactive Tutorial

Click **📖 Tutorial** for a guided walkthrough with:
- Welcome and overview
- Layer palette usage
- Canvas operations
- Properties editing
- Template loading
- Validation understanding
- Code export

## Layer Categories

### Convolutional
Conv1D, Conv2D, Conv3D, SeparableConv2D, DepthwiseConv2D, TransposedConv2D

### Pooling
MaxPooling (1D/2D), AveragePooling (1D/2D), GlobalMaxPooling (1D/2D), GlobalAveragePooling (1D/2D)

### Core
Dense, Flatten, Reshape, Permute, RepeatVector

### Normalization
BatchNormalization, LayerNormalization, GroupNormalization

### Regularization
Dropout, SpatialDropout (1D/2D), GaussianNoise, GaussianDropout

### Recurrent
LSTM, GRU, SimpleRNN, Bidirectional, ConvLSTM2D

### Attention
MultiHeadAttention, Attention

### Embedding
Embedding

### Activation
ReLU, LeakyReLU, PReLU, ELU, Softmax, Sigmoid, Tanh

## Model Templates

| Template | Input Shape | Use Case | Layers |
|----------|-------------|----------|---------|
| **MNIST CNN** | (None, 28, 28, 1) | Digit classification | 8 layers with Conv2D, MaxPooling, Dense |
| **CIFAR-10 VGG** | (None, 32, 32, 3) | Image classification | 12 layers with VGG-style blocks |
| **Text LSTM** | (None, 100) | Text classification | 6 layers with Embedding, stacked LSTM |
| **Transformer** | (None, 512) | NLP tasks | 9 layers with attention mechanism |
| **ResNet Block** | (None, 224, 224, 3) | ImageNet classification | 11 layers with residual connections |

## API Reference (React Flow Interface)

### REST Endpoints

```
GET  /api/layers          - Get all layer types by category
GET  /api/templates       - Get model templates
GET  /api/tutorial        - Get tutorial steps
POST /api/validate        - Validate model and propagate shapes
POST /api/generate-code   - Generate code for all backends
POST /api/save            - Save model to JSON
GET  /api/load/<name>     - Load saved model
GET  /api/models          - List all saved models
```

### Validation API

**Request:**
```json
{
  "input_shape": [null, 28, 28, 1],
  "layers": [
    {"id": "node-1", "type": "Conv2D", "params": {"filters": 32, "kernel_size": [3, 3]}}
  ]
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "shapes": [
    {"layer": "Input", "shape": [null, 28, 28, 1]},
    {"layer": "Conv2D", "shape": [null, 26, 26, 32]}
  ]
}
```

### Code Generation API

**Request:**
```json
{
  "input_shape": [null, 28, 28, 1],
  "layers": [...],
  "optimizer": {"type": "Adam", "params": {"learning_rate": 0.001}},
  "loss": "categorical_crossentropy"
}
```

**Response:**
```json
{
  "dsl": "network MyModel { ... }",
  "tensorflow": "import tensorflow as tf ...",
  "pytorch": "import torch ..."
}
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Delete** | Remove selected layer |
| **Ctrl+C** | Copy selected layer |
| **Ctrl+V** | Paste layer |
| **Ctrl+Z** | Undo (coming soon) |
| **Ctrl+Y** | Redo (coming soon) |
| **Ctrl+S** | Save model (coming soon) |

## Implementation Details

### React Flow Interface
- **Frontend**: React 18 with React Flow 11
- **Backend**: Flask with CORS support
- **Validation**: Real-time shape propagation engine
- **Code Gen**: Neural DSL parser and multi-backend generators
- **Styling**: Custom CSS with dark theme

### Classic Dash Interface
- **Framework**: Plotly Dash with Bootstrap
- **Visualization**: Plotly charts and graphs
- **Components**: Dash Bootstrap Components
- **Integration**: Direct Neural DSL integration

## Customization

### Adding Custom Layers

Edit `neural/no_code/app.py`:

```python
LAYER_CATEGORIES = {
    "Custom": [
        {
            "name": "MyCustomLayer",
            "params": {"param1": 10, "param2": "value"}
        }
    ]
}
```

### Adding Templates

```python
MODEL_TEMPLATES = {
    "my_template": {
        "name": "My Template",
        "description": "Description here",
        "input_shape": [None, 224, 224, 3],
        "layers": [...]
    }
}
```

### Styling

Edit `neural/no_code/static/styles.css` to customize colors, fonts, and layout.

## Troubleshooting

### React Flow UI Not Loading
- Verify Flask is running: `ps aux | grep python`
- Check port 8051 is available: `netstat -an | grep 8051`
- Open browser console (F12) for JavaScript errors
- Ensure CDN resources are accessible (check internet connection)

### Validation Errors
- Ensure model has at least one layer
- Verify input shape format: `[None, height, width, channels]`
- Check layer parameters match expected types
- Review validation panel for specific errors

### Templates Not Loading
- Check `/api/templates` returns data: `curl http://localhost:8051/api/templates`
- Verify template definitions in `app.py`
- Check server logs for Python errors

### Code Generation Fails
- Ensure all layers have valid parameters
- Verify model passes validation
- Check that backend generators are available
- Review error messages in export modal

## Development

### Modify React Components

1. Edit `neural/no_code/static/app.jsx`
2. Make changes to components
3. Refresh browser (no rebuild needed)

### Modify Backend API

1. Edit `neural/no_code/app.py`
2. Add/modify Flask routes
3. Restart server

### Add New Endpoints

```python
@app.route('/api/my-endpoint', methods=['POST'])
def my_endpoint():
    data = request.json
    # Process data
    return jsonify(result)
```

## Future Enhancements

- [ ] Undo/Redo functionality
- [ ] Keyboard shortcuts for all operations
- [ ] Model versioning and history
- [ ] Collaborative editing (multi-user)
- [ ] Export to ONNX format
- [ ] Cloud deployment integration
- [ ] Real-time training visualization
- [ ] Custom layer builder
- [ ] Hyperparameter tuning UI
- [ ] Model performance profiling
- [ ] Dataset integration
- [ ] Auto-architecture search

## Performance Tips

- **Large Models**: Use MiniMap to navigate complex architectures
- **Many Layers**: Collapse unused categories in palette
- **Slow Validation**: Validation runs on every change; wait for edits to complete
- **Memory**: Clear browser cache if UI becomes sluggish

## Contributing

To contribute new features:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test with both UIs
5. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

## License

MIT License - See [LICENSE.md](../../LICENSE.md)
