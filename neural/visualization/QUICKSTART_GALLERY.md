# Visualization Gallery - Quick Start Guide

Get up and running with the Neural Visualization Gallery in 5 minutes!

## 1. Installation

```bash
# Install Neural DSL with visualization support
pip install -e ".[visualization]"

# Or install everything
pip install -e ".[full]"
```

## 2. Create Your First Gallery

### Option A: From Python

```python
from neural.visualization.aquarium_integration import AquariumVisualizationManager

# Define your model
dsl_code = """
network MnistCNN {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}
"""

# Create and generate gallery
manager = AquariumVisualizationManager()
manager.load_model_from_dsl(dsl_code)
gallery = manager.create_gallery()

# Export all visualizations
paths = manager.export_all_visualizations(format='html', output_dir='viz_output')

print("Visualizations created:")
for viz_type, path in paths.items():
    print(f"  {viz_type}: {path}")
```

### Option B: From Command Line

```bash
# Save your model to a file
cat > model.neural << 'EOF'
network MnistCNN {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}
EOF

# Generate visualizations
python -m neural.visualization.gallery_cli generate model.neural
```

## 3. Start the Web Interface

```bash
# Start the server
python -m neural.visualization.gallery_cli serve model.neural

# Or just start the server and load models through the UI
python -m neural.visualization.gallery_cli server
```

Open http://localhost:8052 in your browser!

## 4. What You Get

### 🏗️ Architecture Diagram
Beautiful visualization of your network structure with layers and connections.

### 📊 Shape Propagation
3D interactive chart showing how tensor shapes evolve through your network.

### 💾 FLOPs & Memory Analysis
Detailed breakdown of computational complexity and memory usage.

### ⏱️ Timeline View
Execution timeline showing computation and data transfer times.

## 5. Export Formats

```python
# Export as PNG (high resolution)
manager.export_visualization('architecture', format='png')

# Export as SVG (scalable vector)
manager.export_visualization('shape_propagation', format='svg')

# Export as HTML (interactive)
manager.export_visualization('flops_memory', format='html')

# Export all at once
manager.export_all_visualizations(format='html', output_dir='output')
```

## 6. API Usage

```python
from neural.visualization.aquarium_integration import AquariumVisualizationManager

manager = AquariumVisualizationManager()

# Load model
manager.load_model_from_dsl(dsl_code)

# Create gallery with custom input shape
gallery = manager.create_gallery(input_shape=(1, 224, 224, 3))

# Get specific visualization
arch_viz = gallery.get_visualization('architecture')

# Get all visualizations
all_viz = gallery.get_all_visualizations()

# Get metadata
metadata = gallery.get_gallery_metadata()
print(f"Total Parameters: {metadata['total_parameters']:,}")
print(f"Total FLOPs: {metadata['total_flops']:,}")

# Export to JSON
gallery_json = gallery.to_json()
with open('gallery.json', 'w') as f:
    f.write(gallery_json)
```

## 7. Web Server API

When running the server, you can use these endpoints:

```bash
# Load a model
curl -X POST http://localhost:8052/api/load-model \
  -H "Content-Type: application/json" \
  -d '{"dsl_code": "network TestNet { ... }"}'

# Create gallery
curl -X POST http://localhost:8052/api/create-gallery \
  -H "Content-Type: application/json" \
  -d '{"input_shape": [null, 28, 28, 1]}'

# Export visualization
curl http://localhost:8052/api/export/architecture/png
```

## 8. Integration with Existing Code

```python
from neural.parser.parser import create_parser, ModelTransformer
from neural.visualization.gallery import VisualizationGallery

# If you already have parsed model data
parser = create_parser('network')
parsed = parser.parse(dsl_code)
model_data = ModelTransformer().transform(parsed)

# Create gallery directly
gallery = VisualizationGallery(model_data)
gallery.generate_all_visualizations(input_shape=(None, 28, 28, 1))
```

## 9. Examples

Run the included examples:

```bash
python neural/visualization/gallery_example.py
```

This will:
- Generate visualizations for multiple models
- Export in different formats
- Show usage of all gallery features
- Create example outputs in `example_output/`

## 10. Troubleshooting

### Missing Dependencies

```bash
pip install matplotlib plotly graphviz flask flask-cors
```

### Graphviz Not Found

**Ubuntu/Debian:**
```bash
sudo apt-get install graphviz
```

**macOS:**
```bash
brew install graphviz
```

**Windows:**
Download from https://graphviz.org/download/

### Port Already in Use

```bash
python -m neural.visualization.gallery_cli server --port 8053
```

## Next Steps

- Read the full documentation: [GALLERY_README.md](GALLERY_README.md)
- Explore advanced features
- Customize visualizations
- Integrate with your workflow

## Support

- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See GALLERY_README.md
- Examples: neural/visualization/gallery_example.py

Happy Visualizing! 🎨📊
