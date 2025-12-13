# Neural Visualization Gallery

A comprehensive visualization gallery for Neural DSL models featuring architecture diagrams, shape propagation flowcharts, FLOPs/memory charts, and layer computation timelines.

## Features

### 🏗️ Architecture Visualization
- Beautiful network architecture diagrams
- Layer-by-layer representation with parameters
- Interactive SVG and static PNG/SVG exports
- Graphviz integration for complex networks

### 📊 Shape Propagation
- 3D visualization of tensor shape evolution
- Interactive Plotly charts
- Layer-wise parameter count analysis
- Mermaid flowchart generation

### 💾 FLOPs & Memory Analysis
- Computational complexity breakdown
- Memory usage per layer
- FLOPs distribution pie charts
- Cumulative memory tracking

### ⏱️ Timeline Visualization
- Layer computation timelines
- Gantt-style execution flow
- Computation vs data transfer breakdown
- Performance bottleneck identification

### 📤 Export Options
- **PNG**: High-resolution images (300 DPI)
- **SVG**: Scalable vector graphics
- **HTML**: Interactive web visualizations

## Installation

The visualization gallery is included with Neural DSL. Ensure you have the visualization dependencies:

```bash
pip install -e ".[visualization]"
```

Or install full dependencies:

```bash
pip install -e ".[full]"
```

## Usage

### Command Line Interface

#### Generate All Visualizations

```bash
python -m neural.visualization.gallery_cli generate model.neural
```

Options:
- `--input-shape, -i`: Input shape (e.g., "None,28,28,1")
- `--output-dir, -o`: Output directory (default: visualizations)
- `--format, -f`: Export format (png, svg, html)

Example:
```bash
python -m neural.visualization.gallery_cli generate examples/mnist.neural \
    --input-shape "None,28,28,1" \
    --output-dir output \
    --format html
```

#### Start Web Server

```bash
python -m neural.visualization.gallery_cli serve model.neural
```

Options:
- `--port, -p`: Server port (default: 8052)
- `--host, -h`: Server host (default: 0.0.0.0)
- `--debug`: Enable debug mode

Example:
```bash
python -m neural.visualization.gallery_cli serve examples/mnist.neural --port 8052
```

Then open http://localhost:8052 in your browser.

#### Export Gallery Metadata

```bash
python -m neural.visualization.gallery_cli export-json model.neural
```

#### View Model Information

```bash
python -m neural.visualization.gallery_cli info model.neural
```

### Python API

```python
from neural.visualization.aquarium_integration import AquariumVisualizationManager

# Create manager
manager = AquariumVisualizationManager()

# Load model
with open('model.neural', 'r') as f:
    dsl_code = f.read()
manager.load_model_from_dsl(dsl_code)

# Create gallery
gallery = manager.create_gallery(input_shape=(None, 28, 28, 1))

# Get all visualizations
visualizations = gallery.get_all_visualizations()

# Export specific visualization
manager.export_visualization('architecture', format='png', output_path='arch.png')

# Export all visualizations
paths = manager.export_all_visualizations(format='html', output_dir='output')

# Get gallery metadata
metadata = gallery.get_gallery_metadata()
print(f"Total Parameters: {metadata['total_parameters']:,}")
print(f"Total FLOPs: {metadata['total_flops']:,}")
```

### Web Components

```python
from neural.visualization.aquarium_web_components import AquariumWebComponentRenderer
from neural.visualization.aquarium_integration import AquariumVisualizationManager

manager = AquariumVisualizationManager()
manager.load_model_from_dsl(dsl_code)
gallery = manager.create_gallery()

renderer = AquariumWebComponentRenderer(manager)

# Render gallery view
gallery_html = renderer.render_gallery_view()

# Render specific visualization
viz_html = renderer.render_visualization_detail('architecture')

# Generate thumbnail
thumbnail = renderer.generate_thumbnail('shape_propagation')
```

### Flask Server

```python
from neural.visualization.aquarium_server import start_server

# Start server on default port (8052)
start_server()

# Start on custom port
start_server(port=8080, debug=True)
```

## API Endpoints

When running the server, the following endpoints are available:

### GET /
Gallery homepage with all visualizations

### GET /visualization/<viz_type>
View specific visualization detail page

### POST /api/load-model
Load a Neural DSL model
```json
{
  "dsl_code": "network TestNet { ... }"
}
```

### POST /api/create-gallery
Create visualization gallery
```json
{
  "input_shape": [null, 28, 28, 1]
}
```

### GET /api/visualization/<viz_type>
Get visualization data as JSON

### GET /api/export/<viz_type>/<format>
Export visualization (format: png, svg, html)

### GET /api/export-all/<format>
Export all visualizations

### GET /api/gallery-metadata
Get gallery metadata

### GET /api/gallery-json
Get complete gallery as JSON

### GET /api/visualization/<viz_type>/thumbnail
Get visualization thumbnail

## Visualization Types

### Architecture
- Matplotlib-based diagrams
- Graphviz DOT graphs
- D3.js compatible JSON

### Shape Propagation
- 3D Plotly scatter plots
- Parameter count bar charts
- Mermaid flowcharts

### FLOPs & Memory
- Distribution pie charts
- Layer-wise bar charts
- Cumulative line plots

### Timeline
- Gantt-style timelines
- Stacked bar charts
- Computation/transfer breakdown

## Example

```python
from neural.visualization.aquarium_integration import AquariumVisualizationManager

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

manager = AquariumVisualizationManager()
manager.load_model_from_dsl(dsl_code)
gallery = manager.create_gallery()

# Export all as HTML
paths = manager.export_all_visualizations(format='html', output_dir='visualizations')

print("Created visualizations:")
for viz_type, path in paths.items():
    print(f"  {viz_type}: {path}")
```

## Gallery Metadata

The gallery provides comprehensive metadata:

```json
{
  "model_name": "TestNet",
  "total_layers": 5,
  "input_shape": [null, 28, 28, 1],
  "output_shape": 10,
  "visualizations_available": [
    "architecture",
    "shape_propagation",
    "flops_memory",
    "timeline"
  ],
  "total_parameters": 810890,
  "total_flops": 2400000
}
```

## Integration with Aquarium IDE

The visualization gallery is designed to integrate seamlessly with the Aquarium IDE:

1. Load models from the IDE
2. Generate visualizations on-demand
3. Export for presentations
4. Real-time updates as models change

## Dependencies

- **Core**: numpy, matplotlib, plotly, graphviz
- **Web**: flask, flask-cors
- **Optional**: mpld3 (for matplotlib to HTML conversion)

## Performance

- Efficient shape propagation
- Lazy visualization generation
- Cached exports
- Optimized for large models

## Future Enhancements

- [ ] Real-time model editing
- [ ] Comparison views for multiple models
- [ ] Animation export (GIF, MP4)
- [ ] Customizable color schemes
- [ ] Export templates
- [ ] Batch processing
- [ ] Integration with TensorBoard

## Contributing

Contributions are welcome! Areas for improvement:

- Additional visualization types
- Performance optimizations
- UI/UX enhancements
- Export format support
- Documentation

## License

Part of the Neural DSL project. See main LICENSE file.
