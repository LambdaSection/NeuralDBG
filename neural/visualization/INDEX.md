# Neural Visualization Gallery - Index

Welcome to the Neural Visualization Gallery! This index will help you find what you need quickly.

## 📚 Documentation

| Document | Description | When to Use |
|----------|-------------|-------------|
| [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md) | 5-minute getting started guide | First time users |
| [GALLERY_README.md](GALLERY_README.md) | Complete documentation | Reference and deep dive |
| [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md) | Aquarium IDE integration | Integrating with Aquarium |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Technical overview | Developers and maintainers |
| [INDEX.md](INDEX.md) | This file | Navigation |

## 🚀 Quick Start

### I want to...

#### Generate visualizations from command line
```bash
python -m neural.visualization.gallery_cli generate model.neural
```
→ See [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md#2-create-your-first-gallery)

#### Start the web interface
```bash
python -m neural.visualization.gallery_cli server
```
→ See [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md#3-start-the-web-interface)

#### Use in Python code
```python
from neural.visualization.aquarium_integration import AquariumVisualizationManager
manager = AquariumVisualizationManager()
```
→ See [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md#6-api-usage)

#### Integrate with Aquarium IDE
→ See [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md)

#### Run examples
```bash
python neural/visualization/gallery_example.py
```
→ See [gallery_example.py](gallery_example.py)

## 📁 File Guide

### Core Files

| File | Purpose | Import/Run |
|------|---------|------------|
| `gallery.py` | Main gallery implementation | `from neural.visualization.gallery import VisualizationGallery` |
| `aquarium_integration.py` | Python API | `from neural.visualization.aquarium_integration import AquariumVisualizationManager` |
| `aquarium_web_components.py` | Web UI components | `from neural.visualization.aquarium_web_components import AquariumWebComponentRenderer` |
| `aquarium_server.py` | Flask web server | `python -m neural.visualization.aquarium_server` |
| `gallery_cli.py` | Command-line interface | `python -m neural.visualization.gallery_cli` |
| `component_interface.py` | Simplified component API | `from neural.visualization.component_interface import create_component_interface` |

### Examples

| File | Purpose | Run |
|------|---------|-----|
| `gallery_example.py` | Usage examples | `python neural/visualization/gallery_example.py` |

### Documentation

| File | Purpose |
|------|---------|
| `GALLERY_README.md` | Complete documentation |
| `QUICKSTART_GALLERY.md` | Quick start guide |
| `AQUARIUM_INTEGRATION.md` | Integration guide |
| `IMPLEMENTATION_SUMMARY.md` | Technical summary |
| `INDEX.md` | This file |

## 🎨 Visualization Types

### Architecture Diagram
Shows network structure with layers and connections.

**Features:**
- Matplotlib and Graphviz renderings
- Layer parameters display
- Color-coded layer types
- Export: PNG, SVG, HTML

**Get it:**
```python
arch_viz = gallery.get_visualization('architecture')
```

### Shape Propagation
3D visualization of tensor shape evolution.

**Features:**
- Interactive 3D Plotly charts
- Parameter count analysis
- Shape history tracking
- Export: PNG, SVG, HTML

**Get it:**
```python
shape_viz = gallery.get_visualization('shape_propagation')
```

### FLOPs & Memory
Computational complexity analysis.

**Features:**
- FLOPs distribution
- Memory usage charts
- Cumulative tracking
- Export: PNG, SVG, HTML

**Get it:**
```python
flops_viz = gallery.get_visualization('flops_memory')
```

### Timeline
Execution timeline with timing breakdown.

**Features:**
- Gantt-style timeline
- Computation vs transfer
- Performance analysis
- Export: PNG, SVG, HTML

**Get it:**
```python
timeline_viz = gallery.get_visualization('timeline')
```

## 🔧 API Quick Reference

### AquariumVisualizationManager

```python
manager = AquariumVisualizationManager()
manager.load_model_from_dsl(dsl_code)
gallery = manager.create_gallery()
paths = manager.export_all_visualizations(format='html')
```

### ComponentInterface

```python
interface = create_component_interface(dsl_code)
gallery_info = interface.get_gallery_info()
viz_list = interface.get_visualization_list()
```

### CLI Commands

```bash
# Generate
python -m neural.visualization.gallery_cli generate model.neural

# Serve
python -m neural.visualization.gallery_cli serve model.neural

# Server
python -m neural.visualization.gallery_cli server

# Export JSON
python -m neural.visualization.gallery_cli export-json model.neural

# Info
python -m neural.visualization.gallery_cli info model.neural
```

### REST API

```bash
# Load model
POST /api/load-model
Body: {"dsl_code": "network TestNet { ... }"}

# Create gallery
POST /api/create-gallery
Body: {"input_shape": [null, 28, 28, 1]}

# Get visualization
GET /api/visualization/architecture

# Export
GET /api/export/architecture/png

# Gallery metadata
GET /api/gallery-metadata
```

## 🔍 Common Tasks

### Task: Generate all visualizations
```bash
python -m neural.visualization.gallery_cli generate model.neural \
    --format html \
    --output-dir visualizations
```

### Task: Export specific visualization
```python
manager.export_visualization('architecture', format='png', output_path='arch.png')
```

### Task: Get model statistics
```python
metadata = gallery.get_gallery_metadata()
print(f"Parameters: {metadata['total_parameters']:,}")
print(f"FLOPs: {metadata['total_flops']:,}")
```

### Task: Start web server on custom port
```bash
python -m neural.visualization.gallery_cli server --port 8080
```

### Task: Export gallery as JSON
```python
gallery_json = gallery.to_json()
with open('gallery.json', 'w') as f:
    f.write(gallery_json)
```

## 🐛 Troubleshooting

### Issue: Import errors
**Solution:** Install dependencies
```bash
pip install -e ".[visualization]"
```

### Issue: Graphviz not found
**Solution:** Install system package
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
```

### Issue: Port already in use
**Solution:** Use different port
```bash
python -m neural.visualization.gallery_cli server --port 8053
```

### Issue: Export fails
**Solution:** Check output directory exists and is writable

### Issue: No visualizations shown
**Solution:** Ensure gallery is created first
```python
manager.create_gallery()
```

## 📖 Learning Path

### Beginner
1. Read [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md)
2. Run [gallery_example.py](gallery_example.py)
3. Try CLI commands
4. Explore web interface

### Intermediate
1. Use Python API in your code
2. Customize export options
3. Integrate with your workflow
4. Read [GALLERY_README.md](GALLERY_README.md)

### Advanced
1. Integrate with Aquarium IDE
2. Customize visualizations
3. Build on top of gallery API
4. Read [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md)
5. Study [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 🆘 Getting Help

1. **Check documentation:**
   - [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md) for quick help
   - [GALLERY_README.md](GALLERY_README.md) for detailed info
   - [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md) for integration

2. **Run examples:**
   ```bash
   python neural/visualization/gallery_example.py
   ```

3. **Check implementation:**
   - See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
   - Look at source code in [gallery.py](gallery.py)

4. **GitHub Issues:**
   - https://github.com/Lemniscate-world/Neural/issues

## 🎯 Use Cases

### Use Case 1: Model Documentation
Generate visualizations for documentation:
```bash
python -m neural.visualization.gallery_cli generate model.neural \
    --format png --output-dir docs/images
```

### Use Case 2: Model Analysis
Analyze model complexity:
```python
metadata = gallery.get_gallery_metadata()
flops_viz = gallery.get_visualization('flops_memory')
```

### Use Case 3: Presentation
Export interactive HTML for presentations:
```bash
python -m neural.visualization.gallery_cli generate model.neural \
    --format html --output-dir presentation
```

### Use Case 4: Development
Run web server during development:
```bash
python -m neural.visualization.gallery_cli serve model.neural --debug
```

### Use Case 5: Integration
Integrate with Aquarium IDE:
See [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md)

## 🔗 Links

- **Neural DSL:** https://github.com/Lemniscate-world/Neural
- **Issues:** https://github.com/Lemniscate-world/Neural/issues
- **Documentation:** https://github.com/Lemniscate-world/Neural/tree/main/docs

## 📝 Summary

The Neural Visualization Gallery provides:
- ✅ 4 visualization types
- ✅ 3 export formats (PNG, SVG, HTML)
- ✅ Web interface with REST API
- ✅ Command-line interface
- ✅ Python API
- ✅ Component interface
- ✅ Comprehensive documentation
- ✅ Integration guides

Choose your starting point above and dive in! 🚀
