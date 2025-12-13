# Aquarium Directory Structure

This document describes the intended directory structure for integrating the visualization gallery with the Aquarium IDE.

## Overview

The visualization gallery can be integrated into Aquarium in two ways:
1. **External Integration**: Run as a separate service (current implementation)
2. **Embedded Integration**: Embed components directly in Aquarium (future)

## Current Implementation (External)

The current implementation is located in:
```
neural/visualization/
├── gallery.py
├── aquarium_integration.py
├── aquarium_web_components.py
├── aquarium_server.py
└── ... (other files)
```

This can be used by Aquarium via:
- REST API calls to the server
- Python API integration
- CLI commands

## Proposed Aquarium Structure

When embedding components directly in Aquarium:

```
Aquarium/                                    # Aquarium IDE root
├── src/                                     # Frontend source
│   ├── components/
│   │   ├── visualizations/                  # Visualization components
│   │   │   ├── index.js                    # Main export
│   │   │   ├── GalleryView.jsx             # Gallery grid view
│   │   │   ├── VisualizationCard.jsx       # Individual viz card
│   │   │   ├── ArchitectureView.jsx        # Architecture viewer
│   │   │   ├── ShapeView.jsx               # Shape propagation viewer
│   │   │   ├── FlopsView.jsx               # FLOPs/memory viewer
│   │   │   ├── TimelineView.jsx            # Timeline viewer
│   │   │   ├── ExportButton.jsx            # Export controls
│   │   │   ├── gallery_bridge.ts           # Bridge to Python API
│   │   │   └── styles.css                  # Component styles
│   │   └── ...
│   └── ...
├── src-tauri/                               # Tauri backend (Rust)
│   ├── src/
│   │   ├── main.rs
│   │   ├── visualization_commands.rs        # Tauri commands for viz
│   │   └── ...
│   └── ...
└── ...
```

## Component Files

### GalleryView.jsx
Main gallery component displaying all visualizations in a grid.

**Features:**
- Grid layout
- Metadata display
- Quick actions
- Responsive design

**Usage:**
```jsx
import { GalleryView } from './components/visualizations';
<GalleryView modelData={modelData} />
```

### VisualizationCard.jsx
Individual visualization card component.

**Props:**
- `vizType`: Type of visualization
- `vizData`: Visualization data
- `onView`: View callback
- `onExport`: Export callback

**Usage:**
```jsx
<VisualizationCard 
    vizType="architecture"
    vizData={data}
    onView={handleView}
    onExport={handleExport}
/>
```

### gallery_bridge.ts
TypeScript bridge to Python API.

**Functions:**
```typescript
loadModel(dslCode: string): Promise<GalleryData>
getVisualization(vizType: string): Promise<VizData>
exportVisualization(vizType: string, format: string): Promise<string>
```

### visualization_commands.rs
Rust commands for Tauri.

**Commands:**
```rust
#[tauri::command]
fn load_visualization_model(dsl_code: String) -> Result<String, String>

#[tauri::command]
fn export_visualization(viz_type: String, format: String) -> Result<String, String>
```

## Alternative Structure (Monorepo)

If integrating directly into the Neural repository:

```
neural/
├── aquarium/                                # Aquarium integration
│   ├── src/
│   │   ├── components/
│   │   │   └── visualizations/
│   │   │       ├── __init__.py             # Python exports
│   │   │       ├── gallery_wrapper.py      # Python wrapper
│   │   │       └── ...
│   │   └── ...
│   └── ...
├── visualization/                           # Current location
│   ├── gallery.py
│   ├── aquarium_integration.py
│   └── ...
└── ...
```

## Integration Approaches

### Approach 1: HTTP Bridge (Recommended)

**Pros:**
- Clean separation
- Language agnostic
- Easy debugging
- No subprocess management

**Cons:**
- Requires server running
- Network overhead

**Setup:**
```bash
# Terminal 1: Start server
python -m neural.visualization.aquarium_server

# Terminal 2: Start Aquarium
cd Aquarium && npm run tauri dev
```

**Code:**
```typescript
const response = await fetch('http://localhost:8052/api/visualization/architecture');
const data = await response.json();
```

### Approach 2: Python Subprocess (Alternative)

**Pros:**
- No server needed
- Direct integration
- Self-contained

**Cons:**
- Process management complexity
- Slower than HTTP
- Harder debugging

**Setup:**
Embed Python calls in Rust commands.

**Code:**
```rust
let output = Command::new("python")
    .arg("-c")
    .arg("from neural.visualization.aquarium_integration import ...; ...")
    .output()?;
```

### Approach 3: Embedded Python (Advanced)

**Pros:**
- Direct Python integration
- No separate process
- Fast

**Cons:**
- Complex setup
- Platform-specific
- Requires PyO3

**Setup:**
Use PyO3 to embed Python in Rust.

**Code:**
```rust
use pyo3::prelude::*;

fn load_model(dsl_code: &str) -> PyResult<String> {
    Python::with_gil(|py| {
        let module = py.import("neural.visualization.aquarium_integration")?;
        let manager = module.getattr("AquariumVisualizationManager")?.call0()?;
        // ...
    })
}
```

## File Mapping

### Python → JavaScript

| Python Component | JavaScript Component | Purpose |
|-----------------|---------------------|---------|
| `VisualizationGallery` | `GalleryView.jsx` | Main gallery |
| `ArchitectureVisualizer` | `ArchitectureView.jsx` | Architecture display |
| `ShapePropagationVisualizer` | `ShapeView.jsx` | Shape visualization |
| `FlopsMemoryVisualizer` | `FlopsView.jsx` | FLOPs/memory charts |
| `TimelineVisualizer` | `TimelineView.jsx` | Timeline display |
| `ExportHandler` | `ExportButton.jsx` | Export controls |

### Python → Rust

| Python Function | Rust Command | Purpose |
|----------------|--------------|---------|
| `load_model_from_dsl()` | `load_visualization_model` | Load model |
| `create_gallery()` | `create_gallery` | Create gallery |
| `export_visualization()` | `export_visualization` | Export viz |
| `export_all_visualizations()` | `export_all` | Export all |

## Data Flow

```
User Action (Frontend)
    ↓
JavaScript/TypeScript Bridge
    ↓
Tauri Command (Rust)
    ↓
Python API Call
    ↓
Neural Visualization Gallery
    ↓
Return Data
    ↓
Display in UI
```

## Setup Instructions

### External Integration (Current)

1. Install Neural DSL with visualization support:
```bash
pip install -e ".[visualization]"
```

2. Start the visualization server:
```bash
python -m neural.visualization.aquarium_server
```

3. Configure Aquarium to connect to `http://localhost:8052`

4. Use API endpoints from Aquarium frontend

### Embedded Integration (Future)

1. Copy visualization components to Aquarium:
```bash
# Create directory structure
mkdir -p Aquarium/src/components/visualizations

# Note: Implementation will be added in future
```

2. Implement Tauri commands in Rust

3. Create JavaScript/TypeScript bridges

4. Import and use components in Aquarium

## Environment Variables

```bash
# Visualization server configuration
NEURAL_VIZ_HOST=127.0.0.1
NEURAL_VIZ_PORT=8052
NEURAL_VIZ_DEBUG=false

# Aquarium configuration
AQUARIUM_VIZ_URL=http://localhost:8052
AQUARIUM_VIZ_TIMEOUT=30000  # ms
```

## Testing Integration

### Test HTTP Bridge
```bash
# Start server
python -m neural.visualization.aquarium_server &

# Test connection
curl http://localhost:8052/api/gallery-metadata

# Stop server
kill %1
```

### Test Python Integration
```python
from neural.visualization.aquarium_integration import AquariumVisualizationManager

manager = AquariumVisualizationManager()
manager.load_model_from_dsl(dsl_code)
gallery = manager.create_gallery()
print(gallery.to_json())
```

## Documentation

- Full documentation: [GALLERY_README.md](GALLERY_README.md)
- Integration guide: [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md)
- Quick start: [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md)

## Future Work

- [ ] Create Aquarium component library
- [ ] Implement Tauri commands
- [ ] Add WebSocket support for real-time updates
- [ ] Create React/Vue/Svelte component wrappers
- [ ] Add TypeScript type definitions
- [ ] Implement caching layer
- [ ] Add offline support

## Notes

1. The current implementation is fully functional and can be integrated immediately via HTTP
2. Embedded integration requires additional work but provides better performance
3. Choose integration approach based on your needs and constraints
4. HTTP bridge is recommended for initial integration

## Contact

For questions or help with integration:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- See documentation: [AQUARIUM_INTEGRATION.md](AQUARIUM_INTEGRATION.md)
