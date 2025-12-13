# Aquarium Integration Guide

This document explains how to integrate the Neural Visualization Gallery with the Aquarium IDE.

## Overview

The visualization gallery is implemented in `neural/visualization/` and can be integrated into the Aquarium IDE located in the `Aquarium/` submodule (or `neural/aquarium/` if integrated directly).

## Architecture

```
neural/visualization/
├── gallery.py                      # Main gallery implementation
├── aquarium_integration.py         # Python API for Aquarium
├── aquarium_web_components.py      # Web UI components
├── aquarium_server.py              # Flask server
├── gallery_cli.py                  # CLI commands
├── gallery_example.py              # Usage examples
├── GALLERY_README.md               # Full documentation
├── QUICKSTART_GALLERY.md           # Quick start guide
└── AQUARIUM_INTEGRATION.md         # This file
```

## Integration Points

### 1. Python API (Recommended)

For Rust/Tauri integration in Aquarium:

```python
# In your Aquarium backend Python service
from neural.visualization.aquarium_integration import AquariumVisualizationManager

manager = AquariumVisualizationManager()

# Load model from Aquarium
manager.load_model_from_dict(aquarium_model_data)

# Create gallery
gallery = manager.create_gallery()

# Return data to Tauri frontend
gallery_json = gallery.to_json()
```

### 2. REST API

Start the visualization server:

```bash
python -m neural.visualization.aquarium_server
```

Access from Aquarium frontend:

```javascript
// In Aquarium frontend (JavaScript/TypeScript)
const response = await fetch('http://localhost:8052/api/load-model', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({dsl_code: neuralDslCode})
});

const result = await response.json();

// Create gallery
await fetch('http://localhost:8052/api/create-gallery', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({input_shape: [null, 28, 28, 1]})
});

// Get visualizations
const vizResponse = await fetch('http://localhost:8052/api/visualization/architecture');
const vizData = await vizResponse.json();
```

### 3. Embedded Server

Embed the server in Aquarium:

```python
# In Aquarium backend
from neural.visualization.aquarium_server import AquariumVisualizationServer

server = AquariumVisualizationServer(host='127.0.0.1', port=8052)
# Run in background thread
import threading
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
```

## Directory Structure for Aquarium

If integrating directly into the Aquarium submodule:

```
Aquarium/
├── src/
│   ├── components/
│   │   ├── visualizations/
│   │   │   ├── gallery_bridge.js      # Bridge to Python API
│   │   │   ├── architecture_view.js   # Architecture visualization
│   │   │   ├── shape_view.js          # Shape propagation view
│   │   │   ├── flops_view.js          # FLOPs/memory view
│   │   │   └── timeline_view.js       # Timeline view
│   │   └── ...
│   └── ...
└── src-tauri/
    ├── src/
    │   ├── visualization_commands.rs   # Tauri commands
    │   └── ...
    └── ...
```

## Tauri Commands

Example Rust commands for Aquarium:

```rust
// In src-tauri/src/visualization_commands.rs
use std::process::Command;

#[tauri::command]
pub fn load_visualization_model(dsl_code: String) -> Result<String, String> {
    // Call Python API
    let output = Command::new("python")
        .arg("-c")
        .arg(format!(
            "from neural.visualization.aquarium_integration import AquariumVisualizationManager; \
             manager = AquariumVisualizationManager(); \
             manager.load_model_from_dsl('{}'); \
             gallery = manager.create_gallery(); \
             print(gallery.to_json())",
            dsl_code.replace("'", "\\'")
        ))
        .output()
        .map_err(|e| e.to_string())?;
    
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

#[tauri::command]
pub fn export_visualization(viz_type: String, format: String) -> Result<String, String> {
    let output = Command::new("python")
        .arg("-m")
        .arg("neural.visualization.gallery_cli")
        .arg("export")
        .arg(&viz_type)
        .arg("--format")
        .arg(&format)
        .output()
        .map_err(|e| e.to_string())?;
    
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}
```

Register commands in `main.rs`:

```rust
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            load_visualization_model,
            export_visualization
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

## JavaScript/TypeScript Integration

In Aquarium frontend:

```typescript
// gallery_bridge.ts
import { invoke } from '@tauri-apps/api/tauri';

export interface GalleryMetadata {
    model_name: string;
    total_layers: number;
    total_parameters: number;
    total_flops: number;
    visualizations_available: string[];
}

export interface VisualizationGallery {
    metadata: GalleryMetadata;
    visualizations: {
        architecture: any;
        shape_propagation: any;
        flops_memory: any;
        timeline: any;
    };
}

export async function loadModel(dslCode: string): Promise<VisualizationGallery> {
    const result = await invoke<string>('load_visualization_model', { dslCode });
    return JSON.parse(result);
}

export async function exportVisualization(
    vizType: string,
    format: 'png' | 'svg' | 'html'
): Promise<string> {
    return await invoke('export_visualization', { vizType, format });
}

// Use in React/Svelte/Vue component
export function useVisualizationGallery() {
    const [gallery, setGallery] = useState<VisualizationGallery | null>(null);
    const [loading, setLoading] = useState(false);
    
    const load = async (dslCode: string) => {
        setLoading(true);
        try {
            const result = await loadModel(dslCode);
            setGallery(result);
        } finally {
            setLoading(false);
        }
    };
    
    return { gallery, loading, load };
}
```

## React Component Example

```tsx
// VisualizationGallery.tsx
import React from 'react';
import { useVisualizationGallery } from './gallery_bridge';

export function VisualizationGallery({ dslCode }: { dslCode: string }) {
    const { gallery, loading, load } = useVisualizationGallery();
    
    React.useEffect(() => {
        if (dslCode) {
            load(dslCode);
        }
    }, [dslCode]);
    
    if (loading) return <div>Loading visualizations...</div>;
    if (!gallery) return <div>No gallery loaded</div>;
    
    return (
        <div className="visualization-gallery">
            <div className="metadata">
                <h2>{gallery.metadata.model_name}</h2>
                <div className="stats">
                    <div>Layers: {gallery.metadata.total_layers}</div>
                    <div>Parameters: {gallery.metadata.total_parameters.toLocaleString()}</div>
                    <div>FLOPs: {gallery.metadata.total_flops.toLocaleString()}</div>
                </div>
            </div>
            
            <div className="visualizations-grid">
                {gallery.metadata.visualizations_available.map(vizType => (
                    <VisualizationCard 
                        key={vizType}
                        type={vizType}
                        data={gallery.visualizations[vizType]}
                    />
                ))}
            </div>
        </div>
    );
}
```

## Alternative: HTTP Bridge

If Python-Tauri integration is complex, use HTTP:

```typescript
// http_bridge.ts
const API_URL = 'http://localhost:8052';

export async function loadModel(dslCode: string) {
    await fetch(`${API_URL}/api/load-model`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dsl_code: dslCode})
    });
    
    const response = await fetch(`${API_URL}/api/create-gallery`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    });
    
    return await response.json();
}

export async function getVisualization(vizType: string) {
    const response = await fetch(`${API_URL}/api/visualization/${vizType}`);
    return await response.json();
}

export async function exportVisualization(vizType: string, format: string) {
    const response = await fetch(`${API_URL}/api/export/${vizType}/${format}`);
    return await response.json();
}
```

## CSS Styling

```css
/* styles.css for Aquarium integration */
.visualization-gallery {
    padding: 20px;
    background: #f5f7fa;
    min-height: 100vh;
}

.metadata {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.stats > div {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

.visualizations-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

.viz-card {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.viz-card:hover {
    transform: translateY(-5px);
}
```

## Deployment Considerations

1. **Development**: Run server separately on port 8052
2. **Production**: Bundle server with Aquarium or use embedded mode
3. **Security**: Use localhost only, or add authentication if exposing externally
4. **Performance**: Cache visualizations, generate on-demand

## Testing Integration

```bash
# 1. Start visualization server
python -m neural.visualization.aquarium_server

# 2. Start Aquarium
cd Aquarium
npm run tauri dev

# 3. Test connection
curl http://localhost:8052/api/gallery-metadata
```

## Example: Complete Integration Flow

1. User creates model in Aquarium IDE
2. Aquarium generates Neural DSL code
3. Code sent to visualization server via API
4. Server creates gallery and returns metadata
5. Aquarium displays gallery in sidebar/panel
6. User can export visualizations
7. Visualizations saved to user's project folder

## Documentation

- Full API: See [GALLERY_README.md](GALLERY_README.md)
- Quick Start: See [QUICKSTART_GALLERY.md](QUICKSTART_GALLERY.md)
- Examples: See [gallery_example.py](gallery_example.py)

## Support

For integration help:
- Check examples in `gallery_example.py`
- Run the server standalone to test
- See Flask API documentation in `aquarium_server.py`
- Open issues on GitHub

Happy Integrating! 🚀
