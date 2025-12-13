from __future__ import annotations

from typing import Dict, Any, Optional
import json
import base64
from io import BytesIO

from neural.visualization.aquarium_integration import AquariumVisualizationManager


class AquariumWebComponentRenderer:
    def __init__(self, manager: AquariumVisualizationManager):
        self.manager = manager
    
    def render_gallery_view(self) -> str:
        metadata = self.manager.get_gallery_metadata()
        visualizations = self.manager.get_visualization_list()
        
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Visualization Gallery</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .header h1 {
            color: #1e3c72;
            font-size: 2.5em;
            margin-bottom: 15px;
        }
        
        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .metadata-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #1e3c72;
        }
        
        .metadata-item .label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        .metadata-item .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #1e3c72;
        }
        
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }
        
        .viz-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }
        
        .viz-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        
        .viz-card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: white;
        }
        
        .viz-card-header h3 {
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        
        .viz-card-header p {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .viz-card-body {
            padding: 20px;
        }
        
        .viz-preview {
            background: #f8f9fa;
            height: 200px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 15px;
            color: #999;
            font-size: 3em;
        }
        
        .viz-actions {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5568d3;
        }
        
        .btn-secondary {
            background: #e9ecef;
            color: #495057;
        }
        
        .btn-secondary:hover {
            background: #dee2e6;
        }
        
        .export-menu {
            position: relative;
        }
        
        .export-dropdown {
            display: none;
            position: absolute;
            bottom: 100%;
            right: 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            margin-bottom: 5px;
            min-width: 150px;
            z-index: 10;
        }
        
        .export-dropdown.active {
            display: block;
        }
        
        .export-option {
            padding: 12px 20px;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        
        .export-option:hover {
            background: #f8f9fa;
        }
        
        .export-option:first-child {
            border-radius: 8px 8px 0 0;
        }
        
        .export-option:last-child {
            border-radius: 0 0 8px 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Neural Visualization Gallery</h1>
            <p style="color: #666; font-size: 1.1em; margin-top: 10px;">
                Interactive visualizations for """ + str(metadata.get('model_name', 'your model')) + """
            </p>
            
            <div class="metadata">
                <div class="metadata-item">
                    <div class="label">Total Layers</div>
                    <div class="value">""" + str(metadata.get('total_layers', 0)) + """</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Total Parameters</div>
                    <div class="value">""" + f"{metadata.get('total_parameters', 0):,}" + """</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Total FLOPs</div>
                    <div class="value">""" + f"{metadata.get('total_flops', 0):,}" + """</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Visualizations</div>
                    <div class="value">""" + str(len(visualizations)) + """</div>
                </div>
            </div>
        </div>
        
        <div class="gallery-grid">
"""
        
        icons = {
            'architecture': '🏗️',
            'shape_propagation': '📊',
            'flops_memory': '💾',
            'timeline': '⏱️'
        }
        
        for viz in visualizations:
            icon = icons.get(viz['id'], '📈')
            html += f"""
            <div class="viz-card" data-viz-id="{viz['id']}">
                <div class="viz-card-header">
                    <h3>{icon} {viz['name']}</h3>
                    <p>{viz['description']}</p>
                </div>
                <div class="viz-card-body">
                    <div class="viz-preview">{icon}</div>
                    <div class="viz-actions">
                        <button class="btn btn-primary" onclick="viewVisualization('{viz['id']}')">
                            View
                        </button>
                        <div class="export-menu">
                            <button class="btn btn-secondary" onclick="toggleExportMenu('{viz['id']}')">
                                Export ▼
                            </button>
                            <div class="export-dropdown" id="export-{viz['id']}">
                                <div class="export-option" onclick="exportVisualization('{viz['id']}', 'png')">
                                    PNG Image
                                </div>
                                <div class="export-option" onclick="exportVisualization('{viz['id']}', 'svg')">
                                    SVG Vector
                                </div>
                                <div class="export-option" onclick="exportVisualization('{viz['id']}', 'html')">
                                    Interactive HTML
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
"""
        
        html += """
        </div>
    </div>
    
    <script>
        function viewVisualization(vizId) {
            window.location.href = `/visualization/${vizId}`;
        }
        
        function toggleExportMenu(vizId) {
            const dropdown = document.getElementById(`export-${vizId}`);
            dropdown.classList.toggle('active');
            
            document.addEventListener('click', function closeMenu(e) {
                if (!e.target.closest('.export-menu')) {
                    dropdown.classList.remove('active');
                    document.removeEventListener('click', closeMenu);
                }
            });
        }
        
        function exportVisualization(vizId, format) {
            fetch(`/api/export/${vizId}/${format}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Exported ${vizId} to ${data.path}`);
                        window.location.href = `/download/${data.path}`;
                    } else {
                        alert(`Export failed: ${data.error}`);
                    }
                })
                .catch(error => {
                    alert(`Export error: ${error}`);
                });
        }
    </script>
</body>
</html>
"""
        return html
    
    def render_visualization_detail(self, viz_type: str) -> str:
        if self.manager.current_gallery is None:
            return self._render_error("No gallery available")
        
        viz_data = self.manager.current_gallery.get_visualization(viz_type)
        if viz_data is None:
            return self._render_error(f"Visualization '{viz_type}' not found")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{viz_type.replace('_', ' ').title()} - Neural Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        
        .header .breadcrumb {{
            margin-top: 10px;
            opacity: 0.9;
        }}
        
        .header .breadcrumb a {{
            color: white;
            text-decoration: none;
        }}
        
        .header .breadcrumb a:hover {{
            text-decoration: underline;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px;
        }}
        
        .viz-container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        #visualization {{
            min-height: 600px;
        }}
        
        .actions {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }}
        
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #5568d3;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{viz_type.replace('_', ' ').title()}</h1>
        <div class="breadcrumb">
            <a href="/">← Back to Gallery</a>
        </div>
    </div>
    
    <div class="container">
        <div class="viz-container">
            <div id="visualization"></div>
            <div class="actions">
                <button class="btn btn-primary" onclick="exportVisualization('png')">
                    Export as PNG
                </button>
                <button class="btn btn-primary" onclick="exportVisualization('svg')">
                    Export as SVG
                </button>
                <button class="btn btn-primary" onclick="exportVisualization('html')">
                    Export as HTML
                </button>
            </div>
        </div>
    </div>
    
    <script>
        const vizData = {json.dumps(viz_data, default=str)};
        
        if (vizData.plotly_figure) {{
            Plotly.newPlot('visualization', vizData.plotly_figure.data, vizData.plotly_figure.layout);
        }} else if (vizData.timeline_figure) {{
            Plotly.newPlot('visualization', vizData.timeline_figure.data, vizData.timeline_figure.layout);
        }} else {{
            document.getElementById('visualization').innerHTML = '<p>Visualization data loaded. Use export buttons to save.</p>';
        }}
        
        function exportVisualization(format) {{
            fetch(`/api/export/{viz_type}/${{format}}`)
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        alert(`Exported to ${{data.path}}`);
                        window.location.href = `/download/${{data.path}}`;
                    }} else {{
                        alert(`Export failed: ${{data.error}}`);
                    }}
                }})
                .catch(error => alert(`Error: ${{error}}`));
        }}
    </script>
</body>
</html>
"""
        return html
    
    def _render_error(self, message: str) -> str:
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Neural Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        
        .error-container {{
            background: white;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 500px;
        }}
        
        .error-icon {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
        
        h1 {{
            color: #e74c3c;
            margin-bottom: 15px;
        }}
        
        p {{
            color: #666;
            margin-bottom: 30px;
        }}
        
        a {{
            display: inline-block;
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.3s ease;
        }}
        
        a:hover {{
            background: #5568d3;
        }}
    </style>
</head>
<body>
    <div class="error-container">
        <div class="error-icon">⚠️</div>
        <h1>Error</h1>
        <p>{message}</p>
        <a href="/">← Back to Gallery</a>
    </div>
</body>
</html>
"""
    
    def generate_thumbnail(self, viz_type: str, size: tuple = (300, 200)) -> Optional[str]:
        if self.manager.current_gallery is None:
            return None
        
        viz_data = self.manager.current_gallery.get_visualization(viz_type)
        if viz_data is None:
            return None
        
        try:
            if 'matplotlib_figure' in viz_data:
                fig = viz_data['matplotlib_figure']
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=50, bbox_inches='tight')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/png;base64,{image_base64}"
            elif 'plotly_figure' in viz_data:
                import plotly.io as pio
                img_bytes = pio.to_image(viz_data['plotly_figure'], format='png', width=size[0], height=size[1])
                image_base64 = base64.b64encode(img_bytes).decode()
                return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            return None
