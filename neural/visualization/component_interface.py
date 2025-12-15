from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Optional


@dataclass
class VisualizationMetadata:
    id: str
    name: str
    description: str
    type: str
    thumbnail_url: Optional[str] = None


@dataclass
class GalleryMetadata:
    model_name: str
    total_layers: int
    total_parameters: int
    total_flops: int
    input_shape: Optional[tuple] = None
    output_shape: Optional[int] = None
    visualizations: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ComponentInterface:
    def __init__(self, manager):
        self.manager = manager
    
    def get_gallery_info(self) -> Dict[str, Any]:
        if self.manager.current_gallery is None:
            return {
                'status': 'no_gallery',
                'message': 'No gallery has been created yet'
            }
        
        metadata = self.manager.get_gallery_metadata()
        
        return {
            'status': 'ready',
            'metadata': GalleryMetadata(
                model_name=metadata.get('model_name', 'Unknown'),
                total_layers=metadata.get('total_layers', 0),
                total_parameters=metadata.get('total_parameters', 0),
                total_flops=metadata.get('total_flops', 0),
                input_shape=metadata.get('input_shape'),
                output_shape=metadata.get('output_shape'),
                visualizations=metadata.get('visualizations_available', [])
            ).to_dict()
        }
    
    def get_visualization_list(self) -> List[Dict[str, Any]]:
        if self.manager.current_gallery is None:
            return []
        
        visualizations = []
        viz_list = self.manager.get_visualization_list()
        
        for viz in viz_list:
            visualizations.append(
                VisualizationMetadata(
                    id=viz['id'],
                    name=viz['name'],
                    description=viz['description'],
                    type=self._get_viz_type(viz['id']),
                    thumbnail_url=viz.get('thumbnail')
                ).__dict__
            )
        
        return visualizations
    
    def _get_viz_type(self, viz_id: str) -> str:
        type_map = {
            'architecture': 'diagram',
            'shape_propagation': 'chart',
            'flops_memory': 'chart',
            'timeline': 'timeline'
        }
        return type_map.get(viz_id, 'other')
    
    def get_visualization_data(self, viz_id: str) -> Optional[Dict[str, Any]]:
        if self.manager.current_gallery is None:
            return None
        
        return self.manager.current_gallery.get_visualization(viz_id)
    
    def export_viz(self, viz_id: str, format: str = 'png') -> Dict[str, Any]:
        try:
            path = self.manager.export_visualization(viz_id, format)
            return {
                'success': True,
                'path': path,
                'format': format
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_all(self, format: str = 'png', output_dir: Optional[str] = None) -> Dict[str, Any]:
        try:
            paths = self.manager.export_all_visualizations(format, output_dir)
            return {
                'success': True,
                'paths': paths,
                'count': len(paths)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def create_component_interface(dsl_code: Optional[str] = None) -> ComponentInterface:
    from neural.visualization.aquarium_integration import AquariumVisualizationManager
    
    manager = AquariumVisualizationManager()
    
    if dsl_code:
        manager.load_model_from_dsl(dsl_code)
        manager.create_gallery()
    
    return ComponentInterface(manager)


def get_visualization_preview_html(viz_type: str, viz_data: Dict[str, Any]) -> str:
    icons = {
        'architecture': '🏗️',
        'shape_propagation': '📊',
        'flops_memory': '💾',
        'timeline': '⏱️'
    }
    
    titles = {
        'architecture': 'Architecture Diagram',
        'shape_propagation': 'Shape Propagation',
        'flops_memory': 'FLOPs & Memory',
        'timeline': 'Execution Timeline'
    }
    
    icon = icons.get(viz_type, '📈')
    title = titles.get(viz_type, viz_type.replace('_', ' ').title())
    
    html = f"""
    <div class="viz-preview" data-viz-type="{viz_type}">
        <div class="viz-icon">{icon}</div>
        <div class="viz-title">{title}</div>
        <div class="viz-content" id="viz-{viz_type}">
            <!-- Visualization content will be rendered here -->
        </div>
    </div>
    
    <style>
        .viz-preview {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .viz-icon {{
            font-size: 3em;
            text-align: center;
            margin-bottom: 10px;
        }}
        
        .viz-title {{
            font-size: 1.5em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }}
        
        .viz-content {{
            min-height: 300px;
        }}
    </style>
    
    <script>
        // Visualization data
        const vizData_{viz_type.replace('-', '_')} = {json.dumps(viz_data, default=str)};
        
        // Render function would go here
        // This is a placeholder for the actual rendering logic
        console.log('Visualization data loaded for {viz_type}:', vizData_{viz_type.replace('-', '_')});
    </script>
    """
    
    return html


if __name__ == '__main__':
    example_dsl = """
    network TestNet {
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
    
    interface = create_component_interface(example_dsl)
    
    print("Gallery Info:")
    print(json.dumps(interface.get_gallery_info(), indent=2))
    
    print("\n\nVisualization List:")
    print(json.dumps(interface.get_visualization_list(), indent=2))
    
    print("\n\nExporting architecture visualization...")
    result = interface.export_viz('architecture', 'html')
    print(json.dumps(result, indent=2))
