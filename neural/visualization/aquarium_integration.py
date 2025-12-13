from __future__ import annotations

from typing import Dict, Any, List, Optional
import json

from neural.parser.parser import create_parser, ModelTransformer
from neural.visualization.gallery import VisualizationGallery
from neural.shape_propagation.shape_propagator import ShapePropagator


class AquariumVisualizationManager:
    def __init__(self):
        self.parser = create_parser('network')
        self.transformer = ModelTransformer()
        self.current_gallery: Optional[VisualizationGallery] = None
        self.current_model_data: Optional[Dict[str, Any]] = None
    
    def load_model_from_dsl(self, dsl_code: str) -> Dict[str, Any]:
        parsed = self.parser.parse(dsl_code)
        self.current_model_data = self.transformer.transform(parsed)
        return self.current_model_data
    
    def load_model_from_dict(self, model_data: Dict[str, Any]):
        self.current_model_data = model_data
    
    def create_gallery(self, input_shape: Optional[tuple] = None) -> VisualizationGallery:
        if self.current_model_data is None:
            raise ValueError("No model loaded. Call load_model_from_dsl or load_model_from_dict first.")
        
        if input_shape is None:
            input_shape = self.current_model_data.get('input', {}).get('shape', (None, 28, 28, 1))
            if isinstance(input_shape, list):
                input_shape = tuple(input_shape)
        
        self.current_gallery = VisualizationGallery(self.current_model_data)
        self.current_gallery.generate_all_visualizations(input_shape)
        
        return self.current_gallery
    
    def get_gallery_json(self) -> str:
        if self.current_gallery is None:
            raise ValueError("No gallery created. Call create_gallery first.")
        return self.current_gallery.to_json()
    
    def get_visualization_list(self) -> List[Dict[str, str]]:
        if self.current_gallery is None:
            return []
        
        visualizations = []
        for viz_type in self.current_gallery.visualizations.keys():
            visualizations.append({
                'id': viz_type,
                'name': viz_type.replace('_', ' ').title(),
                'description': self._get_viz_description(viz_type),
                'thumbnail': f'/api/visualization/{viz_type}/thumbnail'
            })
        
        return visualizations
    
    def _get_viz_description(self, viz_type: str) -> str:
        descriptions = {
            'architecture': 'Visual representation of the network layers and connections',
            'shape_propagation': '3D visualization of tensor shape evolution through layers',
            'flops_memory': 'Analysis of computational complexity and memory usage',
            'timeline': 'Layer computation timeline showing execution sequence'
        }
        return descriptions.get(viz_type, '')
    
    def export_visualization(self, viz_type: str, format: str = 'png', output_path: Optional[str] = None) -> str:
        if self.current_gallery is None:
            raise ValueError("No gallery created. Call create_gallery first.")
        
        return self.current_gallery.export_visualization(viz_type, format, output_path)
    
    def export_all_visualizations(self, format: str = 'png', output_dir: Optional[str] = None) -> Dict[str, str]:
        if self.current_gallery is None:
            raise ValueError("No gallery created. Call create_gallery first.")
        
        return self.current_gallery.export_all(format, output_dir)
    
    def get_gallery_metadata(self) -> Dict[str, Any]:
        if self.current_gallery is None:
            return {}
        
        return self.current_gallery.get_gallery_metadata()


def create_aquarium_visualization_api():
    manager = AquariumVisualizationManager()
    
    def handle_load_model(dsl_code: str) -> Dict[str, Any]:
        try:
            model_data = manager.load_model_from_dsl(dsl_code)
            return {
                'success': True,
                'model_data': model_data,
                'message': 'Model loaded successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def handle_create_gallery(input_shape: Optional[tuple] = None) -> Dict[str, Any]:
        try:
            gallery = manager.create_gallery(input_shape)
            return {
                'success': True,
                'metadata': gallery.get_gallery_metadata(),
                'visualizations': manager.get_visualization_list(),
                'message': 'Gallery created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def handle_get_visualization(viz_type: str) -> Dict[str, Any]:
        try:
            if manager.current_gallery is None:
                return {
                    'success': False,
                    'error': 'No gallery available'
                }
            
            viz_data = manager.current_gallery.get_visualization(viz_type)
            if viz_data is None:
                return {
                    'success': False,
                    'error': f'Visualization {viz_type} not found'
                }
            
            return {
                'success': True,
                'viz_type': viz_type,
                'data': viz_data
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def handle_export(viz_type: str, format: str = 'png', output_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            path = manager.export_visualization(viz_type, format, output_path)
            return {
                'success': True,
                'path': path,
                'message': f'Exported {viz_type} to {path}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    return {
        'load_model': handle_load_model,
        'create_gallery': handle_create_gallery,
        'get_visualization': handle_get_visualization,
        'export': handle_export,
        'manager': manager
    }


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
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(example_dsl)
    gallery = manager.create_gallery()
    
    print("Gallery Metadata:")
    print(json.dumps(gallery.get_gallery_metadata(), indent=2))
    
    print("\n\nAvailable Visualizations:")
    for viz in manager.get_visualization_list():
        print(f"  - {viz['name']}: {viz['description']}")
    
    print("\n\nExporting visualizations...")
    paths = manager.export_all_visualizations(format='html', output_dir='output')
    for viz_type, path in paths.items():
        print(f"  {viz_type}: {path}")
