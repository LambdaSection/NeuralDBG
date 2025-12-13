from neural.visualization.static_visualizer.visualizer import NeuralVisualizer

try:
    from neural.visualization.gallery import (
        VisualizationGallery,
        ArchitectureVisualizer,
        ShapePropagationVisualizer,
        FlopsMemoryVisualizer,
        TimelineVisualizer,
        ExportHandler,
    )
    from neural.visualization.aquarium_integration import AquariumVisualizationManager
    from neural.visualization.aquarium_web_components import AquariumWebComponentRenderer
    from neural.visualization.aquarium_server import AquariumVisualizationServer, start_server
    from neural.visualization.component_interface import (
        ComponentInterface,
        VisualizationMetadata,
        GalleryMetadata,
        create_component_interface,
        get_visualization_preview_html,
    )
    
    GALLERY_AVAILABLE = True
except ImportError as e:
    GALLERY_AVAILABLE = False
    print(f"Gallery components not available: {e}")

__all__ = ['NeuralVisualizer']

if GALLERY_AVAILABLE:
    __all__.extend([
        'VisualizationGallery',
        'ArchitectureVisualizer',
        'ShapePropagationVisualizer',
        'FlopsMemoryVisualizer',
        'TimelineVisualizer',
        'ExportHandler',
        'AquariumVisualizationManager',
        'AquariumWebComponentRenderer',
        'AquariumVisualizationServer',
        'start_server',
        'ComponentInterface',
        'VisualizationMetadata',
        'GalleryMetadata',
        'create_component_interface',
        'get_visualization_preview_html',
    ])
