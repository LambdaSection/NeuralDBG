"""
Backend of the Dynamic Visualizer.

Provides data transformation logic that converts Neural DSL model structures
into the JSON format that D3.js expects. The model_to_d3_json method specifically
creates the nodes and links structure that D3.js uses.
"""
from __future__ import annotations

from typing import Any, Dict, List

from neural.exceptions import DependencyError
from neural.parser.parser import ModelTransformer, create_parser
from neural.utils.logging import get_logger


logger = get_logger(__name__)


# Lazy load heavy dependencies
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    Digraph = None
    GRAPHVIZ_AVAILABLE = False

try:
    from matplotlib import pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Make tensorflow optional - allows tests to run without it
try:
    import keras
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    TENSORFLOW_AVAILABLE = False


class NeuralVisualizer:
    def __init__(self, model_data: Dict[str, Any]) -> None:
        self.model_data = model_data
        self.figures: List[Any] = []

    def model_to_d3_json(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert parsed model data to D3 visualization format"""
        nodes = []
        links = []

        input_data = self.model_data.get('input', {})
        nodes.append({
            "id": "input",
            "type": "Input",
            "shape": input_data.get('shape', None)
        })

        layers = self.model_data.get('layers', [])
        for idx, layer in enumerate(layers):
            node_id = f"layer{idx+1}"
            nodes.append({
                "id": node_id,
                "type": layer.get('type', 'Unknown'),
                "params": layer.get('params', {})
            })

            prev_node = "input" if idx == 0 else f"layer{idx}"
            links.append({
                "source": prev_node,
                "target": node_id
            })

        output_layer = self.model_data.get('output_layer', {})
        nodes.append({
            "id": "output",
            "type": output_layer.get('type', 'Output'),
            "params": output_layer.get('params', {})
        })

        if layers:
            links.append({
                "source": f"layer{len(layers)}",
                "target": "output"
            })

        return {"nodes": nodes, "links": links}

    def create_3d_visualization(self, shape_history):
        """Create a 3D visualization of the shape propagation through the network."""
        if not PLOTLY_AVAILABLE:
            raise DependencyError(
                dependency="plotly",
                feature="3D visualization",
                install_hint="pip install plotly"
            )
        
        if not shape_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No shape history available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

        fig = go.Figure()

        for i, (name, shape) in enumerate(shape_history):
            if not shape:
                continue
            
            clean_shape = []
            for dim in shape:
                if dim is None:
                    clean_shape.append(-1)
                else:
                    clean_shape.append(dim)
            
            if not clean_shape:
                continue
            
            fig.add_trace(go.Scatter3d(
                x=[i]*len(clean_shape),
                y=list(range(len(clean_shape))),
                z=clean_shape,
                mode='markers+text',
                text=[str(d) if d != -1 else 'None' for d in clean_shape],
                name=name,
                marker=dict(size=8)
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='Layer Depth',
                yaxis_title='Dimension Index',
                zaxis_title='Dimension Size'
            ),
            title='Shape Propagation Through Network'
        )
        return fig

    def save_architecture_diagram(self, filename):
        """Save the architecture diagram to a file.

        Args:
            filename: The name of the file to save the diagram to.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise DependencyError(
                dependency="matplotlib",
                feature="architecture diagram",
                install_hint="pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=(10, 8))

        d3_data = self.model_to_d3_json()
        nodes = d3_data['nodes']
        links = d3_data['links']

        if not nodes:
            ax.text(0.5, 0.5, 'No architecture to visualize', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return

        for i, node in enumerate(nodes):
            y_pos = len(nodes) - i - 1
            ax.add_patch(plt.Rectangle((0.2, y_pos - 0.4), 0.6, 0.8, fill=True,
                                      color='lightblue', alpha=0.7))
            ax.text(0.5, y_pos, f"{node['type']}", ha='center', va='center', fontweight='bold')

            if 'params' in node and node['params']:
                param_text = ', '.join(f"{k}={v}" for k, v in node['params'].items())
                ax.text(0.5, y_pos - 0.2, param_text, ha='center', va='center', fontsize=8)

        for link in links:
            source_idx = next((i for i, n in enumerate(nodes) if n['id'] == link['source']), None)
            target_idx = next((i for i, n in enumerate(nodes) if n['id'] == link['target']), None)
            
            if source_idx is None or target_idx is None:
                continue
                
            source_y = len(nodes) - source_idx - 1
            target_y = len(nodes) - target_idx - 1
            
            y_diff = target_y - source_y
            arrow_start_y = source_y - 0.4
            arrow_length = y_diff + 0.8
            
            if abs(arrow_length) > 0.1:
                ax.arrow(0.5, arrow_start_y, 0, arrow_length,
                        head_width=0.05, head_length=0.1, fc='black', ec='black',
                        length_includes_head=True)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(nodes) - 0.5)
        ax.axis('off')

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_shape_visualization(self, fig, filename: str) -> None:
        """Save the shape visualization to an HTML file.

        Args:
            fig: The plotly figure to save.
            filename: The name of the file to save the visualization to.
        """
        if not PLOTLY_AVAILABLE:
            raise DependencyError(
                dependency="plotly",
                feature="shape visualization export",
                install_hint="pip install plotly"
            )
        
        import plotly.io
        plotly.io.write_html(fig, filename)


##### EXAMPLE ########


if __name__ == '__main__':
    nr_content = """
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

    parser = create_parser('network')
    parsed = parser.parse(nr_content)
    model_data = ModelTransformer().transform(parsed)

    visualizer = NeuralVisualizer(model_data)
    d3_json = visualizer.model_to_d3_json()
    logger.info(f"Generated D3 JSON: {d3_json}")
