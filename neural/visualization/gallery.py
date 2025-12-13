from __future__ import annotations

from typing import Dict, List, Optional, Any
import json
import os

from neural.visualization.static_visualizer.visualizer import NeuralVisualizer
from neural.shape_propagation.shape_propagator import ShapePropagator


class VisualizationGallery:
    def __init__(self, model_data: Dict[str, Any], shape_propagator: Optional[ShapePropagator] = None):
        self.model_data = model_data
        self.shape_propagator = shape_propagator or ShapePropagator()
        self.visualizer = NeuralVisualizer(model_data)
        
        self.architecture_viz = ArchitectureVisualizer(self.visualizer, model_data)
        self.shape_viz = ShapePropagationVisualizer(self.visualizer, self.shape_propagator)
        self.flops_viz = FlopsMemoryVisualizer(self.shape_propagator)
        self.timeline_viz = TimelineVisualizer(self.shape_propagator)
        self.export_handler = ExportHandler()
        
        self.visualizations: Dict[str, Any] = {}
        
    def generate_all_visualizations(self, input_shape: tuple) -> Dict[str, Any]:
        self._propagate_shapes(input_shape)
        
        self.visualizations = {
            'architecture': self.architecture_viz.generate(),
            'shape_propagation': self.shape_viz.generate(),
            'flops_memory': self.flops_viz.generate(),
            'timeline': self.timeline_viz.generate(),
        }
        
        return self.visualizations
    
    def _propagate_shapes(self, input_shape: tuple):
        layers = self.model_data.get('layers', [])
        current_shape = input_shape
        
        for layer in layers:
            current_shape = self.shape_propagator.propagate(
                current_shape, 
                layer, 
                framework=self.model_data.get('framework', 'tensorflow')
            )
    
    def get_visualization(self, viz_type: str) -> Optional[Any]:
        return self.visualizations.get(viz_type)
    
    def get_all_visualizations(self) -> Dict[str, Any]:
        return self.visualizations
    
    def export_visualization(self, viz_type: str, format: str = 'png', output_path: Optional[str] = None) -> str:
        if viz_type not in self.visualizations:
            raise ValueError(f"Visualization type '{viz_type}' not found")
        
        viz_data = self.visualizations[viz_type]
        return self.export_handler.export(viz_data, viz_type, format, output_path)
    
    def export_all(self, format: str = 'png', output_dir: Optional[str] = None) -> Dict[str, str]:
        export_paths = {}
        for viz_type, viz_data in self.visualizations.items():
            output_path = None
            if output_dir:
                output_path = f"{output_dir}/{viz_type}.{format}"
            export_paths[viz_type] = self.export_handler.export(viz_data, viz_type, format, output_path)
        return export_paths
    
    def get_gallery_metadata(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_data.get('name', 'Unnamed Model'),
            'total_layers': len(self.model_data.get('layers', [])),
            'input_shape': self.model_data.get('input', {}).get('shape'),
            'output_shape': self.model_data.get('output_layer', {}).get('params', {}).get('units'),
            'visualizations_available': list(self.visualizations.keys()),
            'total_parameters': self._calculate_total_parameters(),
            'total_flops': self._calculate_total_flops(),
        }
    
    def _calculate_total_parameters(self) -> int:
        total = 0
        for entry in self.shape_propagator.shape_history:
            layer_name, shape = entry
            if shape and len(shape) > 1:
                params = 1
                for dim in shape[1:]:
                    if dim is not None:
                        params *= dim
                total += params
        return total
    
    def _calculate_total_flops(self) -> int:
        total = 0
        for entry in self.shape_propagator.execution_trace:
            flops = entry.get('flops', 0)
            total += flops
        return int(total)
    
    def to_json(self) -> str:
        gallery_data = {
            'metadata': self.get_gallery_metadata(),
            'visualizations': {
                viz_type: self._serialize_visualization(viz_data)
                for viz_type, viz_data in self.visualizations.items()
            }
        }
        return json.dumps(gallery_data, indent=2, default=str)
    
    def _serialize_visualization(self, viz_data: Any) -> Dict[str, Any]:
        if hasattr(viz_data, 'to_dict'):
            return viz_data.to_dict()
        elif hasattr(viz_data, 'to_json'):
            return json.loads(viz_data.to_json())
        elif isinstance(viz_data, dict):
            return viz_data
        else:
            return {'type': str(type(viz_data)), 'data': str(viz_data)}


class ArchitectureVisualizer:
    def __init__(self, visualizer: NeuralVisualizer, model_data: Dict[str, Any]):
        self.visualizer = visualizer
        self.model_data = model_data
        
    def generate(self) -> Dict[str, Any]:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        from graphviz import Digraph
        
        d3_data = self.visualizer.model_to_d3_json()
        nodes = d3_data['nodes']
        links = d3_data['links']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        node_positions = {}
        y_spacing = 1.5
        
        for i, node in enumerate(nodes):
            y_pos = len(nodes) - i - 1
            node_positions[node['id']] = (0.5, y_pos * y_spacing)
            
            color_map = {
                'Input': '#90EE90',
                'Conv2D': '#87CEEB',
                'MaxPooling2D': '#FFB6C1',
                'Flatten': '#DDA0DD',
                'Dense': '#F0E68C',
                'Output': '#FFA07A',
            }
            color = color_map.get(node['type'], '#D3D3D3')
            
            bbox = FancyBboxPatch(
                (0.2, y_pos * y_spacing - 0.4), 0.6, 0.8,
                boxstyle="round,pad=0.1", 
                facecolor=color, 
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(bbox)
            
            ax.text(0.5, y_pos * y_spacing + 0.1, 
                   node['type'], 
                   ha='center', va='center', 
                   fontweight='bold', fontsize=12)
            
            if 'params' in node and node['params']:
                param_text = ', '.join(f"{k}={v}" for k, v in list(node['params'].items())[:3])
                if len(node['params']) > 3:
                    param_text += '...'
                ax.text(0.5, y_pos * y_spacing - 0.15, 
                       param_text, 
                       ha='center', va='center', 
                       fontsize=8, style='italic')
            
            if 'shape' in node and node['shape']:
                shape_text = str(node['shape'])
                ax.text(0.5, y_pos * y_spacing - 0.3, 
                       shape_text, 
                       ha='center', va='center', 
                       fontsize=7, color='#555')
        
        for link in links:
            source_pos = node_positions.get(link['source'])
            target_pos = node_positions.get(link['target'])
            
            if source_pos and target_pos:
                ax.annotate('', 
                           xy=(target_pos[0], target_pos[1] + 0.4),
                           xytext=(source_pos[0], source_pos[1] - 0.4),
                           arrowprops=dict(
                               arrowstyle='->', 
                               lw=2, 
                               color='#333',
                               connectionstyle="arc3,rad=0"
                           ))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, len(nodes) * y_spacing)
        ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)
        
        dot = Digraph(comment='Neural Network', format='svg')
        dot.attr('node', shape='box', style='filled,rounded', fontname='Arial')
        dot.attr('graph', rankdir='TB', splines='ortho')
        
        for node in nodes:
            label = f"{node['type']}"
            if 'params' in node and node['params']:
                label += f"\\n{', '.join(f'{k}={v}' for k, v in list(node['params'].items())[:2])}"
            dot.node(node['id'], label, fillcolor=color_map.get(node['type'], '#D3D3D3'))
        
        for link in links:
            dot.edge(link['source'], link['target'])
        
        return {
            'matplotlib_figure': fig,
            'graphviz_graph': dot,
            'd3_data': d3_data,
            'type': 'architecture'
        }


class ShapePropagationVisualizer:
    def __init__(self, visualizer: NeuralVisualizer, shape_propagator: ShapePropagator):
        self.visualizer = visualizer
        self.shape_propagator = shape_propagator
        
    def generate(self) -> Dict[str, Any]:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        shape_history = self.shape_propagator.shape_history
        
        if not shape_history:
            shape_history = [
                ('Input', (None, 28, 28, 1)),
                ('Conv2D', (None, 26, 26, 32)),
                ('MaxPooling2D', (None, 13, 13, 32)),
                ('Flatten', (None, 5408)),
                ('Dense', (None, 128)),
                ('Output', (None, 10))
            ]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tensor Shape Evolution', 'Parameter Count per Layer'),
            specs=[[{'type': 'scatter3d'}], [{'type': 'bar'}]],
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        for i, (name, shape) in enumerate(shape_history):
            shape_clean = tuple(d if d is not None else 1 for d in shape)
            
            if len(shape_clean) > 1:
                fig.add_trace(
                    go.Scatter3d(
                        x=[i] * len(shape_clean),
                        y=list(range(len(shape_clean))),
                        z=list(shape_clean),
                        mode='markers+text',
                        text=[str(d) for d in shape],
                        textposition='top center',
                        name=name,
                        marker=dict(size=8, opacity=0.8),
                        hovertemplate=f'{name}<br>Dimension %{{y}}: %{{z}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        layer_names = [name for name, _ in shape_history]
        param_counts = []
        
        for _, shape in shape_history:
            shape_clean = tuple(d if d is not None else 1 for d in shape)
            if len(shape_clean) > 1:
                param_count = np.prod(shape_clean[1:])
            else:
                param_count = np.prod(shape_clean)
            param_counts.append(param_count)
        
        fig.add_trace(
            go.Bar(
                x=layer_names,
                y=param_counts,
                name='Parameters',
                marker=dict(
                    color=param_counts,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Params', x=1.1)
                ),
                hovertemplate='%{x}<br>Parameters: %{y:,}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_scenes(
            xaxis_title='Layer Index',
            yaxis_title='Dimension',
            zaxis_title='Size',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray', type='log'),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text='Layer', row=2, col=1)
        fig.update_yaxes(title_text='Parameter Count', type='log', row=2, col=1)
        
        fig.update_layout(
            height=900,
            title_text='Shape Propagation Analysis',
            showlegend=True,
            template='plotly_white'
        )
        
        flowchart = self._generate_flowchart(shape_history)
        
        return {
            'plotly_figure': fig,
            'shape_history': shape_history,
            'flowchart': flowchart,
            'type': 'shape_propagation'
        }
    
    def _generate_flowchart(self, shape_history: List[tuple]) -> str:
        flowchart = "```mermaid\ngraph TD\n"
        for i, (name, shape) in enumerate(shape_history):
            flowchart += f"    L{i}[\"{name}<br/>{shape}\"]\n"
        
        for i in range(len(shape_history) - 1):
            flowchart += f"    L{i} --> L{i+1}\n"
        
        flowchart += "```\n"
        return flowchart


class FlopsMemoryVisualizer:
    def __init__(self, shape_propagator: ShapePropagator):
        self.shape_propagator = shape_propagator
        
    def generate(self) -> Dict[str, Any]:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        trace_data = self.shape_propagator.execution_trace
        
        if not trace_data:
            trace_data = [
                {'layer': 'Conv2D', 'flops': 1.2e6, 'memory': 4.5},
                {'layer': 'MaxPooling2D', 'flops': 0, 'memory': 1.8},
                {'layer': 'Flatten', 'flops': 0, 'memory': 1.8},
                {'layer': 'Dense', 'flops': 8.0e5, 'memory': 0.5},
                {'layer': 'Output', 'flops': 1.3e3, 'memory': 0.04}
            ]
        
        layer_names = [entry.get('layer', 'Unknown') for entry in trace_data]
        flops = [entry.get('flops', 0) for entry in trace_data]
        memory = [entry.get('memory', 0) for entry in trace_data]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'FLOPs Distribution',
                'Memory Usage',
                'FLOPs per Layer',
                'Cumulative Memory'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        fig.add_trace(
            go.Pie(
                labels=layer_names,
                values=flops,
                name='FLOPs',
                hovertemplate='%{label}<br>FLOPs: %{value:,.0f}<br>%{percent}<extra></extra>',
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(
                labels=layer_names,
                values=memory,
                name='Memory',
                hovertemplate='%{label}<br>Memory: %{value:.2f} MB<br>%{percent}<extra></extra>',
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=layer_names,
                y=flops,
                name='FLOPs',
                marker=dict(
                    color=flops,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title='FLOPs', x=0.46, len=0.4, y=0.25)
                ),
                hovertemplate='%{x}<br>FLOPs: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        cumulative_memory = np.cumsum(memory)
        fig.add_trace(
            go.Scatter(
                x=layer_names,
                y=cumulative_memory,
                mode='lines+markers',
                name='Cumulative Memory',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                hovertemplate='%{x}<br>Cumulative: %{y:.2f} MB<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text='Layer', row=2, col=1)
        fig.update_xaxes(title_text='Layer', row=2, col=2)
        fig.update_yaxes(title_text='FLOPs', type='log', row=2, col=1)
        fig.update_yaxes(title_text='Memory (MB)', row=2, col=2)
        
        fig.update_layout(
            height=900,
            title_text='Computational Complexity Analysis',
            showlegend=True,
            template='plotly_white'
        )
        
        total_flops = sum(flops)
        total_memory = sum(memory)
        peak_memory = max(cumulative_memory) if len(cumulative_memory) > 0 else 0
        
        summary = {
            'total_flops': total_flops,
            'total_memory_mb': total_memory,
            'peak_memory_mb': peak_memory,
            'flops_per_layer': list(zip(layer_names, flops)),
            'memory_per_layer': list(zip(layer_names, memory))
        }
        
        return {
            'plotly_figure': fig,
            'summary': summary,
            'type': 'flops_memory'
        }


class TimelineVisualizer:
    def __init__(self, shape_propagator: ShapePropagator):
        self.shape_propagator = shape_propagator
        
    def generate(self) -> Dict[str, Any]:
        import plotly.figure_factory as ff
        import plotly.graph_objects as go
        import pandas as pd
        from datetime import datetime, timedelta
        
        trace_data = self.shape_propagator.execution_trace
        
        if not trace_data:
            trace_data = [
                {'layer': 'Conv2D', 'compute_time': 0.015, 'transfer_time': 0.002},
                {'layer': 'MaxPooling2D', 'compute_time': 0.003, 'transfer_time': 0.001},
                {'layer': 'Flatten', 'compute_time': 0.0001, 'transfer_time': 0.0001},
                {'layer': 'Dense', 'compute_time': 0.008, 'transfer_time': 0.001},
                {'layer': 'Output', 'compute_time': 0.001, 'transfer_time': 0.0001}
            ]
        
        base_time = datetime.now()
        timeline_data = []
        current_time = base_time
        
        for i, entry in enumerate(trace_data):
            layer_name = entry.get('layer', f'Layer {i}')
            compute_time = entry.get('compute_time', 0) * 1000
            transfer_time = entry.get('transfer_time', 0) * 1000
            
            compute_start = current_time
            compute_end = compute_start + timedelta(milliseconds=compute_time)
            
            timeline_data.append(dict(
                Task=layer_name,
                Start=compute_start,
                Finish=compute_end,
                Resource='Computation'
            ))
            
            transfer_start = compute_end
            transfer_end = transfer_start + timedelta(milliseconds=transfer_time)
            
            timeline_data.append(dict(
                Task=layer_name,
                Start=transfer_start,
                Finish=transfer_end,
                Resource='Data Transfer'
            ))
            
            current_time = transfer_end
        
        colors = {
            'Computation': '#4ECDC4',
            'Data Transfer': '#FFB6C1'
        }
        
        fig = ff.create_gantt(
            timeline_data,
            colors=colors,
            index_col='Resource',
            show_colorbar=True,
            group_tasks=True,
            showgrid_x=True,
            showgrid_y=True,
            title='Layer Computation Timeline'
        )
        
        fig.update_layout(
            height=max(600, len(trace_data) * 40),
            xaxis_title='Time (ms)',
            yaxis_title='Layer',
            template='plotly_white',
            font=dict(size=12)
        )
        
        layer_times = []
        for entry in trace_data:
            layer_name = entry.get('layer', 'Unknown')
            compute = entry.get('compute_time', 0) * 1000
            transfer = entry.get('transfer_time', 0) * 1000
            total = compute + transfer
            layer_times.append({
                'layer': layer_name,
                'compute_ms': compute,
                'transfer_ms': transfer,
                'total_ms': total
            })
        
        breakdown_fig = go.Figure()
        
        layers = [lt['layer'] for lt in layer_times]
        compute_times = [lt['compute_ms'] for lt in layer_times]
        transfer_times = [lt['transfer_ms'] for lt in layer_times]
        
        breakdown_fig.add_trace(go.Bar(
            name='Computation',
            x=layers,
            y=compute_times,
            marker_color='#4ECDC4',
            hovertemplate='%{x}<br>Compute: %{y:.3f} ms<extra></extra>'
        ))
        
        breakdown_fig.add_trace(go.Bar(
            name='Data Transfer',
            x=layers,
            y=transfer_times,
            marker_color='#FFB6C1',
            hovertemplate='%{x}<br>Transfer: %{y:.3f} ms<extra></extra>'
        ))
        
        breakdown_fig.update_layout(
            barmode='stack',
            title='Computation vs Transfer Time Breakdown',
            xaxis_title='Layer',
            yaxis_title='Time (ms)',
            height=500,
            template='plotly_white'
        )
        
        total_time = sum(lt['total_ms'] for lt in layer_times)
        total_compute = sum(lt['compute_ms'] for lt in layer_times)
        total_transfer = sum(lt['transfer_ms'] for lt in layer_times)
        
        return {
            'timeline_figure': fig,
            'breakdown_figure': breakdown_fig,
            'layer_times': layer_times,
            'summary': {
                'total_time_ms': total_time,
                'total_compute_ms': total_compute,
                'total_transfer_ms': total_transfer
            },
            'type': 'timeline'
        }


class ExportHandler:
    def export(self, viz_data: Dict[str, Any], viz_type: str, format: str, output_path: Optional[str] = None) -> str:
        if format == 'png':
            return self._export_png(viz_data, viz_type, output_path)
        elif format == 'svg':
            return self._export_svg(viz_data, viz_type, output_path)
        elif format == 'html':
            return self._export_html(viz_data, viz_type, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_png(self, viz_data: Dict[str, Any], viz_type: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = f"{viz_type}.png"
        
        if 'matplotlib_figure' in viz_data:
            fig = viz_data['matplotlib_figure']
            fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        elif 'plotly_figure' in viz_data:
            import plotly.io as pio
            fig = viz_data['plotly_figure']
            pio.write_image(fig, output_path, format='png', width=1920, height=1080)
        elif 'timeline_figure' in viz_data:
            import plotly.io as pio
            fig = viz_data['timeline_figure']
            pio.write_image(fig, output_path, format='png', width=1920, height=1080)
        else:
            raise ValueError(f"No exportable figure found in {viz_type} visualization")
        
        return output_path
    
    def _export_svg(self, viz_data: Dict[str, Any], viz_type: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = f"{viz_type}.svg"
        
        if 'matplotlib_figure' in viz_data:
            fig = viz_data['matplotlib_figure']
            fig.savefig(output_path, format='svg', bbox_inches='tight')
        elif 'graphviz_graph' in viz_data:
            dot = viz_data['graphviz_graph']
            dot.render(output_path.replace('.svg', ''), format='svg', cleanup=True)
        elif 'plotly_figure' in viz_data:
            import plotly.io as pio
            fig = viz_data['plotly_figure']
            pio.write_image(fig, output_path, format='svg')
        elif 'timeline_figure' in viz_data:
            import plotly.io as pio
            fig = viz_data['timeline_figure']
            pio.write_image(fig, output_path, format='svg')
        else:
            raise ValueError(f"No exportable figure found in {viz_type} visualization")
        
        return output_path
    
    def _export_html(self, viz_data: Dict[str, Any], viz_type: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = f"{viz_type}.html"
        
        if 'plotly_figure' in viz_data:
            fig = viz_data['plotly_figure']
            fig.write_html(output_path, include_plotlyjs='cdn')
        elif 'timeline_figure' in viz_data:
            fig = viz_data['timeline_figure']
            fig.write_html(output_path, include_plotlyjs='cdn')
        elif 'matplotlib_figure' in viz_data:
            import mpld3
            fig = viz_data['matplotlib_figure']
            html_str = mpld3.fig_to_html(fig)
            with open(output_path, 'w') as f:
                f.write(html_str)
        else:
            html_content = self._generate_html_template(viz_data, viz_type)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return output_path
    
    def _generate_html_template(self, viz_data: Dict[str, Any], viz_type: str) -> str:
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{viz_type.replace('_', ' ').title()} Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            margin-bottom: 30px;
            text-align: center;
        }}
        .viz-data {{
            background: #f8f9fa;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
            white-space: pre-wrap;
            font-family: monospace;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{viz_type.replace('_', ' ').title()} Visualization</h1>
        <div class="viz-data">{json.dumps(viz_data, indent=2, default=str)}</div>
    </div>
</body>
</html>"""
        return html
