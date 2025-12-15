from __future__ import annotations

import os

import click

from neural.visualization.aquarium_integration import AquariumVisualizationManager
from neural.visualization.aquarium_server import start_server


@click.group()
def gallery():
    pass


@gallery.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.option('--input-shape', '-i', help='Input shape as tuple (e.g., "None,28,28,1")')
@click.option('--output-dir', '-o', default='visualizations', help='Output directory for visualizations')
@click.option('--format', '-f', type=click.Choice(['png', 'svg', 'html']), default='html', help='Export format')
def generate(model_file: str, input_shape: str, output_dir: str, format: str):
    with open(model_file, 'r') as f:
        dsl_code = f.read()
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    
    if input_shape:
        input_shape_tuple = tuple(None if x.strip().lower() == 'none' else int(x.strip()) 
                                  for x in input_shape.split(','))
    else:
        input_shape_tuple = None
    
    click.echo("🎨 Creating visualization gallery...")
    gallery = manager.create_gallery(input_shape_tuple)
    
    os.makedirs(output_dir, exist_ok=True)
    
    click.echo(f"📊 Exporting visualizations to {output_dir}/")
    paths = manager.export_all_visualizations(format=format, output_dir=output_dir)
    
    click.echo("\n✅ Visualizations created:")
    for viz_type, path in paths.items():
        click.echo(f"  - {viz_type}: {path}")
    
    metadata = gallery.get_gallery_metadata()
    click.echo("\n📈 Model Statistics:")
    click.echo(f"  Total Layers: {metadata['total_layers']}")
    click.echo(f"  Total Parameters: {metadata['total_parameters']:,}")
    click.echo(f"  Total FLOPs: {metadata['total_flops']:,}")


@gallery.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.option('--port', '-p', default=8052, help='Server port')
@click.option('--host', '-h', default='0.0.0.0', help='Server host')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def serve(model_file: str, port: int, host: str, debug: bool):
    with open(model_file, 'r') as f:
        dsl_code = f.read()
    
    manager = AquariumVisualizationManager()
    click.echo("📚 Loading model...")
    manager.load_model_from_dsl(dsl_code)
    
    click.echo("🎨 Creating visualization gallery...")
    gallery = manager.create_gallery()
    
    metadata = gallery.get_gallery_metadata()
    click.echo(f"\n✅ Gallery created with {len(gallery.visualizations)} visualizations")
    click.echo(f"📊 Model: {metadata['model_name']}")
    click.echo(f"🧠 Layers: {metadata['total_layers']}")
    
    click.echo(f"\n🚀 Starting server on http://{host}:{port}")
    click.echo(f"   Open http://localhost:{port} in your browser")
    
    from neural.visualization.aquarium_server import AquariumVisualizationServer
    server = AquariumVisualizationServer(host=host, port=port)
    server.manager = manager
    server.run(debug=debug)


@gallery.command()
@click.option('--port', '-p', default=8052, help='Server port')
@click.option('--host', '-h', default='0.0.0.0', help='Server host')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def server(port: int, host: str, debug: bool):
    click.echo("🚀 Starting Neural Visualization Gallery server...")
    click.echo(f"   Open http://localhost:{port} in your browser")
    start_server(host=host, port=port, debug=debug)


@gallery.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='gallery.json', help='Output JSON file')
def export_json(model_file: str, output: str):
    with open(model_file, 'r') as f:
        dsl_code = f.read()
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    gallery = manager.create_gallery()
    
    gallery_json = gallery.to_json()
    
    with open(output, 'w') as f:
        f.write(gallery_json)
    
    click.echo(f"✅ Gallery metadata exported to {output}")


@gallery.command()
@click.argument('model_file', type=click.Path(exists=True))
def info(model_file: str):
    with open(model_file, 'r') as f:
        dsl_code = f.read()
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    gallery = manager.create_gallery()
    
    metadata = gallery.get_gallery_metadata()
    
    click.echo("\n" + "="*60)
    click.echo("🎨 Neural Visualization Gallery - Model Information")
    click.echo("="*60)
    
    click.echo(f"\n📊 Model: {metadata['model_name']}")
    click.echo("🏗️  Architecture:")
    click.echo(f"   - Total Layers: {metadata['total_layers']}")
    click.echo(f"   - Input Shape: {metadata['input_shape']}")
    click.echo(f"   - Output Shape: {metadata['output_shape']}")
    
    click.echo("\n🧠 Computational Complexity:")
    click.echo(f"   - Total Parameters: {metadata['total_parameters']:,}")
    click.echo(f"   - Total FLOPs: {metadata['total_flops']:,}")
    
    click.echo("\n📈 Available Visualizations:")
    for viz_name in metadata['visualizations_available']:
        click.echo(f"   - {viz_name.replace('_', ' ').title()}")
    
    click.echo("\n" + "="*60)


if __name__ == '__main__':
    gallery()
