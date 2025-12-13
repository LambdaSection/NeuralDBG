from __future__ import annotations

import os
from neural.visualization.aquarium_integration import AquariumVisualizationManager


def example_basic_usage():
    print("="*80)
    print("Example 1: Basic Usage - Generate All Visualizations")
    print("="*80)
    
    dsl_code = """
    network MnistClassifier {
        input: (None, 28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=(3,3), activation="relu")
            MaxPooling2D(pool_size=(2,2))
            Conv2D(filters=64, kernel_size=(3,3), activation="relu")
            MaxPooling2D(pool_size=(2,2))
            Flatten()
            Dense(128, activation="relu")
            Output(10, activation="softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    gallery = manager.create_gallery()
    
    metadata = gallery.get_gallery_metadata()
    print(f"\n📊 Model: {metadata['model_name']}")
    print(f"🏗️  Total Layers: {metadata['total_layers']}")
    print(f"🧠 Total Parameters: {metadata['total_parameters']:,}")
    print(f"⚡ Total FLOPs: {metadata['total_flops']:,}")
    
    visualizations = manager.get_visualization_list()
    print(f"\n📈 Available Visualizations ({len(visualizations)}):")
    for viz in visualizations:
        print(f"  - {viz['name']}: {viz['description']}")
    
    print("\n✅ Example 1 Complete\n")


def example_export_visualizations():
    print("="*80)
    print("Example 2: Export Visualizations to Different Formats")
    print("="*80)
    
    dsl_code = """
    network SimpleNet {
        input: (None, 784)
        layers:
            Dense(256, activation="relu")
            Dense(128, activation="relu")
            Output(10, activation="softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    gallery = manager.create_gallery()
    
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📤 Exporting visualizations...")
    
    for format in ['png', 'svg', 'html']:
        format_dir = f"{output_dir}/{format}"
        os.makedirs(format_dir, exist_ok=True)
        
        print(f"\n  Exporting as {format.upper()}...")
        paths = manager.export_all_visualizations(format=format, output_dir=format_dir)
        
        for viz_type, path in paths.items():
            print(f"    ✓ {viz_type}: {path}")
    
    print("\n✅ Example 2 Complete\n")


def example_specific_visualizations():
    print("="*80)
    print("Example 3: Working with Specific Visualizations")
    print("="*80)
    
    dsl_code = """
    network ResNetBlock {
        input: (None, 56, 56, 64)
        layers:
            Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")
            Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")
            Output(64, activation="linear")
        loss: "mse"
        optimizer: "adam"
    }
    """
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    gallery = manager.create_gallery()
    
    print("\n🏗️  Architecture Visualization:")
    arch_viz = gallery.get_visualization('architecture')
    if arch_viz:
        print(f"  - Type: {arch_viz['type']}")
        print(f"  - Contains matplotlib figure: {'matplotlib_figure' in arch_viz}")
        print(f"  - Contains graphviz graph: {'graphviz_graph' in arch_viz}")
        print(f"  - Contains D3 data: {'d3_data' in arch_viz}")
    
    print("\n📊 Shape Propagation Visualization:")
    shape_viz = gallery.get_visualization('shape_propagation')
    if shape_viz:
        print(f"  - Type: {shape_viz['type']}")
        print(f"  - Number of layers: {len(shape_viz.get('shape_history', []))}")
        for i, (layer_name, shape) in enumerate(shape_viz.get('shape_history', [])):
            print(f"    Layer {i}: {layer_name} -> {shape}")
    
    print("\n💾 FLOPs & Memory Visualization:")
    flops_viz = gallery.get_visualization('flops_memory')
    if flops_viz:
        summary = flops_viz.get('summary', {})
        print(f"  - Total FLOPs: {summary.get('total_flops', 0):,.0f}")
        print(f"  - Total Memory: {summary.get('total_memory_mb', 0):.2f} MB")
        print(f"  - Peak Memory: {summary.get('peak_memory_mb', 0):.2f} MB")
    
    print("\n⏱️  Timeline Visualization:")
    timeline_viz = gallery.get_visualization('timeline')
    if timeline_viz:
        summary = timeline_viz.get('summary', {})
        print(f"  - Total Time: {summary.get('total_time_ms', 0):.3f} ms")
        print(f"  - Compute Time: {summary.get('total_compute_ms', 0):.3f} ms")
        print(f"  - Transfer Time: {summary.get('total_transfer_ms', 0):.3f} ms")
    
    print("\n✅ Example 3 Complete\n")


def example_gallery_json():
    print("="*80)
    print("Example 4: Export Gallery as JSON")
    print("="*80)
    
    dsl_code = """
    network TinyNet {
        input: (None, 32, 32, 3)
        layers:
            Conv2D(filters=16, kernel_size=(3,3), activation="relu")
            MaxPooling2D(pool_size=(2,2))
            Flatten()
            Dense(10, activation="softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    gallery = manager.create_gallery()
    
    gallery_json = gallery.to_json()
    
    output_file = "example_output/gallery.json"
    os.makedirs("example_output", exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(gallery_json)
    
    print(f"\n📄 Gallery JSON exported to: {output_file}")
    print(f"📊 File size: {len(gallery_json):,} bytes")
    
    import json
    gallery_data = json.loads(gallery_json)
    print(f"\n🔍 Gallery Contents:")
    print(f"  - Model Name: {gallery_data['metadata']['model_name']}")
    print(f"  - Visualizations: {len(gallery_data['visualizations'])}")
    for viz_type in gallery_data['visualizations'].keys():
        print(f"    • {viz_type}")
    
    print("\n✅ Example 4 Complete\n")


def example_custom_input_shape():
    print("="*80)
    print("Example 5: Custom Input Shapes")
    print("="*80)
    
    dsl_code = """
    network FlexibleNet {
        input: (None, 224, 224, 3)
        layers:
            Conv2D(filters=64, kernel_size=(7,7), strides=2, activation="relu")
            MaxPooling2D(pool_size=(3,3), strides=2)
            Conv2D(filters=128, kernel_size=(3,3), activation="relu")
            GlobalAveragePooling2D()
            Dense(1000, activation="softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    
    manager = AquariumVisualizationManager()
    manager.load_model_from_dsl(dsl_code)
    
    input_shapes = [
        (1, 224, 224, 3),
        (8, 224, 224, 3),
        (32, 224, 224, 3)
    ]
    
    for batch_size, *spatial_dims in input_shapes:
        input_shape = (batch_size, *spatial_dims)
        print(f"\n📊 Creating gallery with input shape: {input_shape}")
        
        gallery = manager.create_gallery(input_shape=input_shape)
        metadata = gallery.get_gallery_metadata()
        
        print(f"  - Total Parameters: {metadata['total_parameters']:,}")
        print(f"  - Total FLOPs: {metadata['total_flops']:,}")
    
    print("\n✅ Example 5 Complete\n")


def run_all_examples():
    print("\n" + "="*80)
    print("🎨 Neural Visualization Gallery - Examples")
    print("="*80 + "\n")
    
    try:
        example_basic_usage()
        example_export_visualizations()
        example_specific_visualizations()
        example_gallery_json()
        example_custom_input_shape()
        
        print("="*80)
        print("🎉 All Examples Completed Successfully!")
        print("="*80)
        print("\nCheck the 'example_output' directory for exported visualizations.")
        print("To start the web server, run:")
        print("  python -m neural.visualization.gallery_cli serve <model.neural>")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_examples()
