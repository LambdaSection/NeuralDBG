"""
Advanced example: Building custom architectures with the No-Code Interface API

This demonstrates:
- Building a ResNet-style architecture
- Using complex layer configurations
- Real-time validation during construction
- Exporting to multiple backends
"""

import requests
import json


API_BASE = "http://localhost:8051"


def build_resnet_block(filters, stride=1):
    """Build a ResNet-style residual block"""
    
    block_layers = [
        {
            "type": "Conv2D",
            "params": {
                "filters": filters,
                "kernel_size": [3, 3],
                "strides": [stride, stride],
                "padding": "same",
                "activation": None
            }
        },
        {
            "type": "BatchNormalization",
            "params": {}
        },
        {
            "type": "ReLU",
            "params": {}
        },
        {
            "type": "Conv2D",
            "params": {
                "filters": filters,
                "kernel_size": [3, 3],
                "padding": "same",
                "activation": None
            }
        },
        {
            "type": "BatchNormalization",
            "params": {}
        }
    ]
    
    return block_layers


def build_transformer_encoder():
    """Build a Transformer encoder block"""
    
    return [
        {
            "type": "Embedding",
            "params": {
                "input_dim": 10000,
                "output_dim": 256
            }
        },
        {
            "type": "MultiHeadAttention",
            "params": {
                "num_heads": 8,
                "key_dim": 32
            }
        },
        {
            "type": "LayerNormalization",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": 512,
                "activation": "relu"
            }
        },
        {
            "type": "Dense",
            "params": {
                "units": 256
            }
        },
        {
            "type": "LayerNormalization",
            "params": {}
        },
        {
            "type": "GlobalAveragePooling1D",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": 64,
                "activation": "relu"
            }
        },
        {
            "type": "Dense",
            "params": {
                "units": 2,
                "activation": "softmax"
            }
        }
    ]


def build_efficient_net_style():
    """Build an EfficientNet-style architecture"""
    
    layers = [
        {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": [3, 3],
                "strides": [2, 2],
                "padding": "same",
                "activation": None
            }
        },
        {
            "type": "BatchNormalization",
            "params": {}
        },
        {
            "type": "ReLU",
            "params": {}
        }
    ]
    
    # Add MBConv blocks (approximation with available layers)
    for filters in [16, 24, 40, 80]:
        layers.extend([
            {
                "type": "DepthwiseConv2D",
                "params": {
                    "kernel_size": [3, 3],
                    "padding": "same"
                }
            },
            {
                "type": "BatchNormalization",
                "params": {}
            },
            {
                "type": "ReLU",
                "params": {}
            },
            {
                "type": "Conv2D",
                "params": {
                    "filters": filters,
                    "kernel_size": [1, 1],
                    "activation": None
                }
            },
            {
                "type": "BatchNormalization",
                "params": {}
            }
        ])
    
    # Classification head
    layers.extend([
        {
            "type": "GlobalAveragePooling2D",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": 1000,
                "activation": "softmax"
            }
        }
    ])
    
    return layers


def validate_and_print(layers, input_shape, name):
    """Validate model and print results"""
    
    print(f"\n{'='*70}")
    print(f"Validating: {name}")
    print(f"{'='*70}")
    
    response = requests.post(
        f"{API_BASE}/api/validate",
        json={
            "input_shape": input_shape,
            "layers": layers
        }
    )
    
    result = response.json()
    
    if result["valid"]:
        print("✓ Model is VALID")
        print("\nShape Propagation:")
        for i, shape_info in enumerate(result["shapes"]):
            layer_name = shape_info["layer"]
            shape = shape_info["shape"]
            
            if i == 0:
                print(f"  {layer_name:30s} → {shape}")
            else:
                print(f"  {layer_name:30s} → {shape}")
    else:
        print("✗ Model has ERRORS:")
        for error in result["errors"]:
            print(f"  - {error['message']}")
        
        if result["warnings"]:
            print("\nWarnings:")
            for warning in result["warnings"]:
                print(f"  - {warning['message']}")
    
    return result


def generate_and_save_code(layers, input_shape, name, filename):
    """Generate code and save to files"""
    
    print(f"\nGenerating code for {name}...")
    
    response = requests.post(
        f"{API_BASE}/api/generate-code",
        json={
            "input_shape": input_shape,
            "layers": layers,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999
                }
            },
            "loss": "categorical_crossentropy"
        }
    )
    
    code = response.json()
    
    # Save DSL code
    with open(f"{filename}.neural", "w") as f:
        f.write(code["dsl"])
    print(f"  ✓ Saved DSL code to {filename}.neural")
    
    # Save TensorFlow code
    with open(f"{filename}_tf.py", "w") as f:
        f.write(code["tensorflow"])
    print(f"  ✓ Saved TensorFlow code to {filename}_tf.py")
    
    # Save PyTorch code
    with open(f"{filename}_torch.py", "w") as f:
        f.write(code["pytorch"])
    print(f"  ✓ Saved PyTorch code to {filename}_torch.py")


def main():
    """Run advanced examples"""
    
    print("🚀 Neural DSL No-Code Interface - Advanced Examples")
    print("="*70)
    
    # Example 1: ResNet-style architecture
    print("\n📦 Example 1: ResNet-Style Architecture")
    resnet_layers = [
        {
            "type": "Conv2D",
            "params": {
                "filters": 64,
                "kernel_size": [7, 7],
                "strides": [2, 2],
                "padding": "same"
            }
        },
        {
            "type": "BatchNormalization",
            "params": {}
        },
        {
            "type": "ReLU",
            "params": {}
        },
        {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": [3, 3],
                "strides": [2, 2],
                "padding": "same"
            }
        }
    ]
    
    # Add residual blocks
    resnet_layers.extend(build_resnet_block(64))
    resnet_layers.extend(build_resnet_block(128, stride=2))
    resnet_layers.extend(build_resnet_block(256, stride=2))
    
    # Classification head
    resnet_layers.extend([
        {
            "type": "GlobalAveragePooling2D",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": 1000,
                "activation": "softmax"
            }
        }
    ])
    
    result1 = validate_and_print(resnet_layers, [None, 224, 224, 3], "ResNet-Style")
    if result1["valid"]:
        generate_and_save_code(resnet_layers, [None, 224, 224, 3], "ResNet-Style", "resnet_example")
    
    # Example 2: Transformer Encoder
    print("\n\n📦 Example 2: Transformer Encoder")
    transformer_layers = build_transformer_encoder()
    
    result2 = validate_and_print(transformer_layers, [None, 512], "Transformer Encoder")
    if result2["valid"]:
        generate_and_save_code(transformer_layers, [None, 512], "Transformer", "transformer_example")
    
    # Example 3: EfficientNet-style
    print("\n\n📦 Example 3: EfficientNet-Style Architecture")
    efficient_layers = build_efficient_net_style()
    
    result3 = validate_and_print(efficient_layers, [None, 224, 224, 3], "EfficientNet-Style")
    if result3["valid"]:
        generate_and_save_code(efficient_layers, [None, 224, 224, 3], "EfficientNet", "efficientnet_example")
    
    # Summary
    print("\n\n" + "="*70)
    print("📊 Summary")
    print("="*70)
    print(f"ResNet-Style:       {'✓ Valid' if result1['valid'] else '✗ Invalid'} ({len(resnet_layers)} layers)")
    print(f"Transformer:        {'✓ Valid' if result2['valid'] else '✗ Invalid'} ({len(transformer_layers)} layers)")
    print(f"EfficientNet-Style: {'✓ Valid' if result3['valid'] else '✗ Invalid'} ({len(efficient_layers)} layers)")
    print("\n✨ Advanced examples completed!")


if __name__ == "__main__":
    import sys
    
    print("Note: Make sure the no-code interface is running before executing this script.")
    print("Start it with: python neural/no_code/app.py\n")
    
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the no-code interface.")
        print("Please start the server first: python neural/no_code/app.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
