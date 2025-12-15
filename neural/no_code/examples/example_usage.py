"""
Example usage of the No-Code Interface programmatically

This demonstrates how to interact with the no-code interface API
to build models, validate them, and generate code.
"""

import requests


API_BASE = "http://localhost:8051"


def create_simple_cnn():
    """Create a simple CNN model using the API"""
    
    layers = [
        {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": [3, 3],
                "activation": "relu"
            }
        },
        {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": [2, 2]
            }
        },
        {
            "type": "Conv2D",
            "params": {
                "filters": 64,
                "kernel_size": [3, 3],
                "activation": "relu"
            }
        },
        {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": [2, 2]
            }
        },
        {
            "type": "Flatten",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": 128,
                "activation": "relu"
            }
        },
        {
            "type": "Dropout",
            "params": {
                "rate": 0.5
            }
        },
        {
            "type": "Dense",
            "params": {
                "units": 10,
                "activation": "softmax"
            }
        }
    ]
    
    return layers


def validate_model(layers, input_shape=[None, 28, 28, 1]):
    """Validate a model using the API"""
    
    response = requests.post(
        f"{API_BASE}/api/validate",
        json={
            "input_shape": input_shape,
            "layers": layers
        }
    )
    
    return response.json()


def generate_code(layers, input_shape=[None, 28, 28, 1]):
    """Generate code for a model using the API"""
    
    response = requests.post(
        f"{API_BASE}/api/generate-code",
        json={
            "input_shape": input_shape,
            "layers": layers,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "learning_rate": 0.001
                }
            },
            "loss": "categorical_crossentropy"
        }
    )
    
    return response.json()


def save_model(name, layers, input_shape=[None, 28, 28, 1]):
    """Save a model using the API"""
    
    response = requests.post(
        f"{API_BASE}/api/save",
        json={
            "name": name,
            "input_shape": input_shape,
            "layers": layers,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "learning_rate": 0.001
                }
            },
            "loss": "categorical_crossentropy"
        }
    )
    
    return response.json()


def load_model(name):
    """Load a saved model using the API"""
    
    response = requests.get(f"{API_BASE}/api/load/{name}")
    return response.json()


def get_templates():
    """Get all available templates"""
    
    response = requests.get(f"{API_BASE}/api/templates")
    return response.json()


def main():
    """Example workflow"""
    
    print("🚀 Neural DSL No-Code Interface - API Example\n")
    
    # 1. Get available templates
    print("1. Fetching templates...")
    templates = get_templates()
    print(f"   Available templates: {', '.join(templates.keys())}\n")
    
    # 2. Create a simple CNN
    print("2. Creating a simple CNN model...")
    layers = create_simple_cnn()
    print(f"   Created model with {len(layers)} layers\n")
    
    # 3. Validate the model
    print("3. Validating model...")
    validation = validate_model(layers)
    
    if validation["valid"]:
        print("   ✓ Model is valid!")
        print("   Shape propagation:")
        for shape_info in validation["shapes"]:
            print(f"     - {shape_info['layer']}: {shape_info['shape']}")
    else:
        print("   ✗ Model has errors:")
        for error in validation["errors"]:
            print(f"     - {error['message']}")
    
    print()
    
    # 4. Generate code
    if validation["valid"]:
        print("4. Generating code...")
        code = generate_code(layers)
        
        print("\n   Neural DSL Code:")
        print("   " + "="*60)
        print("   " + "\n   ".join(code["dsl"].split("\n")[:10]))
        print("   ...")
        print("   " + "="*60)
        
        print("\n   TensorFlow Code available: ✓")
        print("   PyTorch Code available: ✓")
    
    print()
    
    # 5. Save the model
    print("5. Saving model...")
    save_result = save_model("my_simple_cnn", layers)
    if save_result.get("success"):
        print(f"   ✓ Model saved to: {save_result['path']}\n")
    
    # 6. Load the model back
    print("6. Loading model...")
    loaded_model = load_model("my_simple_cnn")
    print(f"   ✓ Model loaded with {len(loaded_model.get('layers', []))} layers\n")
    
    print("✨ Example completed successfully!")


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
        sys.exit(1)
