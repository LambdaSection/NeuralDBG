"""
Enhanced No-Code Interface with React Flow-based drag-and-drop visual designer
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from neural.code_generation.code_generator import generate_code
from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.no_code.config import get_config


config = get_config()
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, origins=config.CORS_ORIGINS)


LAYER_CATEGORIES = {
    "Convolutional": [
        {"name": "Conv1D", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}},
        {"name": "Conv2D", "params": {"filters": 32, "kernel_size": [3, 3], "activation": "relu"}},
        {"name": "Conv3D", "params": {"filters": 32, "kernel_size": [3, 3, 3], "activation": "relu"}},
        {"name": "SeparableConv2D", "params": {"filters": 32, "kernel_size": [3, 3], "activation": "relu"}},
        {"name": "DepthwiseConv2D", "params": {"kernel_size": [3, 3], "depth_multiplier": 1}},
        {"name": "TransposedConv2D", "params": {"filters": 32, "kernel_size": [3, 3], "strides": [2, 2]}},
    ],
    "Pooling": [
        {"name": "MaxPooling1D", "params": {"pool_size": 2}},
        {"name": "MaxPooling2D", "params": {"pool_size": [2, 2]}},
        {"name": "AveragePooling1D", "params": {"pool_size": 2}},
        {"name": "AveragePooling2D", "params": {"pool_size": [2, 2]}},
        {"name": "GlobalMaxPooling1D", "params": {}},
        {"name": "GlobalMaxPooling2D", "params": {}},
        {"name": "GlobalAveragePooling1D", "params": {}},
        {"name": "GlobalAveragePooling2D", "params": {}},
    ],
    "Core": [
        {"name": "Dense", "params": {"units": 128, "activation": "relu"}},
        {"name": "Flatten", "params": {}},
        {"name": "Reshape", "params": {"target_shape": [1, 1, -1]}},
        {"name": "Permute", "params": {"dims": [2, 1]}},
        {"name": "RepeatVector", "params": {"n": 3}},
    ],
    "Normalization": [
        {"name": "BatchNormalization", "params": {}},
        {"name": "LayerNormalization", "params": {}},
        {"name": "GroupNormalization", "params": {"groups": 32}},
    ],
    "Regularization": [
        {"name": "Dropout", "params": {"rate": 0.5}},
        {"name": "SpatialDropout1D", "params": {"rate": 0.5}},
        {"name": "SpatialDropout2D", "params": {"rate": 0.5}},
        {"name": "GaussianNoise", "params": {"stddev": 0.1}},
        {"name": "GaussianDropout", "params": {"rate": 0.5}},
    ],
    "Recurrent": [
        {"name": "LSTM", "params": {"units": 128, "return_sequences": False}},
        {"name": "GRU", "params": {"units": 128, "return_sequences": False}},
        {"name": "SimpleRNN", "params": {"units": 128, "return_sequences": False}},
        {"name": "Bidirectional", "params": {"layer": "LSTM", "units": 128}},
        {"name": "ConvLSTM2D", "params": {"filters": 32, "kernel_size": [3, 3]}},
    ],
    "Attention": [
        {"name": "MultiHeadAttention", "params": {"num_heads": 8, "key_dim": 64}},
        {"name": "Attention", "params": {}},
    ],
    "Embedding": [
        {"name": "Embedding", "params": {"input_dim": 1000, "output_dim": 64}},
    ],
    "Activation": [
        {"name": "ReLU", "params": {}},
        {"name": "LeakyReLU", "params": {"alpha": 0.3}},
        {"name": "PReLU", "params": {}},
        {"name": "ELU", "params": {"alpha": 1.0}},
        {"name": "Softmax", "params": {}},
        {"name": "Sigmoid", "params": {}},
        {"name": "Tanh", "params": {}},
    ],
}

MODEL_TEMPLATES = {
    "mnist_cnn": {
        "name": "MNIST CNN",
        "description": "Simple CNN for MNIST digit classification",
        "input_shape": [None, 28, 28, 1],
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": [3, 3], "activation": "relu"}},
            {"type": "MaxPooling2D", "params": {"pool_size": [2, 2]}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": [3, 3], "activation": "relu"}},
            {"type": "MaxPooling2D", "params": {"pool_size": [2, 2]}},
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 128, "activation": "relu"}},
            {"type": "Dropout", "params": {"rate": 0.5}},
            {"type": "Dense", "params": {"units": 10, "activation": "softmax"}},
        ]
    },
    "cifar10_vgg": {
        "name": "CIFAR-10 VGG",
        "description": "VGG-style network for CIFAR-10",
        "input_shape": [None, 32, 32, 3],
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": [3, 3], "activation": "relu", "padding": "same"}},
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": [3, 3], "activation": "relu", "padding": "same"}},
            {"type": "MaxPooling2D", "params": {"pool_size": [2, 2]}},
            {"type": "Dropout", "params": {"rate": 0.25}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": [3, 3], "activation": "relu", "padding": "same"}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": [3, 3], "activation": "relu", "padding": "same"}},
            {"type": "MaxPooling2D", "params": {"pool_size": [2, 2]}},
            {"type": "Dropout", "params": {"rate": 0.25}},
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 512, "activation": "relu"}},
            {"type": "Dropout", "params": {"rate": 0.5}},
            {"type": "Dense", "params": {"units": 10, "activation": "softmax"}},
        ]
    },
    "text_lstm": {
        "name": "Text LSTM",
        "description": "LSTM for text classification",
        "input_shape": [None, 100],
        "layers": [
            {"type": "Embedding", "params": {"input_dim": 10000, "output_dim": 128}},
            {"type": "LSTM", "params": {"units": 64, "return_sequences": True}},
            {"type": "LSTM", "params": {"units": 64}},
            {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
            {"type": "Dropout", "params": {"rate": 0.5}},
            {"type": "Dense", "params": {"units": 1, "activation": "sigmoid"}},
        ]
    },
    "transformer": {
        "name": "Transformer Encoder",
        "description": "Transformer encoder block",
        "input_shape": [None, 512],
        "layers": [
            {"type": "Embedding", "params": {"input_dim": 10000, "output_dim": 256}},
            {"type": "MultiHeadAttention", "params": {"num_heads": 8, "key_dim": 32}},
            {"type": "LayerNormalization", "params": {}},
            {"type": "Dense", "params": {"units": 512, "activation": "relu"}},
            {"type": "Dense", "params": {"units": 256}},
            {"type": "LayerNormalization", "params": {}},
            {"type": "GlobalAveragePooling1D", "params": {}},
            {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
            {"type": "Dense", "params": {"units": 2, "activation": "softmax"}},
        ]
    },
    "resnet_block": {
        "name": "ResNet Block",
        "description": "Residual block for image classification",
        "input_shape": [None, 224, 224, 3],
        "layers": [
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": [7, 7], "strides": [2, 2], "padding": "same"}},
            {"type": "BatchNormalization", "params": {}},
            {"type": "ReLU", "params": {}},
            {"type": "MaxPooling2D", "params": {"pool_size": [3, 3], "strides": [2, 2], "padding": "same"}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": [3, 3], "padding": "same"}},
            {"type": "BatchNormalization", "params": {}},
            {"type": "ReLU", "params": {}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": [3, 3], "padding": "same"}},
            {"type": "BatchNormalization", "params": {}},
            {"type": "GlobalAveragePooling2D", "params": {}},
            {"type": "Dense", "params": {"units": 1000, "activation": "softmax"}},
        ]
    },
}

TUTORIAL_STEPS = [
    {
        "id": "welcome",
        "title": "Welcome to Neural DSL No-Code Designer",
        "content": "Build neural networks visually with drag-and-drop. No coding required!",
        "target": None,
    },
    {
        "id": "layer_palette",
        "title": "Layer Palette",
        "content": "Browse and search for layers by category. Drag layers onto the canvas to add them.",
        "target": "layer-palette",
    },
    {
        "id": "canvas",
        "title": "Visual Canvas",
        "content": "This is your network canvas. Connect layers by dragging from output to input ports.",
        "target": "flow-canvas",
    },
    {
        "id": "properties",
        "title": "Layer Properties",
        "content": "Click a layer to edit its parameters. Real-time validation helps you avoid errors.",
        "target": "properties-panel",
    },
    {
        "id": "templates",
        "title": "Model Templates",
        "content": "Start quickly with pre-built templates for common architectures.",
        "target": "templates-button",
    },
    {
        "id": "validation",
        "title": "Shape Validation",
        "content": "We automatically validate tensor shapes as you build. Errors are highlighted in real-time.",
        "target": "validation-panel",
    },
    {
        "id": "export",
        "title": "Export Your Model",
        "content": "Generate Neural DSL code, TensorFlow, or PyTorch code with one click.",
        "target": "export-button",
    },
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/layers')
def get_layers():
    return jsonify(LAYER_CATEGORIES)


@app.route('/api/templates')
def get_templates():
    return jsonify(MODEL_TEMPLATES)


@app.route('/api/tutorial')
def get_tutorial():
    return jsonify(TUTORIAL_STEPS)


@app.route('/api/validate', methods=['POST'])
def validate_model():
    data = request.json
    input_shape = data.get('input_shape', [None, 28, 28, 1])
    layers = data.get('layers', [])
    
    errors = []
    warnings = []
    
    if not layers:
        warnings.append({
            "type": "warning",
            "message": "Model has no layers",
            "layer_index": None
        })
        return jsonify({"valid": True, "errors": errors, "warnings": warnings, "shapes": []})
    
    propagator = ShapePropagator()
    current_shape = tuple(input_shape)
    shapes = [{"layer": "Input", "shape": list(current_shape)}]
    
    for i, layer in enumerate(layers):
        try:
            layer_type = layer.get('type')
            layer_params = layer.get('params', {})
            
            next_shape = propagator.propagate(current_shape, {"type": layer_type, "params": layer_params}, "tensorflow")
            
            if next_shape is None:
                errors.append({
                    "type": "error",
                    "message": f"Cannot propagate shape through {layer_type}",
                    "layer_index": i,
                    "layer_id": layer.get('id')
                })
                break
            
            shapes.append({"layer": layer_type, "shape": list(next_shape)})
            current_shape = next_shape
            
        except Exception as e:
            errors.append({
                "type": "error",
                "message": f"Shape validation error in {layer.get('type')}: {str(e)}",
                "layer_index": i,
                "layer_id": layer.get('id')
            })
            break
    
    valid = len(errors) == 0
    
    return jsonify({
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "shapes": shapes
    })


@app.route('/api/generate-code', methods=['POST'])
def generate_model_code():
    data = request.json
    input_shape = data.get('input_shape', [None, 28, 28, 1])
    layers = data.get('layers', [])
    optimizer = data.get('optimizer', {"type": "Adam", "params": {"learning_rate": 0.001}})
    loss = data.get('loss', "categorical_crossentropy")
    
    model_data = {
        "type": "model",
        "name": "MyModel",
        "input": {"type": "Input", "shape": input_shape},
        "layers": layers,
        "loss": {"value": f'"{loss}"'},
        "optimizer": optimizer
    }
    
    dsl_code = generate_dsl_code(model_data)
    
    try:
        tf_code = generate_code(model_data, "tensorflow")
    except Exception as e:
        tf_code = f"# Error generating TensorFlow code: {str(e)}"
    
    try:
        pytorch_code = generate_code(model_data, "pytorch")
    except Exception as e:
        pytorch_code = f"# Error generating PyTorch code: {str(e)}"
    
    return jsonify({
        "dsl": dsl_code,
        "tensorflow": tf_code,
        "pytorch": pytorch_code
    })


@app.route('/api/save', methods=['POST'])
def save_model():
    data = request.json
    name = data.get('name', f'model_{int(time.time())}')
    
    save_path = os.path.join(config.SAVED_MODELS_DIR, f'{name}.json')
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"success": True, "path": save_path})


@app.route('/api/load/<name>')
def load_model(name):
    load_path = os.path.join(config.SAVED_MODELS_DIR, f'{name}.json')
    
    if not os.path.exists(load_path):
        return jsonify({"error": "Model not found"}), 404
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(data)


@app.route('/api/models')
def list_models():
    models = []
    
    if os.path.exists(config.SAVED_MODELS_DIR):
        for filename in os.listdir(config.SAVED_MODELS_DIR):
            if filename.endswith('.json'):
                models.append(filename.replace('.json', ''))
    
    return jsonify(models)


def generate_dsl_code(model_data):
    input_shape = model_data.get("input", {}).get("shape", [None, 28, 28, 1])
    layers = model_data.get("layers", [])
    loss = model_data.get("loss", {}).get("value", '"categorical_crossentropy"')
    optimizer = model_data.get("optimizer", {})
    
    optimizer_type = optimizer.get("type", "Adam")
    optimizer_params = optimizer.get("params", {})
    
    optimizer_params_str = ", ".join([f"{k}={v}" for k, v in optimizer_params.items()])
    optimizer_str = f"{optimizer_type}({optimizer_params_str})" if optimizer_params_str else optimizer_type
    
    layers_str = ""
    for layer in layers:
        layer_type = layer.get("type", "")
        layer_params = layer.get("params", {})
        params_str = ", ".join([f"{k}={format_param_value(v)}" for k, v in layer_params.items()])
        layers_str += f"        {layer_type}({params_str})\n"
    
    shape_str = format_shape(input_shape)
    
    dsl_code = f"""network MyModel {{
    input: {shape_str}
    layers:
{layers_str}
    loss: {loss}
    optimizer: {optimizer_str}
}}
"""
    return dsl_code


def format_param_value(value):
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return f"({', '.join(map(str, value))})"
    return str(value)


def format_shape(shape):
    if isinstance(shape, (list, tuple)):
        return f"({', '.join('None' if s is None else str(s) for s in shape)})"
    return str(shape)


if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
