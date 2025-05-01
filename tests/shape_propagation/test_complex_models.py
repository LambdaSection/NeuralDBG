import sys
import os

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator

#########################################
# 1. VGG-like model architecture
#########################################
def test_vgg_like_model():
    propagator = ShapePropagator()
    input_shape = (1, 224, 224, 3)

    # VGG-like architecture
    layers = [
        # Block 1
        {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}},

        # Block 2
        {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}},

        # Block 3
        {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}},

        # Block 4
        {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}},

        # Block 5
        {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}},

        # Fully connected layers
        {"type": "Flatten", "params": {}},
        {"type": "Dense", "params": {"units": 4096}},
        {"type": "Dense", "params": {"units": 4096}},
        {"type": "Output", "params": {"units": 1000}}
    ]

    shape = input_shape
    for i, layer in enumerate(layers):
        shape = propagator.propagate(shape, layer, framework="tensorflow")

    # Expected shape after all layers
    # After 5 max pooling layers with stride 2: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    # Final dense layer has 1000 units
    expected = (1, 1000)
    assert shape == expected

#########################################
# 2. ResNet-like model with skip connections
#########################################
def test_resnet_like_model():
    propagator = ShapePropagator()
    input_shape = (1, 224, 224, 3)

    # Initial convolution
    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (7, 7), "padding": "same", "stride": 2}}
    shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert shape == (1, 112, 112, 64)

    # Max pooling
    layer = {"type": "MaxPooling2D", "params": {"pool_size": (3, 3), "stride": 2, "padding": "same"}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 56, 56, 64)

    # First residual block - main path
    residual = shape  # Store for skip connection

    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 56, 56, 64)

    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 56, 56, 64)

    # Skip connection - shapes should match for addition
    assert shape == residual

    # Second residual block - with dimension change
    residual = shape  # Store for skip connection

    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 2}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 28, 28, 128)

    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 28, 28, 128)

    # Skip connection with projection
    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (1, 1), "padding": "same", "stride": 2}}
    residual = propagator.propagate(residual, layer, framework="tensorflow")
    assert residual == (1, 28, 28, 128)

    # Skip connection - shapes should match for addition
    assert shape == residual

    # Global average pooling
    layer = {"type": "GlobalAveragePooling2D", "params": {}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 128)

    # Final classification layer
    layer = {"type": "Output", "params": {"units": 1000}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 1000)

#########################################
# 3. U-Net-like model with encoder-decoder architecture
#########################################
def test_unet_like_model():
    propagator = ShapePropagator()
    input_shape = (1, 256, 256, 3)

    # Encoder path
    # Block 1
    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert shape == (1, 256, 256, 64)

    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    skip1 = shape  # Store for skip connection
    assert skip1 == (1, 256, 256, 64)

    layer = {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 128, 128, 64)

    # Block 2
    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 128, 128, 128)

    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    skip2 = shape  # Store for skip connection
    assert skip2 == (1, 128, 128, 128)

    layer = {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 64, 64, 128)

    # Block 3
    layer = {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 64, 64, 256)

    layer = {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    skip3 = shape  # Store for skip connection
    assert skip3 == (1, 64, 64, 256)

    layer = {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 32, 32, 256)

    # Bottom
    layer = {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 32, 32, 512)

    layer = {"type": "Conv2D", "params": {"filters": 512, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 32, 32, 512)

    # Decoder path
    # Block 3
    layer = {"type": "UpSampling2D", "params": {"size": (2, 2)}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 64, 64, 512)

    # Concatenate with skip3
    # In a real model, we would concatenate with skip3 here
    # For testing, we'll just verify the shapes are compatible for concatenation
    assert shape[1:3] == skip3[1:3]  # Spatial dimensions should match

    layer = {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 64, 64, 256)

    layer = {"type": "Conv2D", "params": {"filters": 256, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 64, 64, 256)

    # Block 2
    layer = {"type": "UpSampling2D", "params": {"size": (2, 2)}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 128, 128, 256)

    # Concatenate with skip2
    assert shape[1:3] == skip2[1:3]  # Spatial dimensions should match

    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 128, 128, 128)

    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 128, 128, 128)

    # Block 1
    layer = {"type": "UpSampling2D", "params": {"size": (2, 2)}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 256, 256, 128)

    # Concatenate with skip1
    assert shape[1:3] == skip1[1:3]  # Spatial dimensions should match

    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 256, 256, 64)

    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 256, 256, 64)

    # Output layer
    layer = {"type": "Conv2D", "params": {"filters": 2, "kernel_size": (1, 1), "padding": "same", "stride": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 256, 256, 2)

#########################################
# 4. Multi-input model
#########################################
def test_multi_input_model():
    propagator = ShapePropagator()

    # Image input branch
    image_input_shape = (1, 224, 224, 3)

    layer = {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "padding": "same", "stride": 2}}
    image_shape = propagator.propagate(image_input_shape, layer, framework="tensorflow")
    assert image_shape == (1, 112, 112, 64)

    layer = {"type": "MaxPooling2D", "params": {"pool_size": (2, 2), "stride": 2}}
    image_shape = propagator.propagate(image_shape, layer, framework="tensorflow")
    assert image_shape == (1, 56, 56, 64)

    layer = {"type": "Conv2D", "params": {"filters": 128, "kernel_size": (3, 3), "padding": "same", "stride": 2}}
    image_shape = propagator.propagate(image_shape, layer, framework="tensorflow")
    assert image_shape == (1, 28, 28, 128)

    layer = {"type": "GlobalAveragePooling2D", "params": {}}
    image_shape = propagator.propagate(image_shape, layer, framework="tensorflow")
    assert image_shape == (1, 128)

    # Metadata input branch
    metadata_input_shape = (1, 10)

    layer = {"type": "Dense", "params": {"units": 32}}
    metadata_shape = propagator.propagate(metadata_input_shape, layer, framework="tensorflow")
    assert metadata_shape == (1, 32)

    layer = {"type": "Dense", "params": {"units": 64}}
    metadata_shape = propagator.propagate(metadata_shape, layer, framework="tensorflow")
    assert metadata_shape == (1, 64)

    # Concatenate the branches
    # In a real model, we would concatenate the branches here
    # For testing, we'll just verify the shapes are compatible for concatenation
    assert image_shape[0] == metadata_shape[0]  # Batch dimensions should match

    # Combined shape would be (1, 128 + 64) = (1, 192)
    combined_shape = (image_shape[0], image_shape[1] + metadata_shape[1])
    assert combined_shape == (1, 192)

    # Final layers
    layer = {"type": "Dense", "params": {"units": 64}}
    shape = propagator.propagate(combined_shape, layer, framework="tensorflow")
    assert shape == (1, 64)

    layer = {"type": "Output", "params": {"units": 1}}
    shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == (1, 1)

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
