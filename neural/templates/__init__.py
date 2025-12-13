"""
Model templates for rapid prototyping.

This module provides pre-built, customizable templates for common architectures.
Templates are designed for education and rapid experimentation.
"""

from typing import Dict, List


def get_template(name: str, **kwargs) -> str:
    """Get a model template by name."""
    templates = {
        'mnist_cnn': _mnist_cnn_template,
        'image_classifier': _image_classifier_template,
        'text_lstm': _text_lstm_template,
    }

    if name not in templates:
        available = ', '.join(templates.keys())
        raise ValueError(f"Template '{name}' not found. Available: {available}")

    return templates[name](**kwargs)


def list_templates() -> List[Dict[str, str]]:
    """List all available templates."""
    return [
        {
            'name': 'mnist_cnn',
            'description': 'Simple CNN for MNIST digit classification',
            'difficulty': 'beginner',
            'use_case': 'Image classification (small images)',
        },
        {
            'name': 'image_classifier',
            'description': 'General-purpose image classifier',
            'difficulty': 'beginner',
            'use_case': 'Image classification (any size)',
        },
        {
            'name': 'text_lstm',
            'description': 'LSTM-based text classification',
            'difficulty': 'beginner',
            'use_case': 'Sentiment analysis, text categorization',
        },
    ]


def _mnist_cnn_template(
    num_classes: int = 10,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.5,
    **kwargs
) -> str:
    return f'''network MNISTClassifier {{
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3,3), "relu")
        MaxPooling2D((2,2))
        Conv2D(64, (3,3), "relu")
        MaxPooling2D((2,2))
        Flatten()
        Dense(128, "relu")
        Dropout({dropout_rate})
        Output({num_classes}, "softmax")
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate={learning_rate})
}}
'''


def _image_classifier_template(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 1000,
    learning_rate: float = 0.001,
    **kwargs
) -> str:
    h, w, c = input_shape
    return f'''network ImageClassifier {{
    input: ({h}, {w}, {c})
    layers:
        Conv2D(32, (3,3), "relu")
        MaxPooling2D((2,2))
        Conv2D(64, (3,3), "relu")
        MaxPooling2D((2,2))
        Flatten()
        Dense(512, "relu")
        Output({num_classes}, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate={learning_rate})
}}
'''


def _text_lstm_template(
    vocab_size: int = 10000,
    max_length: int = 500,
    num_classes: int = 3,
    **kwargs
) -> str:
    return f'''network TextLSTM {{
    input: ({max_length},)
    layers:
        Embedding({vocab_size}, 128)
        LSTM(64)
        Dense(64, "relu")
        Output({num_classes}, "softmax")
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}}
'''
