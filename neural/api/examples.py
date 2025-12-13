"""
Example Neural DSL code for testing the API.
"""

SIMPLE_MLP = """
Model SimpleMLP {
    Input(shape=[784])
    Dense(units=128, activation=relu)
    Dropout(rate=0.2)
    Dense(units=64, activation=relu)
    Dense(units=10, activation=softmax)
}
"""

SIMPLE_CNN = """
Model SimpleCNN {
    Input(shape=[28, 28, 1])
    Conv2D(filters=32, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    Conv2D(filters=64, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(units=128, activation=relu)
    Dropout(rate=0.5)
    Dense(units=10, activation=softmax)
}
"""

RESNET_STYLE = """
Model ResNetStyle {
    Input(shape=[32, 32, 3])
    
    Conv2D(filters=64, kernel_size=7, strides=2, activation=relu)
    MaxPooling2D(pool_size=3, strides=2)
    
    Conv2D(filters=64, kernel_size=3, activation=relu)
    Conv2D(filters=64, kernel_size=3, activation=relu)
    
    Conv2D(filters=128, kernel_size=3, strides=2, activation=relu)
    Conv2D(filters=128, kernel_size=3, activation=relu)
    
    GlobalAveragePooling2D()
    Dense(units=10, activation=softmax)
}
"""

VGG_STYLE = """
Model VGGStyle {
    Input(shape=[224, 224, 3])
    
    Conv2D(filters=64, kernel_size=3, activation=relu)
    Conv2D(filters=64, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    
    Conv2D(filters=128, kernel_size=3, activation=relu)
    Conv2D(filters=128, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    
    Conv2D(filters=256, kernel_size=3, activation=relu)
    Conv2D(filters=256, kernel_size=3, activation=relu)
    Conv2D(filters=256, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    
    Flatten()
    Dense(units=4096, activation=relu)
    Dropout(rate=0.5)
    Dense(units=4096, activation=relu)
    Dropout(rate=0.5)
    Dense(units=1000, activation=softmax)
}
"""

RNN_EXAMPLE = """
Model SimpleRNN {
    Input(shape=[100, 128])
    LSTM(units=128, return_sequences=True)
    Dropout(rate=0.3)
    LSTM(units=64)
    Dense(units=32, activation=relu)
    Dense(units=1, activation=sigmoid)
}
"""

EXAMPLES = {
    "simple_mlp": SIMPLE_MLP,
    "simple_cnn": SIMPLE_CNN,
    "resnet_style": RESNET_STYLE,
    "vgg_style": VGG_STYLE,
    "rnn_example": RNN_EXAMPLE,
}


def get_example(name: str) -> str:
    """
    Get example Neural DSL code by name.
    
    Args:
        name: Example name (simple_mlp, simple_cnn, resnet_style, vgg_style, rnn_example)
        
    Returns:
        Neural DSL code
    """
    return EXAMPLES.get(name, SIMPLE_MLP)


def list_examples() -> list:
    """
    List all available examples.
    
    Returns:
        List of example names
    """
    return list(EXAMPLES.keys())
