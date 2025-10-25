import sys
sys.path.append('.')

from neural.parser.parser import create_parser, ModelTransformer

# Test parsing to see the layer structure
content = '''
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
'''

parser = create_parser('network')
tree = parser.parse(content)
transformer = ModelTransformer()
model_data = transformer.transform(tree)

print('Model data structure:')
print(f'Input: {model_data["input"]}')
print(f'Layers:')
for i, layer in enumerate(model_data['layers']):
    print(f'  Layer {i}: {layer}')
    print(f'  Type: {type(layer)}')
    if isinstance(layer, dict):
        print(f'  Keys: {layer.keys()}')
