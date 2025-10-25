import sys
import os
import tempfile
import traceback
sys.path.append('.')

from click.testing import CliRunner
from neural.cli import cli

# Create a sample neural file
with tempfile.NamedTemporaryFile(suffix='.neural', delete=False) as f:
    f.write(b'''
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
''')
    temp_path = f.name

try:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['visualize', temp_path, '--format', 'png', '--no-cache'])
        print(f'Exit code: {result.exit_code}')
        print(f'Output: {result.output}')
        if result.exception:
            print(f'Exception: {result.exception}')
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
