import pytest

from neural.code_generation.code_generator import generate_code
# Skip docgen import if module doesn't exist
try:
    from neural.docgen.docgen import generate_markdown
    DOCGEN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DOCGEN_AVAILABLE = False
    generate_markdown = None


def make_model(input_shape, layers):
    return {
        "input": {"shape": list(input_shape)},
        "layers": layers,
    }


def test_tf_dense_is_applied():
    # Vector input so no flatten needed
    model = make_model((4,), [{"type": "Dense", "params": {"units": 3, "activation": "relu"}}])
    code = generate_code(model, backend="tensorflow")
    # Ensure layer is applied with (x)
    assert "layers.Dense(" in code and ")(x)" in code


def test_output_on_4d_strict_raises():
    # Image-like input; Output should require flatten unless auto flag is set
    model = make_model((28, 28, 1), [{"type": "Output", "params": {"units": 10, "activation": "softmax"}}])
    with pytest.raises(ValueError):
        _ = generate_code(model, backend="tensorflow")


def test_output_on_4d_auto_flatten_allows():
    model = make_model((28, 28, 1), [{"type": "Output", "params": {"units": 10, "activation": "softmax"}}])
    code = generate_code(model, backend="tensorflow", auto_flatten_output=True)
    assert "layers.Flatten()(x)" in code or "Flatten()(x)" in code


def test_docgen_contains_math_and_shapes():
    model = make_model((4,), [
        {"type": "Dense", "params": {"units": 3}},
        {"type": "Output", "params": {"units": 2, "activation": "softmax"}},
    ])
    md = generate_markdown(model)
    assert "Model Documentation" in md
    assert "y = softmax(Wx + b)" in md
    assert "Input shape" in md

