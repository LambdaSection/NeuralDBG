import pytest

# Skip test if docgen module doesn't exist
try:
    from neural.docgen.docgen import generate_markdown
    DOCGEN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DOCGEN_AVAILABLE = False


def _model(input_shape, layers):
    return {"input": {"shape": list(input_shape)}, "layers": layers}


@pytest.mark.skipif(not DOCGEN_AVAILABLE, reason="docgen module not available")
def test_docgen_v11_summary_and_layers():
    md = generate_markdown(_model((4,), [
        {"type": "Dense", "params": {"units": 3}},
        {"type": "Output", "params": {"units": 2, "activation": "softmax"}},
    ]))
    assert "# Model Documentation" in md
    assert "## Summary" in md
    assert "## Layers" in md
    assert "y = softmax(Wx + b)" in md

