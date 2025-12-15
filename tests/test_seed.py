import random

import numpy as np
import pytest

from neural.utils.seed import set_seed


def test_python_and_numpy_determinism():
    set_seed(123)
    r1 = [random.random() for _ in range(3)]
    a1 = np.random.rand(5)

    set_seed(123)
    r2 = [random.random() for _ in range(3)]
    a2 = np.random.rand(5)

    assert r1 == r2
    assert np.allclose(a1, a2)


def test_torch_determinism():
    torch = pytest.importorskip("torch")
    set_seed(123)
    t1 = torch.randn(4)
    set_seed(123)
    t2 = torch.randn(4)
    assert torch.allclose(t1, t2)


def test_tensorflow_determinism():
    tf = pytest.importorskip("tensorflow")
    set_seed(123)
    x1 = tf.random.normal((4,))
    set_seed(123)
    x2 = tf.random.normal((4,))
    # Convert to numpy for robust comparison
    assert np.allclose(x1.numpy(), x2.numpy())

