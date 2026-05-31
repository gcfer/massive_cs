import numpy as np
import torch

from cs_massive_access.cs import fista, ista, omp
from cs_massive_access.metrics import activity_metrics_np
from cs_massive_access.models import LISTA


def test_omp_recovers_noiseless_sparse_signal():
    rng = np.random.default_rng(42)
    n, k, ka = 16, 32, 3
    s = rng.standard_normal((n, k))
    s = s / np.linalg.norm(s, axis=0, keepdims=True)
    support = np.array([2, 9, 21])
    x = np.zeros(k, dtype=np.complex64)
    x[support] = np.array([1.0, -0.75, 0.5], dtype=np.complex64)
    y = s @ x
    x_hat = omp(y, s, sparsity=ka)
    metrics = activity_metrics_np(x_hat, x, threshold=0.0, mode="topk", ka=ka)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_ista_and_fista_return_expected_shape():
    rng = np.random.default_rng(43)
    n, k = 8, 16
    s = rng.standard_normal((n, k))
    s = s / np.linalg.norm(s, axis=0, keepdims=True)
    y = rng.standard_normal(n)
    assert ista(y, s, iterations=2).shape == (k,)
    assert fista(y, s, iterations=2).shape == (k,)


def test_lista_forward_depth_and_shape():
    s = torch.randn(8, 16, dtype=torch.complex64)
    model = LISTA(s, layers=4)
    y = torch.randn(3, 8, dtype=torch.complex64)
    outputs = model(y, return_layers=True)
    assert len(outputs) == 4
    assert outputs[-1].shape == (3, 16)

