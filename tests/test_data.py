import numpy as np
import torch

from cs_massive_access.data import CSConfig, generate_batch, generate_sample, make_signature_matrix


def test_signature_columns_are_unit_norm_for_all_types():
    for signature_type in ["complex_gaussian", "real_gaussian", "rademacher"]:
        config = CSConfig(n=16, k=32, signature_type=signature_type, seed=123)
        s = make_signature_matrix(config)
        column_norms = np.linalg.norm(s, axis=0)
        assert np.allclose(column_norms, 1.0, atol=1e-6)


def test_generate_sample_has_fixed_sparsity():
    config = CSConfig(n=12, k=24, ka=3, e=5.0, signature_type="rademacher", seed=7)
    s = make_signature_matrix(config)
    y, x, active = generate_sample(s, config)
    assert y.shape == (config.n,)
    assert x.shape == (config.k,)
    assert len(active) == config.ka
    assert np.count_nonzero(x) == config.ka


def test_generate_batch_shapes_and_sparsity():
    config = CSConfig(n=12, k=24, ka=3, e=5.0, signature_type="rademacher", seed=7)
    s_np = make_signature_matrix(config)
    s = torch.tensor(s_np, dtype=torch.complex64)
    generator = torch.Generator().manual_seed(8)
    y, x, active = generate_batch(s, config, batch_size=5, generator=generator)
    assert y.shape == (5, config.n)
    assert x.shape == (5, config.k)
    assert active.shape == (5, config.k)
    assert torch.all(active.sum(dim=1) == config.ka)

