from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CSConfig:
    n: int = 32
    k: int = 128
    ka: int = 8
    e: float = 10.0
    snr_db: float = 20.0
    signature_type: str = "complex_gaussian"
    symbol_type: str = "qpsk"
    channel_type: str = "flat"
    seed: int = 0


def complex_normal_np(shape: tuple[int, ...], rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return scale * (real + 1j * imag) / np.sqrt(2.0)


def complex_normal_torch(
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
    scale: float = 1.0,
) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
    imag = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
    return scale * torch.complex(real, imag) / np.sqrt(2.0)


def make_signature_matrix(config: CSConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a spreading matrix with unit-norm columns.

    Supported types:
    - complex_gaussian: entries sampled from CN(0, 1)
    - real_gaussian: entries sampled from N(0, 1)
    - rademacher: entries sampled uniformly from {-1, +1}
    """
    rng = np.random.default_rng(config.seed) if rng is None else rng
    if config.signature_type == "complex_gaussian":
        s = complex_normal_np((config.n, config.k), rng)
    elif config.signature_type == "real_gaussian":
        s = rng.standard_normal((config.n, config.k)).astype(np.float32)
    elif config.signature_type == "rademacher":
        s = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(config.n, config.k))
    else:
        raise ValueError(f"Unsupported signature_type: {config.signature_type}")
    norms = np.linalg.norm(s, axis=0, keepdims=True)
    return s / np.maximum(norms, 1e-12)


def _draw_symbols_np(count: int, symbol_type: str, rng: np.random.Generator) -> np.ndarray:
    if symbol_type == "gaussian":
        return complex_normal_np((count,), rng)
    if symbol_type == "qpsk":
        choices = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
        return rng.choice(choices, size=count)
    raise ValueError(f"Unsupported symbol_type: {symbol_type}")


def _draw_symbols_torch(
    batch_size: int,
    ka: int,
    symbol_type: str,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if symbol_type == "gaussian":
        return complex_normal_torch((batch_size, ka), generator, device, dtype)
    if symbol_type == "qpsk":
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        idx = torch.randint(0, 4, (batch_size, ka), generator=generator, device=device)
        real = torch.where(idx < 2, 1.0, -1.0).to(real_dtype)
        imag = torch.where((idx % 2) == 0, 1.0, -1.0).to(real_dtype)
        return torch.complex(real, imag) / np.sqrt(2.0)
    raise ValueError(f"Unsupported symbol_type: {symbol_type}")


def generate_sample(
    s: np.ndarray,
    config: CSConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one `(y, x, active_indices)` sample."""
    rng = np.random.default_rng(config.seed) if rng is None else rng
    active = rng.choice(config.k, size=config.ka, replace=False)
    x = np.zeros(config.k, dtype=np.complex64)

    symbols = _draw_symbols_np(config.ka, config.symbol_type, rng)
    if config.channel_type == "rayleigh":
        channels = complex_normal_np((config.ka,), rng)
    elif config.channel_type == "flat":
        channels = np.ones(config.ka, dtype=np.complex64)
    else:
        raise ValueError(f"Unsupported channel_type: {config.channel_type}")
    x[active] = symbols * channels

    clean = np.sqrt(config.e) * (s @ x)
    noise = complex_normal_np((config.n,), rng, scale=1.0)
    y = clean + noise
    return y.astype(np.complex64), x.astype(np.complex64), active


def generate_batch(
    s: torch.Tensor,
    config: CSConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch directly as PyTorch complex tensors."""
    dtype = s.dtype
    x = torch.zeros((batch_size, config.k), dtype=dtype, device=device)
    active_mask = torch.zeros((batch_size, config.k), dtype=torch.bool, device=device)

    scores = torch.rand((batch_size, config.k), generator=generator, device=device)
    active = torch.topk(scores, k=config.ka, dim=1).indices
    symbols = _draw_symbols_torch(batch_size, config.ka, config.symbol_type, generator, device, dtype)
    if config.channel_type == "rayleigh":
        channels = complex_normal_torch((batch_size, config.ka), generator, device, dtype)
    elif config.channel_type == "flat":
        channels = torch.ones((batch_size, config.ka), dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported channel_type: {config.channel_type}")

    values = symbols * channels
    x.scatter_(1, active, values)
    active_mask.scatter_(1, active, True)

    clean = np.sqrt(config.e) * (x @ s.T)
    noise = complex_normal_torch((batch_size, config.n), generator, device, dtype)
    y = clean + noise
    return y, x, active_mask
