from __future__ import annotations

import numpy as np


def soft_threshold_complex(z: np.ndarray, lam: float | np.ndarray) -> np.ndarray:
    mag = np.abs(z)
    scale = np.maximum(mag - lam, 0.0) / np.maximum(mag, 1e-12)
    return scale * z


def spectral_step_size(s: np.ndarray) -> float:
    lipschitz = np.linalg.norm(s, ord=2) ** 2
    return 1.0 / max(lipschitz, 1e-12)


def ista(
    y: np.ndarray,
    s: np.ndarray,
    lam: float = 0.05,
    iterations: int = 100,
    step_size: float | None = None,
    tol: float | None = None,
) -> np.ndarray:
    step = spectral_step_size(s) if step_size is None else step_size
    x = np.zeros(s.shape[1], dtype=np.complex64)
    sh = s.conj().T
    for _ in range(iterations):
        x_prev = x
        gradient = sh @ (s @ x - y)
        x = soft_threshold_complex(x - step * gradient, step * lam)
        if tol is not None:
            relative_change = np.linalg.norm(x - x_prev) / max(np.linalg.norm(x_prev), 1e-12)
            if relative_change < tol:
                break
    return x


def fista(
    y: np.ndarray,
    s: np.ndarray,
    lam: float = 0.05,
    iterations: int = 100,
    step_size: float | None = None,
    tol: float | None = None,
) -> np.ndarray:
    step = spectral_step_size(s) if step_size is None else step_size
    x = np.zeros(s.shape[1], dtype=np.complex64)
    z = x.copy()
    t = 1.0
    sh = s.conj().T
    for _ in range(iterations):
        gradient = sh @ (s @ z - y)
        x_next = soft_threshold_complex(z - step * gradient, step * lam)
        if tol is not None:
            relative_change = np.linalg.norm(x_next - x) / max(np.linalg.norm(x), 1e-12)
            if relative_change < tol:
                x = x_next
                break
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_next + ((t - 1.0) / t_next) * (x_next - x)
        x, t = x_next, t_next
    return x


def omp(y: np.ndarray, s: np.ndarray, sparsity: int) -> np.ndarray:
    residual = y.copy()
    support: list[int] = []
    x = np.zeros(s.shape[1], dtype=np.complex64)
    sh = s.conj().T

    for _ in range(sparsity):
        correlations = np.abs(sh @ residual)
        correlations[support] = -np.inf
        idx = int(np.argmax(correlations))
        support.append(idx)
        sub_s = s[:, support]
        coeffs, *_ = np.linalg.lstsq(sub_s, y, rcond=None)
        residual = y - sub_s @ coeffs

    x[support] = coeffs
    return x
