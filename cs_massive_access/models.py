from __future__ import annotations

import torch
from torch import nn


def complex_soft_threshold(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    mag = z.abs()
    scale = (mag - lam).clamp_min(0.0) / mag.clamp_min(1e-12)
    return scale * z


class LISTA(nn.Module):
    """Learned ISTA with complex-valued fixed sensing matrix and real parameters."""

    def __init__(
        self,
        s: torch.Tensor,
        layers: int = 8,
        vector_thresholds: bool = False,
        initial_lambda: float = 0.05,
    ) -> None:
        super().__init__()
        if not torch.is_complex(s):
            raise TypeError("LISTA expects a complex-valued signature matrix")

        self.layers = layers
        self.vector_thresholds = vector_thresholds
        self.register_buffer("s", s)
        self.register_buffer("sh", s.conj().T)

        lipschitz = torch.linalg.matrix_norm(s, ord=2).square().real.clamp_min(1e-12)
        initial_delta = torch.log(torch.expm1(1.0 / lipschitz))
        self.raw_delta = nn.Parameter(initial_delta.repeat(layers))

        threshold_shape = (layers, s.shape[1]) if vector_thresholds else (layers,)
        raw_lambda = torch.full(threshold_shape, initial_lambda, dtype=torch.float32)
        self.raw_lambda = nn.Parameter(torch.log(torch.expm1(raw_lambda)))

    @property
    def deltas(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_delta)

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_lambda)

    def forward(self, y: torch.Tensor, return_layers: bool = False) -> torch.Tensor | list[torch.Tensor]:
        x = torch.zeros((y.shape[0], self.s.shape[1]), dtype=y.dtype, device=y.device)
        outputs: list[torch.Tensor] = []

        for layer in range(self.layers):
            residual = x @ self.s.T - y
            gradient = residual @ self.sh.T
            lam = self.lambdas[layer]
            x = complex_soft_threshold(x - self.deltas[layer] * gradient, lam)
            outputs.append(x)

        return outputs if return_layers else x

