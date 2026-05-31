from __future__ import annotations

import numpy as np
import torch


def nmse_np(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.linalg.norm(x_hat - x_true) ** 2 / max(np.linalg.norm(x_true) ** 2, 1e-12))


def nmse_torch(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    numerator = (x_hat - x_true).abs().square().sum(dim=1)
    denominator = x_true.abs().square().sum(dim=1).clamp_min(1e-12)
    return numerator / denominator


def detect_activity_np(
    x_hat: np.ndarray,
    mode: str,
    threshold: float,
    ka: int | None = None,
) -> np.ndarray:
    magnitudes = np.abs(x_hat)
    if mode == "threshold":
        return magnitudes > threshold
    if mode == "topk":
        if ka is None:
            raise ValueError("topk detection requires ka")
        pred = np.zeros_like(magnitudes, dtype=bool)
        pred[np.argpartition(magnitudes, -ka)[-ka:]] = True
        return pred
    if mode == "gap":
        order = np.argsort(magnitudes)[::-1]
        sorted_magnitudes = magnitudes[order]
        gaps = sorted_magnitudes[:-1] - sorted_magnitudes[1:]
        if len(gaps) == 0:
            return magnitudes > threshold
        cutoff = int(np.argmax(gaps)) + 1
        pred = np.zeros_like(magnitudes, dtype=bool)
        pred[order[:cutoff]] = True
        return pred
    raise ValueError(f"Unsupported detection mode: {mode}")


def activity_metrics_np(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    threshold: float,
    mode: str = "threshold",
    ka: int | None = None,
) -> dict[str, float]:
    pred = detect_activity_np(x_hat, mode=mode, threshold=threshold, ka=ka)
    true = np.abs(x_true) > 0
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    tn = np.logical_and(~pred, ~true).sum()
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "missed_detection_rate": float(fn / max(tp + fn, 1)),
    }
