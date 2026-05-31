from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .cs import fista, ista, omp
from .data import CSConfig, generate_sample, make_signature_matrix
from .metrics import activity_metrics_np, nmse_np
from .models import LISTA


def summarize(results: dict[str, list[dict[str, float]]]) -> None:
    for name, rows in results.items():
        keys = rows[0].keys()
        means = {key: np.mean([row[key] for row in rows]) for key in keys}
        print(
            f"{name:>8} "
            f"nmse={means['nmse']:.4f} "
            f"precision={means['precision']:.3f} "
            f"recall={means['recall']:.3f} "
            f"far={means['false_alarm_rate']:.3f} "
            f"mdr={means['missed_detection_rate']:.3f}"
        )


def evaluate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    config = CSConfig(
        n=args.n,
        k=args.k,
        ka=args.ka,
        e=args.E,
        snr_db=args.snr_db,
        signature_type=args.signature_type,
        symbol_type=args.symbol_type,
        channel_type=args.channel_type,
        seed=args.seed,
    )
    s_np = make_signature_matrix(config, rng)

    model = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        ckpt_config = CSConfig(**checkpoint["config"])
        config = CSConfig(
            n=ckpt_config.n,
            k=ckpt_config.k,
            ka=ckpt_config.ka,
            e=args.E,
            snr_db=args.snr_db,
            seed=args.seed,
            signature_type=ckpt_config.signature_type,
            symbol_type=ckpt_config.symbol_type,
            channel_type=ckpt_config.channel_type,
        )
        s_np = checkpoint["signature_matrix"]
        s_torch = torch.tensor(s_np, dtype=torch.complex64, device=args.device)
        model = LISTA(np.sqrt(config.e) * s_torch, layers=checkpoint["layers"], vector_thresholds=checkpoint["vector_thresholds"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

    results: dict[str, list[dict[str, float]]] = {"ista": [], "fista": [], "omp": []}
    if model is not None:
        results["lista"] = []

    for _ in range(args.samples):
        y, x, _ = generate_sample(s_np, config, rng)
        sensing_np = np.sqrt(config.e) * s_np
        estimates = {
            "ista": ista(y, sensing_np, lam=args.lam, iterations=args.iterations, tol=args.tol),
            "fista": fista(y, sensing_np, lam=args.lam, iterations=args.iterations, tol=args.tol),
            "omp": omp(y, sensing_np, sparsity=config.ka),
        }
        if model is not None:
            with torch.no_grad():
                y_t = torch.tensor(y[None, :], dtype=torch.complex64, device=args.device)
                estimates["lista"] = model(y_t).cpu().numpy()[0]

        for name, x_hat in estimates.items():
            row = {"nmse": nmse_np(x_hat, x)}
            row.update(activity_metrics_np(x_hat, x, threshold=args.threshold, mode=args.detect, ka=config.ka))
            results[name].append(row)

    summarize(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CS baselines and optional LISTA checkpoint.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--N", "--n", dest="n", type=int, default=32, help="Measurement dimension / signature length.")
    parser.add_argument("--K", "--k", dest="k", type=int, default=128, help="Number of potential users.")
    parser.add_argument("--Ka", "--ka", dest="ka", type=int, default=8, help="Number of active users per sample.")
    parser.add_argument("--E", type=float, default=10.0, help="Per-user signal energy in y = sqrt(E) Sx + n.")
    parser.add_argument("--snr-db", type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--signature-type",
        choices=["complex_gaussian", "real_gaussian", "rademacher"],
        default="complex_gaussian",
    )
    parser.add_argument("--symbol-type", choices=["qpsk", "gaussian"], default="qpsk")
    parser.add_argument("--channel-type", choices=["flat", "rayleigh"], default="flat")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tol", type=float, help="Optional relative-change stopping tolerance for ISTA/FISTA.")
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--detect",
        choices=["threshold", "topk", "gap"],
        default="threshold",
        help="Activity detector: fixed threshold, top-Ka magnitudes, or largest sorted-magnitude gap.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
