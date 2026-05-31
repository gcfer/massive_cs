from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import CSConfig, generate_batch, make_signature_matrix
from .metrics import nmse_torch
from .models import LISTA


def deep_supervision_loss(outputs: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.2, 1.0, steps=len(outputs), device=target.device)
    losses = torch.stack([(out - target).abs().square().mean() for out in outputs])
    return torch.sum(weights * losses) / torch.sum(weights)


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
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
    rng = np.random.default_rng(args.seed)
    s_np = make_signature_matrix(config, rng)
    s = torch.tensor(s_np, dtype=torch.complex64, device=device)
    sensing = np.sqrt(config.e) * s

    model = LISTA(sensing, layers=args.L, vector_thresholds=args.vector_thresholds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    steps_per_epoch = max(args.train_samples // args.batch_size, 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            y, x, _ = generate_batch(s, config, args.batch_size, generator, device)
            outputs = model(y, return_layers=True)
            loss = deep_supervision_loss(outputs, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            y_val, x_val, _ = generate_batch(s, config, args.val_samples, generator, device)
            pred = model(y_val)
            val_nmse = nmse_torch(pred, x_val).mean().item()
        print(f"epoch={epoch:03d} loss={epoch_loss / steps_per_epoch:.6f} val_nmse={val_nmse:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "layers": args.L,
            "vector_thresholds": args.vector_thresholds,
            "signature_matrix": s_np,
        },
        output,
    )
    print(f"saved checkpoint to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LISTA detector for massive access.")
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
    parser.add_argument("--L", "--layers", dest="L", type=int, default=8, help="LISTA depth / number of layers.")
    parser.add_argument(
        "--num-samples",
        "--train-samples",
        dest="train_samples",
        type=int,
        default=4096,
        help="Number of synthetic training samples generated per epoch.",
    )
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vector-thresholds", action="store_true")
    parser.add_argument("--output", default="runs/lista.pt")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
