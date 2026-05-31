from __future__ import annotations

from argparse import Namespace

from .evaluate import evaluate
from .train import train


def main() -> None:
    checkpoint = train(
        Namespace(
            n=24,
            k=64,
            ka=4,
            E=10.0,
            snr_db=20.0,
            signature_type="rademacher",
            symbol_type="qpsk",
            channel_type="flat",
            L=6,
            train_samples=1024,
            val_samples=256,
            batch_size=64,
            epochs=5,
            lr=1e-3,
            seed=7,
            device="cpu",
            vector_thresholds=False,
            output="runs/demo_lista.pt",
        )
    )
    evaluate(
        Namespace(
            checkpoint=checkpoint,
            n=24,
            k=64,
            ka=4,
            E=10.0,
            snr_db=20.0,
            signature_type="rademacher",
            symbol_type="qpsk",
            channel_type="flat",
            samples=40,
            iterations=80,
            lam=0.05,
            threshold=0.05,
            seed=99,
            device="cpu",
        )
    )


if __name__ == "__main__":
    main()
