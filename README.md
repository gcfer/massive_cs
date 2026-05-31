# Compressed Sensing for Massive Random Access

Python experiments for massive machine-type access using classical compressed
sensing and a deep unfolded LISTA network.

The implemented signal model is

```text
y = sqrt(E) S x + n,    n ~ CN(0, I_N)
```

where:

- `K` is the number of potential users
- `Ka` is the number of active users in a slot
- `N` is the signature length / measurement dimension
- `S` is an `N x K` spreading matrix with unit-norm columns
- `E` is the per-user signal energy
- `x_k = gamma_k h_k b_k`, with flat channel `h_k = 1` by default

The repository includes:

- ISTA, FISTA, and OMP classical compressed sensing baselines
- LISTA deep unfolding with learnable per-layer step sizes and thresholds
- synthetic complex-valued data generation
- activity detection by fixed threshold, top-`Ka`, or adaptive magnitude gap

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you do not need tests, this is enough:

```bash
pip install -e .
```

## Quick Start

Run a small end-to-end experiment:

```bash
csma-demo
```

or equivalently:

```bash
python -m cs_massive_access.demo
```

## Train LISTA

```bash
csma-train \
  --N 32 \
  --K 128 \
  --Ka 8 \
  --E 10 \
  --L 10 \
  --signature-type rademacher \
  --channel-type flat \
  --num-samples 20000 \
  --epochs 20
```

Training samples are generated synthetically on the fly. There is no stored
training database unless you choose to add one.

## Evaluate

Evaluate classical baselines plus a trained LISTA checkpoint:

```bash
csma-evaluate \
  --checkpoint runs/lista.pt \
  --E 10 \
  --iterations 1000 \
  --detect topk
```

If `--checkpoint` is omitted, only ISTA, FISTA, and OMP are evaluated.

Activity-detection modes:

- `--detect threshold`: declare active when `|x_hat_k| > threshold`
- `--detect topk`: declare the largest `Ka` entries active
- `--detect gap`: sort magnitudes and choose the largest drop as the cutoff

For synthetic experiments where `Ka` is known, `--detect topk` is usually the
cleanest activity-detection metric.

## Important Parameters

```text
--N                 measurement dimension / signature length
--K                 number of potential users
--Ka                number of active users per sample
--E                 per-user signal energy in y = sqrt(E) Sx + n
--L                 LISTA depth / number of unfolded layers
--num-samples       synthetic training samples per epoch
--signature-type    complex_gaussian, real_gaussian, or rademacher
--channel-type      flat or rayleigh
--iterations        ISTA/FISTA iteration budget during evaluation
--tol               optional ISTA/FISTA early-stopping tolerance
```

## Development

```bash
pytest
```

or:

```bash
make test
```

The generated folders `.venv/`, `runs/`, and Python caches are ignored by git.

