
## Requirements

```
python >= 3.10
torch >= 2.0
numpy, scipy, sympy, matplotlib
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on CPU, CUDA, or Apple MPS; select with `--device {cpu,cuda,mps}`.

---

## Repository layout

| File | Contents |
|------|----------|
| `helpers.py` | Geometry backbone: normal estimation, local-polynomial and Taubin curvature (`K`, `H`), simplicial-complex construction (Algorithm 1), oriented boundary operators `B1`, `B2`, Hodge Laplacians `L0`, `L1`, simplex features, Betti numbers, Hodge decomposition utilities. |
| `manifolds.py` | Point-cloud samplers: torus (analytic `K`), distorted torus, Stanford Bunny. |
| `pdes.py` | Fisher–KPP, Gray–Scott, Turing (Gierer–Meinhardt). Ground truth via a metric-weighted Laplace–Beltrami operator on a dense $(\theta,\varphi)$ grid with RK4, interpolated to the cloud — independent of the model's graph. |
| `model.py` | The CASIN network |
| `train_eval.py` | training/evaluation |
| `calibrate.py` | Evaluation protocols and RMSE reporting. |

---

## Quick start

Train and evaluate a single configuration:

```bash
python train_eval.py --manifold torus --pde fisher_kpp --seeds 5 --device cpu
```

Reproduce the main results table:

```bash
python reproduce.py --seeds 5 --device cuda
```



---

## Configuration

Default configuration

| Parameter | Value |
|---|---|
| Hidden dimension | 96 |
| Simplicial convolution layers | 4 |
| Optimizer | Adam, lr $3\times10^{-3}$, cosine annealing |
| Gradient clipping | 1.0 |
| Epochs | 500 |
| Sampled points | 800 |
| Time steps | 40 |
|simplicial complex| 12 |
|normals and curvature| 24 |
| Curvature estimator | degree-3 local polynomial (Monge patch) |

Fields are min–max normalised per channel; reported RMSE is on this normalised scale.

---
