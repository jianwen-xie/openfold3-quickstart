# OpenFold3 Training Quickstart

An executable notebook that takes [OpenFold3](https://github.com/aqlaboratory/openfold-3) from a
fresh checkout to a completed training run on **one GPU** — environment, a small real dataset,
the config that matters, the launch, and how to read what came out.

**[→ Open `quickstart.ipynb`](quickstart.ipynb)** — it ships with outputs, so you can read every
result without a GPU.

This is the single-GPU rung. The same config scales to multi-GPU and multi-node by changing two
numbers; the notebook ends by showing which.

## Training OpenFold3 is lighter than it sounds

The phrase "train an AlphaFold3-style model" suggests a cluster and a storage budget. For a run
that proves your setup works, it doesn't:

| | |
|---|---|
| Training data | **71 MB** (8 training + 4 validation structures) |
| Peak GPU memory | **28.3 GiB** at `token_budget: 384` — fits a 40 GB card |
| Wall time | **~5 minutes** end to end, 2 epochs |
| Environment | ~14 GB (pixi) |
| Per-run checkpoint | ~6 GB |

Measured on a single [NVIDIA H100 80GB HBM3](https://www.nvidia.com/en-us/data-center/h100/).
The full tables are in the notebook.

## What the notebook covers

1. Environment check — GPU, torch/CUDA, `openfold3` version
2. Building the training subset, and a path trap that silently breaks the run
3. The config settings that decide whether it runs and how much memory it takes
4. Launching the run
5. Reading the results — losses, peak memory, where the checkpoint went
6. What changes for multi-GPU and the full dataset

## Three things that cost time to discover

**The committed config is a decoy.** `scripts/datasets/train_pdb_subset.yaml` looks like the
config to use, but it still holds the absolute paths of whoever generated it. The real one is
written to `datasets/train_pdb_subset.yaml` at the repo root, with your paths.

**A successful run records nothing by default.** `LoggingConfig` accepts Weights & Biases or
nothing, and the generated config sets `wandb_config: null`. Every loss — `distogram`,
`diffusion`, `plddt`, `pae` and the rest — is computed and discarded, leaving an empty `logs/`
directory next to a 6 GB checkpoint.

Worse, the obvious fix doesn't work. Setting `wandb_config.offline: true` in the YAML is not
enough — the flag never reaches `wandb.init`, which still tries to log in and kills the run
before training starts. The environment variable `WANDB_MODE=offline` is what actually works.
With it, metrics are captured locally with no account; the notebook then converts them to a
`metrics.csv`.

**Enabling logging moves the checkpoint.** It goes from `<output_dir>/checkpoints/` to
`<output_dir>/<project>/<run id>/checkpoints/`, so anything with a hardcoded path breaks the
moment you turn logging on.

## Prerequisites

The notebook drives an existing checkout rather than installing one:

```bash
git clone https://github.com/aqlaboratory/openfold-3.git
cd openfold-3
pixi install -e openfold3-cuda12
```

Needs [pixi](https://pixi.sh) 0.80+, a CUDA GPU, and about 20 GB free. Point the notebook at the
checkout with the `OPENFOLD3_DIR` environment variable, or edit the path in its first cell.

Most cells shell out through `pixi run`, so the kernel only needs to start subprocesses — except
the results cell, which imports `wandb` to read the metrics and therefore wants the pixi
environment as its kernel:

```bash
pixi run -e openfold3-cuda12 python -m pip install ipykernel
pixi run -e openfold3-cuda12 python -m ipykernel install --user \
    --name openfold3-cuda12 --display-name "OpenFold3 (pixi cuda12)"
```

## A note on what a smoke test shows

The loss in the notebook does not go down, and it isn't presented as if it might. Eight
optimiser steps over eight structures proves the data pipeline, model, loss, backward pass and
checkpoint writer work together — nothing about convergence. Real training is orders of
magnitude more steps against the full PDB set.

## Links

- OpenFold3: <https://github.com/aqlaboratory/openfold-3>
- Training guide: <https://openfold-3.readthedocs.io/en/latest/training.html>
- Configuration reference: <https://openfold-3.readthedocs.io/en/latest/configuration_reference.html>
- Datasets: <https://portal.openfold.omsf.io/datasets>

## License

Apache-2.0. See [LICENSE](LICENSE).
