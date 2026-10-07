# OpenFold3 Quickstart

Install [OpenFold3](https://github.com/aqlaboratory/openfold-3), predict a structure, and train
on one GPU. Every command in the notebook is the command you type — it runs them as shell cells
and keeps the real output.

**[→ Open `quickstart.ipynb`](quickstart.ipynb)** — it ships with outputs, so you can read every
result without a GPU.

| Section | What it does |
|---|---|
| 1. Install | pixi environment, model parameters |
| 2. Inference | predict ubiquitin from sequence |
| 3. Training | two epochs on a small PDB subset |
| 4. Scaling | what changes for multi-GPU and the full dataset |

## Measured on one H100

| | |
|---|---|
| Inference | 5 diffusion samples, mean pLDDT ~90, ~10 s of GPU time |
| Training | 8 steps in ~5 minutes |
| Peak GPU memory | **28.3 GiB** at `token_budget: 384` — fits a 40 GB card |
| Training data | 71 MB (8 train + 4 validation structures) |
| Model parameters | 4.3 GB |
| pixi environment | ~14 GB |
| Checkpoint per run | 6.3 GB |

Measured on a single [NVIDIA H100 80GB HBM3](https://www.nvidia.com/en-us/data-center/h100/)
with `openfold3` 0.5.1.dev55, torch 2.10.0 / CUDA 12.9.

Training OpenFold3 is lighter than the phrase suggests: for a run that proves your setup works,
the dataset is tens of megabytes and the model fits on a mid-range card.

## Three things that cost time to discover

**The committed config is a decoy.** `scripts/datasets/train_pdb_subset.yaml` looks like the
config to use, but it still holds the absolute paths of whoever generated it. The generator
writes a fresh copy, with your paths, to `datasets/train_pdb_subset.yaml`. The notebook prints
both side by side.

**A successful run records nothing by default, and the obvious fix doesn't work.**
`LoggingConfig` accepts Weights & Biases or nothing, and the generated config sets
`wandb_config: null` — so every loss is computed and thrown away, leaving an empty `logs/`
directory beside a 6 GB checkpoint. Setting `wandb_config.offline: true` in the YAML is *not*
enough: the flag never reaches `wandb.init`, which still tries to log in and aborts the run
before training starts. The environment variable `WANDB_MODE=offline` is what works.

**Enabling logging moves the checkpoint**, from `<output_dir>/checkpoints/` to
`<output_dir>/<project>/<run id>/checkpoints/`. Anything with a hardcoded path breaks the moment
you turn logging on.

## Running it

Everything below is copy-pasteable. Needs [pixi](https://pixi.sh) 0.80+, a CUDA GPU
(H100/H200 class for these figures), and roughly 25 GB free.

### 1. Install

```bash
git clone https://github.com/aqlaboratory/openfold-3.git
cd openfold-3

# ~14 GB environment
pixi install -e openfold3-cuda12

# model parameters -> ~/.openfold3/ (~4.3 GB); idempotent
pixi run -e openfold3-cuda12 setup_openfold --non-interactive
```

Check it worked:

```bash
pixi run -e openfold3-cuda12 python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. Inference

Predict a structure from sequence. The repo ships example queries; ubiquitin is the smallest
at 76 residues.

```bash
pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json examples/example_inference_inputs/query_ubiquitin.json \
    --use-msa-server true \
    --output-dir ./quickstart_inference
```

`--use-msa-server true` uses the public ColabFold server, so this step needs outbound network
and is subject to their rate limits. Results land in
`quickstart_inference/ubiquitin/seed_42/` — five diffusion samples, each an mmCIF structure
with a matching `_confidences.json` holding per-atom `plddt`, plus `pae` and `pde`.

### 3. Training

Sample a small subset and download only those structures — 8 training and 4 validation by
default, about 71 MB. The first run also pulls a 1.6 GB index to sample from, which is not
training data and can be deleted afterwards.

```bash
pixi run -e openfold3-cuda12 setup-pdb-subset
```

This writes a config to `datasets/train_pdb_subset.yaml`. **Use that one** — the committed
`scripts/datasets/train_pdb_subset.yaml` is a template still holding another user's absolute
paths.

Add offline logging, or the run will discard every loss it computes:

```bash
pixi run -e openfold3-cuda12 python -c "
import yaml
c = yaml.safe_load(open('datasets/train_pdb_subset.yaml'))
c['logging_config'] = {'log_lr': True, 'wandb_config': {
    'project': 'openfold3-quickstart', 'experiment_name': 'single-h100', 'offline': True}}
c['experiment_settings']['output_dir'] = './quickstart_run'
yaml.safe_dump(c, open('datasets/quickstart_single_h100.yaml', 'w'), sort_keys=False)"
```

Then train. `WANDB_MODE=offline` is required — the YAML flag alone does not prevent a login
attempt:

```bash
WANDB_MODE=offline pixi run -e openfold3-cuda12 run_openfold train \
    --runner-yaml datasets/quickstart_single_h100.yaml --seed 42
```

Two epochs of four steps, a few minutes. The checkpoint lands in
`quickstart_run/openfold3-quickstart/<run id>/checkpoints/last.ckpt` and the losses in
`quickstart_run/wandb/offline-run-*/run-*.wandb`. The notebook has a cell that converts that
binary log to a CSV.

### Running the notebook instead

The notebook runs all of the above and keeps the output. Most of its cells are these same
shell commands; two are Python analysis, and one imports `wandb`, so the pixi environment is
the most convenient kernel:

```bash
pixi run -e openfold3-cuda12 python -m pip install ipykernel
pixi run -e openfold3-cuda12 python -m ipykernel install --user \
    --name openfold3-cuda12 --display-name "OpenFold3 (pixi cuda12)"
```

Then edit the `%cd` in its first cell to point at your checkout.

## What a smoke test shows

The training loss does not go down, and the notebook doesn't present it as if it might. Eight
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
