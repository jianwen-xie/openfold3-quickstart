# OpenFold3 Quickstart

Run [OpenFold3](https://github.com/aqlaboratory/openfold-3) training and inference, and watch
the training curve. This is the **single-GPU** stage; multi-GPU and Slurm multi-node follow.

**[→ Open `quickstart.ipynb`](quickstart.ipynb)** — the same steps, executed, with outputs
kept so you can read them without a GPU.

| Stage | Status |
|---|---|
| Single H100 | this guide |
| Multiple H100s on one node | to follow |
| Multi-node via Slurm | to follow |

Everything here is a smoke-test-sized run: a small subset of the PDB, a few minutes, enough to
confirm the pipeline works end to end and to see losses being logged. The settings to turn it
into a real run are called out as you go.

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

Everything below is copy-pasteable. Needs [pixi](https://pixi.sh) 0.80+, a CUDA GPU, and
roughly 25 GB free — most of it the pixi environment and the model parameters.

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

### 4. Monitoring the training curve

The run writes its metrics to a local wandb file. Convert them to a CSV and plot:

```bash
pixi run -e openfold3-cuda12 python -c "
import csv, glob, json
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

store = datastore.DataStore()
store.open_for_scan(glob.glob('quickstart_run/wandb/offline-run-*/run-*.wandb')[0])
rows = []
while True:
    data = store.scan_data()
    if data is None: break
    rec = pb.Record(); rec.ParseFromString(data)
    if rec.WhichOneof('record_type') != 'history': continue
    row = {i.key or '.'.join(i.nested_key): json.loads(i.value_json) for i in rec.history.item}
    if 'train/loss' in row: rows.append(row)

cols = ['_step', 'train/loss', 'train/distogram_loss', 'train/diffusion_loss']
with open('quickstart_run/metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} steps -> quickstart_run/metrics.csv')"
```

The notebook does this and plots the curve inline. For a quick look from a shell:

```bash
column -s, -t quickstart_run/metrics.csv
```

#### Watching it live instead

Offline logging only yields a curve once the run finishes. For a live dashboard — which is what
you want for long runs, and for the multi-GPU and Slurm stages — use wandb online:

```bash
pixi run -e openfold3-cuda12 wandb login     # once, needs a wandb account
```

Then set `offline: false` in the config's `wandb_config` and drop `WANDB_MODE=offline` from the
launch command. The run streams to your wandb project and the curve updates as it trains.

An already-finished offline run can also be uploaded after the fact:

```bash
pixi run -e openfold3-cuda12 wandb sync quickstart_run/wandb/offline-run-*
```

### Running longer

The default is deliberately short — two epochs of four steps, enough to prove the pipeline.
Eight points is not a curve. To get a trace worth monitoring, raise these in the config:

| Setting | Default here | For a longer run |
|---|---|---|
| `data_module_args.epoch_len` | `4` | `50`+ — steps per epoch |
| `pl_trainer_args.max_epochs` | `2` | `10`+ |
| `--train-size` on `setup-pdb-subset` | `8` | more structures to sample from |

At roughly 10 s per training step on one H100, 100 steps is about 20 minutes plus validation
between epochs. Note that loss will not meaningfully descend on a handful of structures however
long you run — that needs the full dataset.

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

## What's next

| Stage | What changes |
|---|---|
| Multiple H100s, one node | `pl_trainer_args.devices: 8`, same launch command |
| Multi-node via Slurm | `num_nodes: N` plus an `sbatch` submission script |

Both will be added to this guide. Real training also swaps the sampled subset for the full PDB
set and starts from a config in `examples/training_yamls/`:

```bash
aws s3 sync s3://openfold3-data/pdb_training_set/ /shared/openfold3/pdb_training_set/ --no-sign-request
```

## Links

- OpenFold3: <https://github.com/aqlaboratory/openfold-3>
- Training guide: <https://openfold-3.readthedocs.io/en/latest/training.html>
- Configuration reference: <https://openfold-3.readthedocs.io/en/latest/configuration_reference.html>
- Datasets: <https://portal.openfold.omsf.io/datasets>

## License

Apache-2.0. See [LICENSE](LICENSE).
