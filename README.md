# OpenFold3 Quickstart

Run [OpenFold3](https://github.com/aqlaboratory/openfold-3) inference and training, and watch
the training curve.

**[→ Open `quickstart.ipynb`](quickstart.ipynb)** — the single-GPU path, executed, with outputs
kept so you can read them without a GPU.

| Stage | Inference | Training |
|---|---|---|
| One GPU | documented, executed | documented, executed |
| Multiple GPUs, one node | to follow | to follow |
| Multiple nodes (Slurm) | to follow | to follow |

Everything here is a smoke-test-sized run: a small subset of the PDB, a few minutes, enough to
confirm the pipeline works end to end and to see losses being logged. The settings to turn it
into a real run are called out as you go.

Needs [pixi](https://pixi.sh) 0.80+, a CUDA GPU, and roughly 25 GB free — most of it the pixi
environment and the model parameters.

## Install

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

## Inference

### On one GPU

Predict a structure from sequence. The repo ships example queries; ubiquitin is the smallest
at 76 residues.

```bash
pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json examples/example_inference_inputs/query_ubiquitin.json \
    --use-msa-server true \
    --output-dir ./quickstart_inference
```

`--use-msa-server true` uses the public ColabFold server, so this step needs outbound network
and is subject to their rate limits.

Results land under the `--output-dir` you passed, in a directory named after the query key and
the seed:

```
quickstart_inference/
├── ubiquitin/seed_42/              ← query key / seed
│   ├── ..._sample_1_model.cif                      the structure
│   ├── ..._sample_1_confidences.json               full arrays
│   ├── ..._sample_1_confidences_aggregated.json    summary -- rank on this
│   ├── ...                                         (×5 samples)
│   └── timing.json
├── msas/                           alignments fetched from ColabFold
├── summary.txt                     queries succeeded / failed
├── model_config.json               fully-resolved config
└── openfold3_intermediates/
```

The structure module is a diffusion model, so one run produces **five alternative structures**
rather than one — they are separate samples, not successive refinements. Per sample:

| File | Contents |
|---|---|
| `*_model.cif` | the predicted structure, mmCIF — opens in PyMOL or ChimeraX |
| `*_confidences_aggregated.json` | per-sample summary: `avg_plddt`, `ptm`, `iptm`, `has_clash`, `sample_ranking_score` |
| `*_confidences.json` | the full arrays: per-atom `plddt`, and `pae` / `pde` as residue × residue matrices |

**Picking a sample.** Rank by `sample_ranking_score` in the aggregated file:

```bash
pixi run -e openfold3-cuda12 python -c "
import glob, json
for f in sorted(glob.glob('quickstart_inference/ubiquitin/seed_42/*_aggregated.json')):
    d = json.load(open(f))
    n = f.split('_sample_')[1].split('_')[0]
    print(f'sample {n}  score {d[\"sample_ranking_score\"]:.3f}  plddt {d[\"avg_plddt\"]:.1f}'
          f'  ptm {d[\"ptm\"]:.3f}  clash {d[\"has_clash\"]:.0f}')"
```

**Reading the numbers.** `plddt` is per-atom confidence from 0 to 100 — above 90 is very high,
70 to 90 confident, below 50 usually disordered. `ptm` is the predicted TM-score for the whole
structure; `iptm` is its interface counterpart and is only meaningful for complexes (it reads
`0.0` for a single chain). `pae` is the predicted aligned error in angstroms: entry (i, j) is
how far off residue *i* is expected to be if you superimpose on residue *j*, which is what tells
you whether two domains are confidently placed relative to one another.

Timing per stage is written to `timing.json` in the same directory.

### Across multiple GPUs

*To follow.* OpenFold3 distributes a multi-query job across GPUs — parallel across queries,
not within a single structure. See the upstream
[inference docs](https://openfold-3.readthedocs.io/en/latest/inference.html#inference-run-on-multiple-gpus)
in the meantime.

## Training

### On one GPU

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

Two epochs of four steps, a few minutes. Output goes to the `output_dir` set in the config:

```
quickstart_run/
├── openfold3-quickstart/<run id>/checkpoints/
│   └── last.ckpt                   the model, ~6 GB
├── wandb/offline-run-<ts>-<run id>/
│   ├── run-<run id>.wandb          the losses -- binary, see the next section
│   └── files/                      resolved configs, package versions
└── logs/                           always empty
```

Two things to know about that layout. Enabling logging **moves the checkpoint**: without wandb
it is simply `quickstart_run/checkpoints/`, so anything with a hardcoded path breaks the moment
you turn logging on. And `logs/` looks like where you would check what happened, but never
contains anything — the metrics are only inside the binary `.wandb` file.

### Monitoring the training curve

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

### Training on the full dataset

The subset above exists so the pipeline can be proven quickly. Real training swaps it for the
full PDB set and starts from one of the configs in `examples/training_yamls/` —
`initial_training.yml` through `finetune_3.yml` — which carry real schedules rather than a
two-epoch smoke test.

```bash
aws s3 sync s3://openfold3-data/pdb_training_set/ /shared/openfold3/pdb_training_set/ --no-sign-request
```

### Across multiple GPUs, one node

*To follow.* Set `pl_trainer_args.devices: 8`; the launch command is unchanged.

### Across multiple nodes (Slurm)

*To follow.* Set `pl_trainer_args.num_nodes: N` and submit with `sbatch`. Submission scripts
will live in `slurm/`.

## Running the notebook instead

The notebook runs all of the above and keeps the output. Most of its cells are these same
shell commands; two are Python analysis, and one imports `wandb`, so the pixi environment is
the most convenient kernel:

```bash
pixi run -e openfold3-cuda12 python -m pip install ipykernel
pixi run -e openfold3-cuda12 python -m ipykernel install --user \
    --name openfold3-cuda12 --display-name "OpenFold3 (pixi cuda12)"
```

Then edit the `%cd` in its first cell to point at your checkout.

## If something goes wrong

| Symptom | Cause |
|---|---|
| Run exits before training starts, traceback mentions `wandb_login` | `wandb_config.offline: true` does not stop `wandb.init` from attempting a login. Set `WANDB_MODE=offline` in the environment, as above. |
| Missing-file errors for dataset caches or structures | You pointed `--runner-yaml` at `scripts/datasets/train_pdb_subset.yaml`. That is a committed template holding another user's absolute paths; use the generated `datasets/train_pdb_subset.yaml`. |
| Training finished but `logs/` is empty and there are no metrics | `LoggingConfig` takes Weights & Biases or nothing, and defaults to `wandb_config: null` — every loss is computed and discarded. Add the `logging_config` block from step 3. |
| Scripts can't find the checkpoint | Enabling logging moves it from `<output_dir>/checkpoints/` to `<output_dir>/<project>/<run id>/checkpoints/`. |
| Inference hangs or fails at the MSA stage | `--use-msa-server true` calls the public ColabFold server — check outbound network and their rate limits. |

## Links

- OpenFold3: <https://github.com/aqlaboratory/openfold-3>
- Training guide: <https://openfold-3.readthedocs.io/en/latest/training.html>
- Configuration reference: <https://openfold-3.readthedocs.io/en/latest/configuration_reference.html>
- Datasets: <https://portal.openfold.omsf.io/datasets>

## License

Apache-2.0. See [LICENSE](LICENSE).

