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

**Seeing it.** OpenFold3 writes pLDDT into the mmCIF's B-factor column, so any viewer can
colour the structure by confidence:

<p align="center">
  <img src="assets/ubiquitin_cartoon.png" alt="Ubiquitin cartoon with side chains, coloured by pLDDT" width="380">
</p>

Coloured by [AlphaFold's four confidence bands](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/):
dark blue ≥ 90 (very high), cyan 70–90 (confident), yellow 50–70 (low), orange < 50 (very low).

The β-grasp fold is dark blue throughout, while the C-terminal tail runs through yellow into
orange — pLDDT falls to 43 at residue 76. That is the model correctly reporting that the tail
is flexible in isolation rather than failing to predict it. Low confidence is frequently the
right answer.

<details>
<summary>How this image was made</summary>

Rendered with [3Dmol.js](https://3dmol.csb.pitt.edu/) — the engine behind `py3Dmol`. The four
bands need a colour function, because 3Dmol has no selection syntax for B-factor ranges:

```html
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<div id="v" style="width:1400px;height:1100px;position:relative"></div>
<script>
function band(atom) {               // pLDDT lives in the B-factor column
  if (atom.b >= 90) return "#0053D6";   // very high
  if (atom.b >= 70) return "#65CBF3";   // confident
  if (atom.b >= 50) return "#FFDB13";   // low
  return "#FF7D45";                     // very low
}
const v = $3Dmol.createViewer("v", {backgroundColor: "white"});
v.addModel(cifText, "cif");
v.setStyle({}, {cartoon: {colorfunc: band, thickness: 0.35}});
v.addStyle({not: {atom: ["N", "C", "O"]}}, {stick: {radius: 0.12, colorfunc: band}});
v.zoomTo(); v.rotate(110, "y"); v.zoom(1.3); v.render();
</script>
```

py3Dmol is a JavaScript widget, so it renders in a running notebook but not on GitHub. The
static image above came from putting the above in an HTML page and screenshotting it:

```bash
google-chrome --headless --disable-gpu --screenshot=out.png \
    --window-size=1400,1100 --virtual-time-budget=25000 file:///tmp/view.html
```

`--virtual-time-budget` matters — without it the screenshot fires before WebGL has drawn
anything and you get a blank canvas.

From Python, `py3Dmol` cannot pass a JS function, so the closest one-liner is a continuous
ramp rather than the four bands:

```python
import py3Dmol
view = py3Dmol.view(width=900, height=700)
view.addModel(open("..._model.cif").read(), "cif")
view.setStyle({}, {"cartoon": {"colorscheme":
    {"prop": "b", "gradient": "roygb", "min": 50, "max": 100}}})
view.zoomTo(); view.show()
```

</details>

**Checking it against the experiment.** Ubiquitin has a well-known crystal structure
([1UBQ](https://www.rcsb.org/structure/1UBQ), 1.8 Å), so the prediction can be scored rather
than merely admired:

<p align="center">
  <img src="assets/ubiquitin_vs_1ubq.png" alt="Prediction superposed on the 1UBQ crystal structure" width="320">
</p>

Prediction in navy, crystal structure in grey.

| Superposition | Cα RMSD |
|---|---|
| All 76 residues | 0.73 Å |
| Residues 1–72, excluding the flexible tail | 0.44 Å |

Sub-ångström over the folded core — and the gap between the two rows is the C-terminal tail,
exactly the region pLDDT marked as uncertain. The confidence estimate predicted where the error
would be before any reference was consulted.

<details>
<summary>How to score a prediction against a reference</summary>

```python
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

pred = pdbx.get_structure(pdbx.CIFFile.read(
    "quickstart_inference/ubiquitin/seed_42/ubiquitin_seed_42_sample_1_model.cif"), model=1)
ref = pdbx.get_structure(pdbx.CIFFile.read("1ubq.cif"), model=1)
ref = ref[struc.filter_amino_acids(ref) & ~ref.hetero]      # drop waters and ligands

pred_ca, ref_ca = pred[pred.atom_name == "CA"], ref[ref.atom_name == "CA"]
fitted, transform = struc.superimpose(ref_ca, pred_ca)
print(f"RMSD {struc.rmsd(ref_ca, fitted):.2f} A")

pred_fitted = transform.apply(pred)                          # for rendering the overlay
```

Get the reference with `curl -O https://files.rcsb.org/download/1UBQ.cif`. This works because
the residue numbering matches one to one; different constructs need a sequence alignment first.

</details>

**Reading the numbers.** `plddt` is per-atom confidence from 0 to 100 — above 90 is very high,
70 to 90 confident, below 50 usually disordered. `ptm` is the predicted TM-score for the whole
structure; `iptm` is its interface counterpart and is only meaningful for complexes (it reads
`0.0` for a single chain). `pae` is the predicted aligned error in angstroms: entry (i, j) is
how far off residue *i* is expected to be if you superimpose on residue *j*, which is what tells
you whether two domains are confidently placed relative to one another.

Timing per stage is written to `timing.json` in the same directory.

### Co-folding a protein with a ligand

Ubiquitin is a single chain — the easy case, and the one AlphaFold2 already handled. What
AlphaFold3-style models add is **co-folding**: predicting a complex and its partners together.
The repo ships an example that does this, and whose name is the PDB entry it reproduces:

```bash
pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json examples/example_inference_inputs/query_single_protein_single_ligand.json \
    --use-msa-server true \
    --output-dir ./quickstart_cofold
```

The query is T4 lysozyme L99A (164 residues) plus toluene, given as SMILES (`Cc1ccccc1`) rather
than a sequence. The reference is [7L39](https://www.rcsb.org/structure/7L39) at 1.35 Å — the
classic model system for ligand binding in a buried hydrophobic cavity.

<p align="center">
  <img src="assets/lysozyme_toluene.png" alt="Predicted lysozyme-toluene complex superposed on 7L39" width="560">
</p>

Prediction in navy, crystal structure in grey. Left: the whole complex. Right: the cavity, with
the predicted toluene (blue sticks) on top of the crystallographic one (grey), surrounded by the
side chains lining the pocket.

| | |
|---|---|
| Protein Cα RMSD | 0.23 Å over 162 residues |
| Toluene centroid vs. crystallographic position | 0.33 Å |
| `avg_plddt` / `ptm` / `iptm` | 97.8 / 0.98 / 0.94 |

Two things to notice. The interesting question here is not whether the backbone is right —
lysozyme folds correctly either way — but whether the ligand found the right pocket in the right
pose. It did, to within a third of an ångström.

And `iptm` is finally meaningful. It read `0.0` for ubiquitin because a single chain has no
interface; here it scores the protein–ligand interface at 0.94. For complexes, `iptm` is the
number to watch, not `ptm`.

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

