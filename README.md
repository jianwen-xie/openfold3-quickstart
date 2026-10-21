# OpenFold3 Quickstart

Run [OpenFold3](https://github.com/aqlaboratory/openfold-3) inference and training, and watch
the training curve.

**[→ Open `quickstart.ipynb`](quickstart.ipynb)**: the single-GPU path, executed, with outputs
kept so you can read them without a GPU.

| Stage | Inference | Training |
|---|---|---|
| One GPU | documented, executed | documented, executed |
| Multiple GPUs, one node | to follow | to follow |
| Multiple nodes (Slurm) | to follow | to follow |

Everything here is a smoke-test-sized run: a small subset of the PDB, a few minutes, enough to
confirm the pipeline works end to end and to see losses being logged. The settings to turn it
into a real run are called out as you go.

Inference is worked through three kinds of input (a single protein, a protein with a ligand,
and a multi-chain complex), each scored against its crystal structure, because the confidence
metrics worth reading differ in each case.

Needs [pixi](https://pixi.sh) 0.80+, a CUDA GPU, and roughly 25 GB free, most of it the pixi
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
rather than one. Per sample:

| File | Contents |
|---|---|
| `*_model.cif` | the predicted structure, mmCIF; opens in PyMOL or ChimeraX |
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

Everything above is the same for any query. What differs between inputs is which of these
numbers is worth looking at, so the rest of this section works three examples:

1. **A single protein**: ubiquitin. Read pLDDT.
2. **A protein with a ligand**: T4 lysozyme and toluene. Read the ligand pose.
3. **Co-folding a complex**: a four-chain assembly. Read `iptm` and `chain_pair_iptm`.

#### (1) A single protein

Ubiquitin, as launched above. For a single chain, pLDDT is the figure that matters. There are
no interfaces, so `iptm` reads `0.0` and carries no information.

**Seeing it.** OpenFold3 writes pLDDT into the mmCIF's B-factor column, so any viewer can
colour the structure by confidence:

<p align="center">
  <img src="assets/ubiquitin_cartoon.png" alt="Ubiquitin cartoon with side chains, coloured by pLDDT" width="380">
</p>

Coloured by [AlphaFold's four confidence bands](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/):
dark blue ≥ 90 (very high), cyan 70–90 (confident), yellow 50–70 (low), orange < 50 (very low).

The β-grasp fold is dark blue throughout, while the C-terminal tail runs through yellow into
orange, with pLDDT falling to 43 at residue 76. That is the model correctly reporting that the tail
is flexible in isolation rather than failing to predict it. Low confidence is frequently the
right answer.

<details>
<summary>How this figure was made</summary>

`scripts/render_figure.py` in this repo does the whole job: superpose, render through
headless Chrome, then trim and scale to a size that belongs in a README.

Set it up once inside your `openfold-3` checkout, keeping the script, the references and the
output together in `figures/`. Anything scored against an experimental structure needs that
structure, and RCSB serves them under an uppercase id, so name the download with `-o`:

```bash
mkdir -p figures && cd figures
curl -O https://raw.githubusercontent.com/jianwen-xie/openfold3-quickstart/main/scripts/render_figure.py
curl -o 1ubq.cif https://files.rcsb.org/download/1UBQ.cif
cd ..
```

Then render, from the root of the `openfold-3` checkout:

```bash
pixi run -e openfold3-cuda12 python figures/render_figure.py plddt \
    quickstart_inference/ubiquitin/seed_42/ubiquitin_seed_42_sample_1_model.cif \
    figures/ubiquitin_cartoon.png --fit-to figures/1ubq.cif --rotate 60 --zoom 1.3 \
    --width 620 --canvas 1400x1100
```

Running it through `pixi run` is what supplies biotite. Chrome and ImageMagick are not in the
pixi environment; if either is missing the script says which one and stops. To install them,
`sudo apt install chromium-browser imagemagick` on Debian or Ubuntu, or
`brew install --cask google-chrome && brew install imagemagick` on macOS.

`--fit-to` renders the prediction **after superposing it on the reference**, so this figure and
the superposition below look at the molecule from the same angle. Rendering the raw file gives
the same colours in a different orientation, and the two figures become hard to compare.

The four bands come from a JavaScript colour callback rather than `py3Dmol`, which cannot pass
a function from Python; its nearest equivalent is a continuous rainbow ramp, which is a
different figure.

</details>

**Checking it against the experiment.** Ubiquitin has a well-known crystal structure
([1UBQ](https://www.rcsb.org/structure/1UBQ), 1.8 Å), so the prediction can be scored rather
than merely admired:

<p align="center">
  <img src="assets/ubiquitin_vs_1ubq.png" alt="Prediction superposed on the 1UBQ crystal structure" width="320">
</p>

Prediction in navy, crystal structure in grey.

<details>
<summary>How this figure was made</summary>

```bash
pixi run -e openfold3-cuda12 python figures/render_figure.py super \
    quickstart_inference/ubiquitin/seed_42/ubiquitin_seed_42_sample_1_model.cif \
    figures/1ubq.cif figures/ubiquitin_vs_1ubq.png \
    --rotate 60 --zoom 1.3 --width 540 --canvas 1400x1100
```

`--rotate 60` is the same angle as the confidence figure above, so the two can be read
together.

</details>

| Superposition | Cα RMSD |
|---|---|
| All 76 residues | 0.73 Å |
| Residues 1–72, excluding the flexible tail | 0.44 Å |

Sub-ångström over the folded core, and the gap between the two rows is the C-terminal tail,
exactly the region pLDDT marked as uncertain. The confidence estimate predicted where the error
would be before any reference was consulted.

<details>
<summary>How these numbers were made</summary>

`score` reports the RMSD without rendering, so it needs neither Chrome nor ImageMagick:

```bash
pixi run -e openfold3-cuda12 python figures/render_figure.py score \
    quickstart_inference/ubiquitin/seed_42/ubiquitin_seed_42_sample_1_model.cif \
    figures/1ubq.cif --residues 1-72
```

```
  all matched atoms       0.73 A  over 76
  residues 1-72           0.44 A  over 72
```

`--residues` produces the second row: each range is superposed on its own, so it measures that
region's accuracy rather than its placement in the whole.

</details>

**Reading the numbers.** `plddt` is per-atom confidence from 0 to 100: above 90 is very high,
70 to 90 confident, below 50 usually disordered. `ptm` is the predicted TM-score for the whole
structure; `iptm` is its interface counterpart and is only meaningful for complexes (it reads
`0.0` for a single chain). `pae` is the predicted aligned error in angstroms: entry (i, j) is
how far off residue *i* is expected to be if you superimpose on residue *j*, which is what tells
you whether two domains are confidently placed relative to one another.

Timing per stage is written to `timing.json` in the same directory.

#### (2) A protein with a ligand

The simplest co-folding case: one protein and one small molecule. The repo ships an example,
named after the PDB entry it reproduces:

```bash
pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json examples/example_inference_inputs/query_single_protein_single_ligand.json \
    --use-msa-server true \
    --output-dir ./quickstart_cofold
```

The query is T4 lysozyme L99A (164 residues) plus toluene, given as SMILES (`Cc1ccccc1`)
rather than a sequence. The reference is [7L39](https://www.rcsb.org/structure/7L39) at
1.35 Å, the classic model system for ligand binding in a buried hydrophobic cavity. The output
layout, the five samples and the ranking step are all as described above.

Confidence is not the interesting question here: lysozyme folds correctly either way, and
96.3% of atoms score ≥ 90. What matters is whether the ligand found the right pocket in the
right pose, so the figure to look at is the superposition:

<p align="center">
  <img src="assets/lysozyme_toluene.png" alt="Predicted lysozyme-toluene complex superposed on 7L39" width="600">
</p>

Prediction in navy, crystal structure in grey. **Left:** the whole complex. The binding
cavity is buried, so the ligand is only glimpsed through the fold. **Right:** the cavity
close up, with the protein translucent, where the **blue ring is the predicted ligand** and
the **black ring beneath it the crystallographic one**, among the side chains lining the
pocket.

<details>
<summary>How this figure was made</summary>

Two renders, side by side. The second uses `--focus`, which frames a bound ligand and draws
the protein translucent with the pocket's side chains around it:

```bash
curl -o figures/7l39.cif https://files.rcsb.org/download/7L39.cif

pixi run -e openfold3-cuda12 python figures/render_figure.py super \
    quickstart_cofold/pdb_7L39/seed_42/pdb_7L39_seed_42_sample_1_model.cif figures/7l39.cif \
    figures/lyso_left.png --ref-ligand MBN \
    --rotate 0 --zoom 1.2 --width 420 --canvas 1000x1000

pixi run -e openfold3-cuda12 python figures/render_figure.py super \
    quickstart_cofold/pdb_7L39/seed_42/pdb_7L39_seed_42_sample_1_model.cif figures/7l39.cif \
    figures/lyso_right.png --focus LIG0 --focus-ref MBN --zoom 0.62 --width 420 --canvas 1000x1000

convert \
    \( figures/lyso_left.png  -background white -gravity east -splice 110x0 \) \
    \( figures/lyso_right.png -background white -gravity west -splice 110x0 \) \
    -background white -gravity center +append \
    -resize 900x -colors 256 -strip figures/lysozyme_toluene.png
```

The crystal's ligand has to be named, as `--ref-ligand` or `--focus-ref`, because the two
files call the same molecule different things: `LIG0` in the prediction, `MBN` in the crystal.

Only the close-up is `--translucent`. It is the only way to see into a buried cavity, but it
washes out the navy against grey that makes the left panel readable.

</details>


| Superposition | Deviation |
|---|---|
| Protein Cα, 162 residues | 0.23 Å |
| Toluene centroid vs. crystallographic position | 0.33 Å |

The ligand found the right pocket in the right pose, to within a third of an ångström.

`iptm` is finally meaningful here, scoring 0.94 alongside `avg_plddt` 97.8 and `ptm` 0.98. It
read `0.0` for ubiquitin, where a single chain has no interface to score. For anything with
more than one component it is the number to watch, not `ptm`.

<details>
<summary>How these numbers were made</summary>

`avg_plddt`, `ptm` and `iptm` are read from the aggregated confidences file that inference
writes. The rest are measured:

```bash
pixi run -e openfold3-cuda12 python figures/render_figure.py score \
    quickstart_cofold/pdb_7L39/seed_42/pdb_7L39_seed_42_sample_1_model.cif figures/7l39.cif \
    --ref-ligand MBN --plddt
```

```
  all matched atoms       0.23 A  over 162
  ligand centroid         0.33 A  (LIG0 vs MBN)
  mean pLDDT              97.8  over 1312 atoms
     very high >=90    96.3%
     confident 70-90    3.6%
     low 50-70          0.1%
     very low <50       0.0%
```

The ligand distance is measured after the protein has been superposed, so it reports the
pose rather than the fold. `--plddt` reads the B-factor column, which is where OpenFold3
stores per-atom confidence.

</details>

#### (3) Co-folding a complex

The ligand case has a single protein chain. The harder version is several chains that have to
fold *and* find their arrangement with one another. The repo ships one:

```bash
pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json examples/example_inference_inputs/query_multimer.json \
    --use-msa-server true \
    --output-dir ./quickstart_multimer
```

The query is phosphatidylserine decarboxylase from *E. coli*, reproducing
[7CNX](https://www.rcsb.org/structure/7CNX) at 2.63 Å. Two sequences, four chains: a
253-residue β subunit supplied as chains `A` and `C`, and a 42-residue α subunit as `B` and
`D`. The two come from a single precursor that cleaves itself in half, which the crystal's
numbering still records, with β running 1–253 and α picking up at 255.

<p align="center">
  <img src="assets/psd_chains.png" alt="Predicted assembly, one colour per chain" width="320">
</p>

One colour per chain: β subunits in dark and light blue (`A`, `C`), α subunits in red and
orange (`B`, `D`). Each small α subunit packs against its own β partner, and the two αβ
heterodimers meet through a β–β interface in the middle. Getting this right means getting four
chains and three distinct interfaces right at once.

<details>
<summary>How this figure was made</summary>

```bash
curl -o figures/7cnx.cif https://files.rcsb.org/download/7CNX.cif

pixi run -e openfold3-cuda12 python figures/render_figure.py chains quickstart_multimer/7cnx/seed_42/7cnx_seed_42_sample_1_model.cif \
    figures/psd_chains.png --fit-to figures/7cnx.cif --ref-chains A,B,C,D \
    --rotate 20 --tilt 10 --zoom 1.10 --width 640 --canvas 1050x1200 \
    --colour 'A:#27408B' --colour 'C:#7FB2E5' --colour 'B:#C2461E' --colour 'D:#F0A33C'
```

The `--colour` pins give the two β subunits one hue and the two α subunits another, so each
pair reads as a pair rather than as four unrelated chains.

</details>

<p align="center">
  <img src="assets/psd_vs_7cnx.png" alt="Predicted assembly superposed on the 7CNX crystal structure" width="320">
</p>

Prediction in navy, crystal structure in grey, same orientation as above. The colours no
longer identify chains as they did in the previous figure; here they separate the prediction
from the experiment.

<details>
<summary>How this figure was made</summary>

```bash
curl -o figures/7cnx.cif https://files.rcsb.org/download/7CNX.cif

pixi run -e openfold3-cuda12 python figures/render_figure.py super  quickstart_multimer/7cnx/seed_42/7cnx_seed_42_sample_1_model.cif \
    figures/7cnx.cif figures/psd_vs_7cnx.png --ref-chains A,B,C,D \
    --rotate 20 --tilt 10 --zoom 1.10 --width 640 --canvas 1050x1200
```

`--ref-chains A,B,C,D` picks the matching copy of the tetramer: 7CNX holds two in its
asymmetric unit. It reports 0.57 Å over 557 atoms, the first row of the table below.

</details>

| Superposition | Cα RMSD |
|---|---|
| All four chains at once, 557 residues | 0.57 Å |
| β subunit `A` alone, 246 residues | 0.29 Å |
| α subunit `B` alone, 38 residues | 1.11 Å |
| β subunit `C` alone, 240 residues | 0.35 Å |
| α subunit `D` alone, 33 residues | 0.20 Å |

The first row is the one that matters. Superposing all four chains together scores the folds
*and* their arrangement at the same time, so 0.57 Å means the assembly is right, not merely
four chains that are each correct but wrongly placed. The navy helix on the left with no grey
beneath it is a stretch the crystal never resolved; the reference chain starts at residue 8.

<details>
<summary>How these numbers were made</summary>

```bash
pixi run -e openfold3-cuda12 python figures/render_figure.py score \
    quickstart_multimer/7cnx/seed_42/7cnx_seed_42_sample_1_model.cif figures/7cnx.cif \
    --ref-chains A,B,C,D --per-chain
```

`--per-chain` adds the rows below the first, superposing each chain on its own instead of
moving the assembly as a unit.

</details>

**Reading `chain_pair_iptm`.** Overall the run scores `avg_plddt` 92.8, `ptm` 0.90, `iptm` 0.89
and no clashes. But the aggregated file also scores every *pair* of chains separately, and this
is where a complex tells you more than any single number can:

| | measured closest approach | `chain_pair_iptm` |
|---|---|---|
| `(A, B)`, α against its own β | 2.72 Å | 0.96 |
| `(A, C)`, the β–β interface | 2.65 Å | 0.85 |
| `(A, D)`, across the assembly | 11.42 Å | 0.78 |
| `(B, C)`, across the assembly | 11.34 Å | 0.78 |
| `(B, D)`, the two α subunits | 27.08 Å | 0.54 |
| `(C, D)`, α against its own β | 2.72 Å | 0.96 |

The two αβ pairs score identically, `(A, B)` and `(C, D)` both at 2.72 Å and 0.96, which is the
symmetry of the assembly showing up in the numbers.

The low score is the interesting one, and it is not a warning. `(B, D)` reads 0.54 because
those two chains sit 27 Å apart and never touch. The metric is scoring an interface that does
not exist. A low `chain_pair_iptm` means *this pair is not confidently placed relative to each
other*, which for a non-contacting pair is both true and harmless. Check it against the
geometry before treating it as a failure.

<details>
<summary>How these numbers were made</summary>

The `chain_pair_iptm` column is read from the file inference already writes:

```bash
python3 -c "
import json
f = 'quickstart_multimer/7cnx/seed_42/7cnx_seed_42_sample_1_confidences_aggregated.json'
d = json.load(open(f))
print({k: d[k] for k in ('avg_plddt', 'ptm', 'iptm', 'has_clash')})
print(d['chain_pair_iptm'])"
```

The distances are measured against the crystal:

```bash
pixi run -e openfold3-cuda12 python figures/render_figure.py score \
    quickstart_multimer/7cnx/seed_42/7cnx_seed_42_sample_1_model.cif figures/7cnx.cif \
    --ref-chains A,B,C,D --contacts
```

</details>

### Across multiple GPUs

*To follow.* OpenFold3 distributes a multi-query job across GPUs, parallel across queries
rather than within a single structure. See the upstream
[inference docs](https://openfold-3.readthedocs.io/en/latest/inference.html#inference-run-on-multiple-gpus)
in the meantime.

## Training

### Prepare the training data

Generate a subset of the PDB training data:

```bash
# --train-size  how many training structures to sample
# --seed        fixes which ones, so the subset is reproducible
pixi run -e openfold3-cuda12 generate-pdb-subset --train-size 8 --seed 42
```

Download the corresponding data:

```bash
# --train-size  must match the line above; each command finds its cache by this number
# --workers     parallel download threads (default 8)
# ends on "1 failed": one validation template is absent from the bucket, which is harmless
pixi run -e openfold3-cuda12 download-pdb-subset --train-size 8 --workers 24
```

The dataset configuration for training is written to `datasets/train_pdb_subset.yaml`, and the
default `max_epochs` is 2. The 1.6 GB catalogue the first command fetches is a listing of the
whole PDB training set rather than training data, and can be deleted once the subset caches
exist.

### On one GPU

Training streams its losses to a Weights & Biases dashboard, which needs a free account.
`wandb login` asks for an API key, created at [wandb.ai/authorize](https://wandb.ai/authorize),
and saves it to `~/.netrc`. With no terminal to prompt in, a batch job for instance, set
`WANDB_API_KEY` in the environment instead.

```bash
pixi run -e openfold3-cuda12 wandb login
```

If you would rather not create one, skip this step and open
[Without a wandb account](#without-a-wandb-account) at the end of this section.

The generated config logs nothing, so every loss it computes is discarded. Copy it and turn
logging on. Copy rather than edit in place: the file's first line reads
`# Auto-generated by generate_subset_cache.py -- do not edit by hand`, and re-running the
generator would overwrite it.

```bash
cp datasets/train_pdb_subset.yaml datasets/quickstart_single_h100.yaml
```

Then change three values in the copy:

```yaml
experiment_settings:
  output_dir: ./quickstart_run     # was ./test_train_output, renamed to match this guide

logging_config:
  log_lr: true                     # was false
  wandb_config:                    # was null, which is what discards the losses
    project: openfold3-quickstart
    experiment_name: single-h100
```

<details>
<summary>The same three edits as a command</summary>

```bash
pixi run -e openfold3-cuda12 python -c "
import yaml

# read the config generate-pdb-subset wrote
c = yaml.safe_load(open('datasets/train_pdb_subset.yaml'))

# it ships with wandb_config: null, which is why losses are computed and thrown away
c['logging_config'] = {'log_lr': True, 'wandb_config': {
    'project': 'openfold3-quickstart', 'experiment_name': 'single-h100'}}

# checkpoints and metrics land here
c['experiment_settings']['output_dir'] = './quickstart_run'

# write a NEW file; train_pdb_subset.yaml is left exactly as generated
yaml.safe_dump(c, open('datasets/quickstart_single_h100.yaml', 'w'), sort_keys=False)"
```

`yaml.safe_dump` rewrites the file in its own style, so comments and key order from the
generated config are not preserved. The values are the same either way.

</details>

Train:

```bash
# --runner-yaml points at the file written just above, not at train_pdb_subset.yaml
pixi run -e openfold3-cuda12 run_openfold train \
    --runner-yaml datasets/quickstart_single_h100.yaml --seed 42
```

Two epochs of four steps, a few minutes. The curve appears in your wandb project and updates
as it trains.

Output goes to the `output_dir` set in the config:

```
quickstart_run/
├── openfold3-quickstart/<run id>/checkpoints/
│   └── last.ckpt                   the model, ~6 GB
├── wandb/<run dir>/
│   ├── run-<run id>.wandb          the metrics, binary
│   └── files/                      resolved configs, package versions
└── logs/                           always empty
```

<a id="without-a-wandb-account"></a>

<details>
<summary>Without a wandb account</summary>

No extra edits: the same config as above, the same three values. Skip `wandb login` and prefix
the training command with `WANDB_MODE=offline`:

```bash
WANDB_MODE=offline pixi run -e openfold3-cuda12 run_openfold train \
    --runner-yaml datasets/quickstart_single_h100.yaml --seed 42
```

`WANDB_MODE` is the only switch. `offline: true` under `wandb_config` is accepted by the config
schema but ignored by the trainer, so it will not do this on its own.

Confirm it stayed local. An offline run is written to `offline-run-*`, an uploaded one to `run-*`:

```bash
ls -dt quickstart_run/wandb/*run-* | head -1
```

The metrics are in a binary `.wandb` file. `scripts/read_metrics.py` in this repo decodes it:

```bash
curl -O https://raw.githubusercontent.com/jianwen-xie/openfold3-quickstart/main/scripts/read_metrics.py

pixi run -e openfold3-cuda12 python read_metrics.py quickstart_run --summary
```

`--plot curve.png` draws the curves, which is the offline substitute for the dashboard:

```bash
pixi run -e openfold3-cuda12 python read_metrics.py quickstart_run --plot curve.png
```

`--csv out.csv` exports instead, `--list-keys` shows everything logged, and without `--summary`
it prints every step.

To upload afterwards and get the dashboard back:

```bash
pixi run -e openfold3-cuda12 wandb sync quickstart_run/wandb/offline-run-*
```

</details>

Run length is set in the config: `data_module_args.epoch_len` is steps per epoch and
`pl_trainer_args.max_epochs` is how many, so the defaults of 4 and 2 give 8 steps.

### Across multiple GPUs

*To follow.* Set `pl_trainer_args.devices: 8`; the launch command is unchanged.

### Across multiple nodes

*To follow.* Set `pl_trainer_args.num_nodes: N` and submit with `sbatch`. Submission scripts
will live in `slurm/`.

### Training on the full dataset

The subset above exists so the pipeline can be proven quickly. Real training swaps it for the
full PDB set and starts from one of the configs in `examples/training_yamls/`
(`initial_training.yml` through `finetune_3.yml`), which carry real schedules rather than a
two-epoch smoke test.

```bash
aws s3 sync s3://openfold3-data/pdb_training_set/ /shared/openfold3/pdb_training_set/ --no-sign-request
```

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

## Links

- OpenFold3: <https://github.com/aqlaboratory/openfold-3>
- Training guide: <https://openfold-3.readthedocs.io/en/latest/training.html>
- Configuration reference: <https://openfold-3.readthedocs.io/en/latest/configuration_reference.html>
- Datasets: <https://portal.openfold.omsf.io/datasets>

## License

Apache-2.0. See [LICENSE](LICENSE).

