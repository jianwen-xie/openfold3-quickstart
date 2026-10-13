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
<summary>How this image was made</summary>

Rendered with [3Dmol.js](https://3dmol.csb.pitt.edu/), the engine behind `py3Dmol`. The four
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
v.zoomTo(); v.rotate(60, "y"); v.zoom(1.3); v.render();
</script>
```

`cifText` here is the prediction after it has been superposed onto the reference (the
`pred_fitted` produced in the next section), so that this figure and the superposition below
look at the molecule from the same angle. Rendering the raw file instead gives the same
colours in a different orientation, which makes the two figures hard to compare.

py3Dmol is a JavaScript widget, so it renders in a running notebook but not on GitHub. The
static image above came from putting the above in an HTML page and screenshotting it:

```bash
google-chrome --headless --disable-gpu --screenshot=out.png \
    --window-size=1400,1100 --virtual-time-budget=25000 file:///tmp/view.html
```

`--virtual-time-budget` matters: without it the screenshot fires before WebGL has drawn
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

Sub-ångström over the folded core, and the gap between the two rows is the C-terminal tail,
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
print(f"RMSD {struc.rmsd(ref_ca, fitted):.2f} A")          # -> 0.73

pred_fitted = transform.apply(pred)                          # for rendering the overlay

# second row of the table: the folded core, excluding the flexible tail
core = pred_ca.res_id <= 72
core_fitted, _ = struc.superimpose(ref_ca[core], pred_ca[core])
print(f"RMSD 1-72 {struc.rmsd(ref_ca[core], core_fitted):.2f} A")   # -> 0.44
```

Get the reference with `curl -O https://files.rcsb.org/download/1UBQ.cif`. This works because
the residue numbering matches one to one; different constructs need a sequence alignment first.

</details>

**Reading the numbers.** `plddt` is per-atom confidence from 0 to 100: above 90 is very high,
70 to 90 confident, below 50 usually disordered. `ptm` is the predicted TM-score for the whole
structure; `iptm` is its interface counterpart and is only meaningful for complexes (it reads
`0.0` for a single chain). `pae` is the predicted aligned error in angstroms: entry (i, j) is
how far off residue *i* is expected to be if you superimpose on residue *j*, which is what tells
you whether two domains are confidently placed relative to one another.

Timing per stage is written to `timing.json` in the same directory.

#### (2) A protein with a ligand

Ubiquitin is a single chain, the easy case, and the one AlphaFold2 already handled. What
AlphaFold3-style models add is **co-folding**: predicting a structure and its partners
together. The simplest version of that is one protein plus one small molecule. The repo ships
an example that does this, and whose name is the PDB entry it reproduces:

```bash
pixi run -e openfold3-cuda12 run_openfold predict \
    --query-json examples/example_inference_inputs/query_single_protein_single_ligand.json \
    --use-msa-server true \
    --output-dir ./quickstart_cofold
```

The output layout, the five samples and the ranking step are all as described above. Only
what differs is shown here.

The query is T4 lysozyme L99A (164 residues) plus toluene, given as SMILES (`Cc1ccccc1`) rather
than a sequence. The reference is [7L39](https://www.rcsb.org/structure/7L39) at 1.35 Å, the
classic model system for ligand binding in a buried hydrophobic cavity.

Confidence is not the interesting question here: lysozyme folds correctly either way, and
96.3% of atoms score ≥ 90. What matters is whether the ligand found the right pocket in the
right pose, so the figure to look at is the superposition:

<p align="center">
  <img src="assets/lysozyme_toluene.png" alt="Predicted lysozyme-toluene complex superposed on 7L39" width="560">
</p>

Prediction in navy, crystal structure in grey.

**Left:** the whole complex superposed. **Right:** a close-up of the binding cavity, looking at
the toluene itself. The **blue ring is the predicted ligand** and the **black ring directly
beneath it is the crystallographic one**. They overlap almost exactly, which is the 0.33 Å in
the table below. The faint grey sticks are the side chains lining the pocket, and the protein
is drawn at 30% opacity because the cavity is buried and an opaque ribbon would hide the
ligand entirely.

| Superposition | Deviation |
|---|---|
| Protein Cα, 162 residues | 0.23 Å |
| Toluene centroid vs. crystallographic position | 0.33 Å |

The ligand found the right pocket in the right pose, to within a third of an ångström.

Overall the run scores `avg_plddt` 97.8, `ptm` 0.98 and `iptm` 0.94. That last number is
finally meaningful here. It read `0.0` for ubiquitin because a single chain has no interface;
here it scores the protein–ligand interface. For anything with more than one component, `iptm` is the
number to watch, not `ptm`.

<details>
<summary>How this image was made</summary>

Superpose on the protein, then carry the same transform to the ligand so its position is
compared in the reference's frame:

```python
import biotite.structure as struc, biotite.structure.io.pdbx as pdbx

pred = pdbx.get_structure(pdbx.CIFFile.read(
    "quickstart_cofold/pdb_7L39/seed_42/pdb_7L39_seed_42_sample_1_model.cif"), model=1)
ref = pdbx.get_structure(pdbx.CIFFile.read("7l39.cif"), model=1)

pred_ca = pred[struc.filter_amino_acids(pred) & (pred.atom_name == "CA")]
ref_ca = ref[struc.filter_amino_acids(ref) & ~ref.hetero & (ref.atom_name == "CA")]
n = min(pred_ca.array_length(), ref_ca.array_length())
fitted, transform = struc.superimpose(ref_ca[:n], pred_ca[:n])
print(f"protein RMSD {struc.rmsd(ref_ca[:n], fitted):.2f} A")

pred_fit = transform.apply(pred)                      # ligand moves with the protein
lig_pred = pred_fit[pred_fit.res_name == "LIG0"]      # predicted ligand is named LIG0
lig_ref = ref[(ref.res_name == "MBN") & (ref.element != "H")]   # MBN = toluene
import numpy as np
print(f"ligand centroid {np.linalg.norm(lig_pred.coord.mean(0) - lig_ref.coord.mean(0)):.2f} A")

# write both as mmCIF -- PDB format rejects the 4-character residue name LIG0
for name, arr in (("pred", pred_fit), ("ref", ref)):
    f = pdbx.CIFFile(); pdbx.set_structure(f, arr); f.write(f"{name}.cif")
```

The `[:n]` truncation is safe here only because both structures are numbered contiguously from
1 and the crystal simply stops two residues earlier, so the first 162 line up one to one. It
would silently misalign against a reference with internal gaps or a different numbering. The
complex example below shows how to match residues properly when that happens.

Then render as before, with two changes: the ligands get their own stick style, and the
close-up drops the cartoon to 35% opacity because the toluene is buried and an opaque ribbon
hides it.

```js
v.addModel(ref, "cif");
v.setStyle({model:0}, {cartoon:{color:"#dedede", thickness:0.28, opacity:0.35}});
v.setStyle({model:0, resn:"MBN"}, {stick:{radius:0.28, color:"#777777"}});
v.addModel(pred, "cif");
v.setStyle({model:1}, {cartoon:{color:"#27408B", thickness:0.28, opacity:0.35}});
v.setStyle({model:1, resn:"LIG0"}, {stick:{radius:0.28, color:"#1f6feb"}});
// cavity-lining side chains for context, then frame on the ligand
v.addStyle({model:0, within:{distance:6.0, sel:{resn:"MBN"}}, hetflag:false},
           {stick:{radius:0.10, color:"#999999"}});
v.zoomTo({model:0, resn:"MBN"}); v.zoom(0.62); v.render();
```

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

<p align="center">
  <img src="assets/psd_vs_7cnx.png" alt="Predicted assembly superposed on the 7CNX crystal structure" width="320">
</p>

Prediction in navy, crystal structure in grey, same orientation as above.

| Superposition | Cα RMSD |
|---|---|
| All four chains at once, 557 residues | 0.57 Å |
| β subunit `A` alone, 246 residues | 0.29 Å |
| β subunit `C` alone, 240 residues | 0.35 Å |
| α subunit `B` alone, 38 residues | 1.11 Å |
| α subunit `D` alone, 33 residues | 0.20 Å |

The first row is the one that matters. Superposing all four chains together scores the folds
*and* their arrangement at the same time, so 0.57 Å means the assembly is right, not merely
four chains that are each correct but wrongly placed. The navy helix on the left with no grey
beneath it is a stretch the crystal never resolved; the reference chain starts at residue 8.

**Reading `chain_pair_iptm`.** Overall the run scores `avg_plddt` 92.8, `ptm` 0.90, `iptm` 0.89
and no clashes. But the aggregated file also scores every *pair* of chains separately, and this
is where a complex tells you more than any single number can:

| | measured closest approach | `chain_pair_iptm` |
|---|---|---|
| `(A, B)` and `(C, D)`, α against its own β | 2.7 Å | 0.96 |
| `(A, C)`, the β–β interface | 2.6 Å | 0.85 |
| `(A, D)` and `(B, C)`, across the assembly | 11.4 Å | 0.78 |
| `(B, D)`, the two α subunits | 27.1 Å | 0.54 |

The low score is the interesting one, and it is not a warning. `(B, D)` reads 0.54 because
those two chains sit 27 Å apart and never touch. The metric is scoring an interface that does
not exist. A low `chain_pair_iptm` means *this pair is not confidently placed relative to each
other*, which for a non-contacting pair is both true and harmless. Check it against the
geometry before treating it as a failure.

<details>
<summary>How these images were made</summary>

The prediction and the crystal number their chains differently. The α subunit is 1–42 in the
prediction and 255–292 in the reference, so matching on residue ID alone silently drops both
α chains and leaves you scoring only the β subunits. Find the offset per chain rather than
assuming one:

```python
import numpy as np, biotite.structure as struc, biotite.structure.io.pdbx as pdbx

def ca(path):
    a = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    a = a[struc.filter_amino_acids(a) & ~a.hetero]
    return a[a.atom_name == "CA"]

pred, ref = ca("..._sample_1_model.cif"), ca("7cnx.cif")

offsets = {}
for ch in "ABCD":
    p, r = pred[pred.chain_id == ch], ref[ref.chain_id == ch]
    pm = dict(zip(p.res_id.tolist(), p.res_name))
    rm = dict(zip(r.res_id.tolist(), r.res_name))
    offsets[ch] = max(range(-5, 300),
                      key=lambda o: sum(pm[i] == rm[i + o] for i in pm if i + o in rm))
# -> {"A": 0, "B": 253, "C": 0, "D": 253}, each matching at 100% residue identity
```

7CNX holds two copies of the tetramer in its asymmetric unit (chains `A`–`H`); `A`–`D` are the
copy that corresponds. Superpose on all four chains at once, then apply that one transform to
every atom so both figures end up in the same frame:

```python
fitted, transform = struc.superimpose(ref_coords, pred_coords)
pred_fit = transform.apply(pred_full)
```

Render as before. The chain figure needs no colour function, just one colour per selection:

```js
v.setStyle({chain: "A"}, {cartoon: {color: "#27408B", thickness: 0.30}});
v.setStyle({chain: "C"}, {cartoon: {color: "#7FB2E5", thickness: 0.30}});
v.setStyle({chain: "B"}, {cartoon: {color: "#C2461E", thickness: 0.42}});
v.setStyle({chain: "D"}, {cartoon: {color: "#F0A33C", thickness: 0.42}});
```

The α subunits are drawn thicker so a 42-residue chain stays visible next to a 253-residue one.

</details>

### Across multiple GPUs

*To follow.* OpenFold3 distributes a multi-query job across GPUs, parallel across queries
rather than within a single structure. See the upstream
[inference docs](https://openfold-3.readthedocs.io/en/latest/inference.html#inference-run-on-multiple-gpus)
in the meantime.

## Training

### On one GPU

Pick a handful of structures and download only those:

```bash
pixi run -e openfold3-cuda12 setup-pdb-subset
```

This writes a config to `datasets/train_pdb_subset.yaml`. **Use that one**. The committed
`scripts/datasets/train_pdb_subset.yaml` is a template still holding another user's absolute
paths.

<details>
<summary>What it downloads, including 1.6 GB you can delete afterwards</summary>

| Written to `datasets/` | Size | What it is |
|---|---|---|
| `training_cache_with_templates.json` | 1.6 GB | A catalogue of the whole PDB training set: which entries exist, their chains, their templates. No coordinates. |
| `validation_cache_with_templates.json` | 35 MB | The same, for validation |
| `pdb_training_set/` | 73 MB | The real data for the 12 chosen entries, 435 files |
| `*_subset_8.json`, `*_subset_4.json` | 35 MB | The catalogue pruned to those 12. Training reads these |

The catalogue exists only so the script can pick 12 entries out of the full set. Once the
pruned caches are written it is never read again: the generated config points at
`..._subset_8.json` and `..._subset_4.json` and never at the full file, so deleting it is safe.

</details>

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

Then train. `WANDB_MODE=offline` is required: the YAML flag alone does not prevent a login
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

Two traps in that layout. Enabling logging **moves the checkpoint** out of
`quickstart_run/checkpoints/`, so hardcoded paths break. And `logs/` is always empty; the
metrics live only in the binary `.wandb` file.

### Monitoring the training curve

The run writes its metrics to a local wandb file, not to `logs/`. Convert them to a CSV:

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

The notebook runs the same extraction and prints the result as a table. For a quick look from
a shell:

```bash
column -s, -t quickstart_run/metrics.csv
```

Eight steps is too few to plot, so the figure below is the same pipeline run longer:
`epoch_len: 25` and `max_epochs: 6`, which is 150 steps and about 25 minutes on one H100.

<p align="center">
  <img src="assets/training_curve.png" alt="Training loss falling while validation lDDT stays flat" width="820">
</p>

| | start | end |
|---|---:|---:|
| `train/loss` | 6.74 | 4.96 |
| `val/lddt_intra_protein` | 0.052 | 0.061 |
| `val/rmsd` | 20.96 Å | 20.30 Å |

**The loss falls and it means nothing.** It drops by a quarter while accuracy on the four
held-out structures does not move: lDDT stays near 0.05 where 1.0 is perfect, and RMSD sits
above 20 Å. The model is fitting eight structures, not learning to fold. Read this curve as a
pipeline check, not as progress.

<details>
<summary>Why it cannot do better, and how the figure was made</summary>

**Training starts from random weights.** `restart_checkpoint_path: last` finds no checkpoint on
a first run, and Lightning logs `no last checkpoint available. No checkpoint will be loaded`.
It does not fall back to the released parameters that inference uses, so this is a fresh model,
which is why its predictions sit ~20 Å from the answer. The quickstart trains a new model
badly; it does not fine-tune a good one. That, and eight structures being far too few to
generalise from, is the whole story. For a curve that means something, see
[Training on the full dataset](#training-on-the-full-dataset).

The spikes to 12 to 19 are single steps rather than instability: batch size is one structure,
and a large or awkward crop costs much more loss than a small one.

To plot, keep the extraction above as a list of records instead of writing the CSV. Its
`if 'train/loss' in row` filter drops every validation record: this run logs 18 of them, and
not one carries that key.

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def series(key):                      # rows: every history record, in order
    pts = sorted((r["_step"], r[key]) for r in rows
                 if key in r and "_step" in r and isinstance(r[key], (int, float)))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

ts, tv = series("train/loss")                   # 150 points, one per step
vs, vv = series("val/lddt_intra_protein")       # 6 points, one per epoch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1), dpi=150)
ax1.plot(ts, tv, color="#2a78d6", linewidth=1.0, alpha=0.28)
w = 10                                          # 10-step running mean
ax1.plot(ts[w - 1:], np.convolve(tv, np.ones(w) / w, mode="valid"),
         color="#2a78d6", linewidth=2.0)
ax2.plot(vs, vv, color="#eb6834", linewidth=2.0, marker="o", markersize=8,
         markeredgecolor="#fcfcfb", markeredgewidth=2)
ax1.set_ylim(0, 20)
ax2.set_ylim(0, 1)                              # full lDDT range, not zoomed
ax2.set_xlim(ax1.get_xlim())                    # same step axis on both panels
```

Two panels rather than twin y-axes: a loss of ~5 and an lDDT of ~0.05 share no scale, and
overlaying them lets the crossing point be moved anywhere by choosing the limits. The lDDT
panel is drawn over its full 0 to 1 range for the same reason; zoomed to 0.05 to 0.06 those
six points look like a healthy climb.

</details>

#### Watching it live instead

Offline logging only yields a curve once the run finishes. For a live dashboard, which is what
you want for long runs and for the multi-GPU and Slurm stages, use wandb online:

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

The default is deliberately short: two epochs of four steps, enough to prove the pipeline.
Eight points is not a curve. To get a trace worth monitoring, raise these in the config:

| Setting | Default here | For a longer run |
|---|---|---|
| `data_module_args.epoch_len` | `4` | `50`+, steps per epoch |
| `pl_trainer_args.max_epochs` | `2` | `10`+ |
| `--train-size` on `generate-pdb-subset` | `8` | more structures to sample from |

For more structures, call the two underlying tasks rather than `setup-pdb-subset`, passing the
same `--train-size` to **both**:

```bash
pixi run -e openfold3-cuda12 generate-pdb-subset --train-size 32
pixi run -e openfold3-cuda12 download-pdb-subset --train-size 32
```

At roughly 8 to 10 s per step on one H100, 150 steps is about 20 minutes plus validation
between epochs.

<details>
<summary>Why the size goes on both commands, and not on <code>setup-pdb-subset</code></summary>

Each script resolves the cache from the size by filename,
`training_cache_with_templates_subset_{train_size}.json`, and both default to `8`. Give the
size only to `generate-pdb-subset` and the download silently falls back to 8, finds the
existing subset-8 cache and fetches those files instead, with no error at setup. Training then
fails later on missing structures, because the generated config points at the subset-32 cache
that was never downloaded.

`setup-pdb-subset` cannot take the flag at all. It declares only `depends-on` and has no
command of its own, so it has nothing to forward arguments to and drops them without warning.

The validation split is fixed at 4: `download_subset.py` takes a `--val-size`, but
`generate_subset_cache.py` has no such argument, so there is no way to generate a different
one. A comment in the repo's `pixi.toml` advertises `--val-size` on the generate side; passing
it there fails.

</details>

### Training on the full dataset

The subset above exists so the pipeline can be proven quickly. Real training swaps it for the
full PDB set and starts from one of the configs in `examples/training_yamls/`
(`initial_training.yml` through `finetune_3.yml`), which carry real schedules rather than a
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
| Training finished but `logs/` is empty and there are no metrics | `LoggingConfig` takes Weights & Biases or nothing, and defaults to `wandb_config: null`, so every loss is computed and discarded. Add the `logging_config` block from step 3. |
| Scripts can't find the checkpoint | Enabling logging moves it from `<output_dir>/checkpoints/` to `<output_dir>/<project>/<run id>/checkpoints/`. |
| Inference hangs or fails at the MSA stage | `--use-msa-server true` calls the public ColabFold server; check outbound network and their rate limits. |
| `setup-pdb-subset` reports `1 failed` with a 404 on a template `.npz` | Expected. One validation template, `7kud_A.npz`, is absent from the bucket, so 387 of 388 files download and the run proceeds normally. A 404 here is a missing object, not a network fault. |
| Flags passed to `setup-pdb-subset` appear to do nothing | It is a `depends-on` aggregate with no command of its own, so arguments are dropped without warning. Pass them to `generate-pdb-subset` and `download-pdb-subset` directly instead. |
| Training fails on missing structures after changing `--train-size` | The size has to go to `download-pdb-subset` as well. On its own it defaults to 8, downloads the subset-8 files without error, and leaves the config pointing at a cache whose structures were never fetched. |

## Links

- OpenFold3: <https://github.com/aqlaboratory/openfold-3>
- Training guide: <https://openfold-3.readthedocs.io/en/latest/training.html>
- Configuration reference: <https://openfold-3.readthedocs.io/en/latest/configuration_reference.html>
- Datasets: <https://portal.openfold.omsf.io/datasets>

## License

Apache-2.0. See [LICENSE](LICENSE).

