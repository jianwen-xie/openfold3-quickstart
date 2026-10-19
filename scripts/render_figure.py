#!/usr/bin/env python3
"""Render an OpenFold3 prediction to a README-ready PNG.

Three modes:

  plddt   colour by the four AlphaFold confidence bands (pLDDT is in the B-factor column)
  chains  one colour per chain, for showing what is in a complex
  super   a prediction superposed on a reference structure

Needs biotite (in the pixi environment), headless Chrome and ImageMagick's `convert`.
Chrome and ImageMagick are not part of the pixi environment; install them separately.

Examples, reproducing the figures in the README:

  render_figure.py plddt  quickstart_inference/ubiquitin/seed_42/*_sample_1_model.cif \\
      assets/ubiquitin_cartoon.png --fit-to 1ubq.cif --rotate 60 --width 620

  render_figure.py chains quickstart_multimer/7cnx/seed_42/*_sample_1_model.cif \\
      assets/psd_chains.png --rotate 20 --tilt 10 --zoom 1.10 --width 640

  render_figure.py super  demo_mhc/.../*_sample_1_model.cif 1duz.cif \\
      assets/demo_mhc_vs_1duz.png --rotate 70 --sticks C --ref-sticks F --width 420
"""
import argparse, json, pathlib, shutil, subprocess, sys

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

BANDS = """function band(a){
  if (a.b >= 90) return "#0053D6";   // very high
  if (a.b >= 70) return "#65CBF3";   // confident
  if (a.b >= 50) return "#FFDB13";   // low
  return "#FF7D45";                  // very low
}"""

# categorical palette, validated for colourblind separation
CHAIN_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script></head>
<body style="margin:0"><div id="v" style="width:%(w)dpx;height:%(h)dpx;position:relative"></div>
<script>
%(bands)s
const MODELS = %(models)s;
const v = $3Dmol.createViewer("v", {backgroundColor: "%(bg)s"});
MODELS.forEach(m => v.addModel(m, "cif"));
%(style)s
v.zoomTo(); v.rotate(%(rot)d, "y"); v.rotate(%(tilt)d, "x"); v.zoom(%(zoom)s); v.render();
</script></body></html>"""


def load(path, b_factor=False):
    extra = ["b_factor"] if b_factor else []
    return pdbx.get_structure(pdbx.CIFFile.read(path), model=1, extra_fields=extra)


def _offset_by_identity(pred_names, ref_names, lo=-60, hi=300, floor=5):
    """Pick the residue-id offset that maximises IDENTICAL residue names.

    Choosing the offset that merely maximises how many ids overlap silently
    misaligns chains whose numbering differs, and still returns a plausible RMSD.
    """
    best = (-1, 0)
    for o in range(lo, hi):
        ids = [i for i in pred_names if i + o in ref_names]
        if len(ids) < floor:
            continue
        agree = sum(pred_names[i] == ref_names[i + o] for i in ids)
        if agree > best[0]:
            best = (agree, o)
    return best[1]


def matched_coords(pred, ref, chain_map=None):
    """Pair up atoms between prediction and reference, chain by chain.

    Protein chains are matched on residue identity. Nucleic chains are matched by
    order along the strand instead: a crystal often numbers the second strand from
    51 while the prediction numbers it from 1, and matching on id would drop it.
    """
    P, R = [], []
    pp = pred[struc.filter_amino_acids(pred) & ~pred.hetero]
    rp = ref[struc.filter_amino_acids(ref) & ~ref.hetero]
    for ch in sorted(set(pp.chain_id)):
        rch = (chain_map or {}).get(ch, ch)
        if rch not in set(rp.chain_id):
            continue
        p = pp[(pp.chain_id == ch) & (pp.atom_name == "CA")]
        r = rp[(rp.chain_id == rch) & (rp.atom_name == "CA")]
        pn = dict(zip(p.res_id.tolist(), p.res_name))
        rn = dict(zip(r.res_id.tolist(), r.res_name))
        pc = dict(zip(p.res_id.tolist(), p.coord))
        rc = dict(zip(r.res_id.tolist(), r.coord))
        o = _offset_by_identity(pn, rn)
        for i in sorted(pn):
            if i + o in rn and pn[i] == rn[i + o]:
                P.append(pc[i]); R.append(rc[i + o])
    pnuc = pred[struc.filter_nucleotides(pred)]
    rnuc = ref[struc.filter_nucleotides(ref)]
    for ch in sorted(set(pnuc.chain_id)):
        rch = (chain_map or {}).get(ch, ch)
        if rch not in set(rnuc.chain_id):
            continue
        a = pnuc[(pnuc.chain_id == ch) & (pnuc.atom_name == "C1'")]
        b = rnuc[(rnuc.chain_id == rch) & (rnuc.atom_name == "C1'")]
        a = a.coord[np.argsort(a.res_id)]
        b = b.coord[np.argsort(b.res_id)]
        n = min(len(a), len(b))
        P += list(a[:n]); R += list(b[:n])
    if not P:
        sys.exit("error: no atoms could be matched between the two structures")
    return np.array(P), np.array(R)


def superpose(pred, ref, chain_map=None):
    P, R = matched_coords(pred, ref, chain_map)
    fitted, transform = struc.superimpose(R, P)
    return transform.apply(pred), float(struc.rmsd(R, fitted)), len(P)


def _score(pred, ref, chain_map, ranges) -> None:
    """Report RMSD over everything matched, then over each requested residue range."""
    P, R = matched_coords(pred, ref, chain_map)
    fitted, _ = struc.superimpose(R, P)
    print(f"  all matched atoms      {struc.rmsd(R, fitted):5.2f} A  over {len(P)}")
    if not ranges:
        return
    # the ranges are re-matched from scratch so each is superposed on itself,
    # which is what a per-region RMSD means
    pca = pred[struc.filter_amino_acids(pred) & ~pred.hetero]
    pca = pca[pca.atom_name == "CA"]
    for lo, hi in ranges:
        sel = pred[np.isin(pred.res_id, np.arange(lo, hi + 1))]
        Ps, Rs = matched_coords(sel, ref, chain_map)
        f, _ = struc.superimpose(Rs, Ps)
        print(f"  residues {lo}-{hi:<13} {struc.rmsd(Rs, f):5.2f} A  over {len(Ps)}")


def to_cif_text(arr, tmp, name):
    path = tmp / f"{name}.cif"
    f = pdbx.CIFFile(); pdbx.set_structure(f, arr); f.write(str(path))
    return path.read_text()


def style_plddt():
    return ('v.setStyle({}, {cartoon:{colorfunc: band, thickness:0.35}});\n'
            'v.addStyle({not:{atom:["N","C","O"]}}, {stick:{radius:0.12, colorfunc: band}});')


def style_chains(chains, sticks, translucent, spheres, override=None):
    out = []
    override = override or {}
    op = ", opacity:0.55" if translucent else ""
    for i, ch in enumerate(chains):
        col = override.get(ch, CHAIN_COLOURS[i % len(CHAIN_COLOURS)])
        thick = 0.50 if ch in sticks else 0.32
        out.append(f'v.setStyle({{chain:"{ch}"}}, {{cartoon:{{color:"{col}", thickness:{thick}{op}}}}});')
        if ch in sticks:
            out.append(f'v.addStyle({{chain:"{ch}"}}, {{stick:{{radius:0.18, color:"{col}"}}}});')
    for j, resn in enumerate(spheres):
        col = override.get(resn, CHAIN_COLOURS[(len(chains) + j) % len(CHAIN_COLOURS)])
        out.append(f'v.addStyle({{resn:"{resn}"}}, {{sphere:{{radius:1.1, color:"{col}"}}}});')
    return "\n".join(out)


def style_focus(pred_resn: str, ref_resn: str, radius: float) -> str:
    """Close up on a bound ligand: translucent protein, the pocket's side chains, both copies
    of the ligand. The cartoon stays visible rather than washing out to white, which is what
    happens below roughly 50% opacity on a pale colour."""
    return "\n".join([
        f'v.setStyle({{model:0}}, {{cartoon:{{color:"#b9bec9", thickness:0.26, opacity:0.68}}}});',
        f'v.setStyle({{model:1}}, {{cartoon:{{color:"#27408B", thickness:0.26, opacity:0.68}}}});',
        f'v.setStyle({{model:0, resn:"{ref_resn}"}}, {{stick:{{radius:0.30, color:"#1a1a1a"}}}});',
        f'v.setStyle({{model:1, resn:"{pred_resn}"}}, {{stick:{{radius:0.30, color:"#2a78d6"}}}});',
        # the residues lining the pocket, for context
        f'v.addStyle({{model:0, within:{{distance:{radius}, sel:{{resn:"{ref_resn}"}}}}, hetflag:false}},'
        f' {{stick:{{radius:0.10, color:"#8b8b8b"}}}});',
    ])


def nonpolymer(arr) -> list[str]:
    """Residue names a cartoon cannot draw: ligands, ions, anything not a chain backbone.
    Left to a cartoon alone they are simply absent from the figure."""
    other = arr[~struc.filter_amino_acids(arr) & ~struc.filter_nucleotides(arr)]
    return sorted({r for r in other.res_name.tolist() if r != "HOH"})


def style_super(sticks, ref_sticks, spheres, ligands=(), ref_ligands=(),
                translucent=False):
    op = ", opacity:0.82" if translucent else ""
    out = [
        f'v.setStyle({{model:0}}, {{cartoon:{{color:"#d4d4d4", thickness:0.28{op}}}}});',
        f'v.setStyle({{model:1}}, {{cartoon:{{color:"#27408B", thickness:0.28{op}}}}});',
    ]
    # ligands are drawn as sticks because a cartoon omits them entirely
    for resn in ref_ligands:
        out.append(f'v.setStyle({{model:0, resn:"{resn}"}}, '
                   f'{{stick:{{radius:0.26, color:"#333333"}}}});')
    for resn in ligands:
        out.append(f'v.setStyle({{model:1, resn:"{resn}"}}, '
                   f'{{stick:{{radius:0.26, color:"#2a78d6"}}}});')
    # A short chain has no secondary structure for a cartoon to trace and draws as
    # nothing; the crystal copy goes on thick so the prediction reads on top of it.
    for ch in ref_sticks:
        out.append(f'v.setStyle({{model:0, chain:"{ch}"}}, {{stick:{{radius:0.46, color:"#333333"}}}});')
    for ch in sticks:
        out.append(f'v.setStyle({{model:1, chain:"{ch}"}}, {{stick:{{radius:0.24, color:"#5aa9f5"}}}});')
    # An opaque crystal sphere hides a predicted ion sitting a fraction of an angstrom inside it.
    for resn in spheres:
        out.append(f'v.addStyle({{model:0, resn:"{resn}"}}, '
                   f'{{sphere:{{radius:1.45, color:"#444444", opacity:0.40}}}});')
        out.append(f'v.addStyle({{model:1, resn:"{resn}"}}, '
                   f'{{sphere:{{radius:0.80, color:"#5aa9f5"}}}});')
    return "\n".join(out)


# These ship under different binary names depending on the platform: Chromium on most
# Linux distributions, and ImageMagick 7 replaced `convert` with `magick`.
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
MAGICK_NAMES = ("magick", "convert")


def _find(names: tuple[str, ...], label: str, hint: str) -> str:
    for name in names:
        if shutil.which(name):
            return name
    sys.exit(f"error: {label} not found on PATH (looked for: {', '.join(names)}).\n"
             f"       {hint}")


def render(html: str, out_png: str, canvas: tuple[int, int],
           width: int, border: int) -> None:
    chrome = _find(CHROME_NAMES, "Chrome or Chromium",
                   "Ubuntu: sudo apt install chromium-browser  |  macOS: brew install --cask google-chrome")
    magick = _find(MAGICK_NAMES, "ImageMagick",
                   "Ubuntu: sudo apt install imagemagick       |  macOS: brew install imagemagick")
    out = pathlib.Path(out_png)
    page = out.with_suffix(".html")
    page.write_text(html)
    subprocess.run([chrome, "--headless", "--disable-gpu",
                    f"--screenshot={out}", f"--window-size={canvas[0]},{canvas[1]}",
                    "--virtual-time-budget=30000", f"file://{page.absolute()}"],
                   check=True, capture_output=True)
    # Raw renders are ~430 KB with large margins. Trim, add a small margin back so
    # nothing touches the edge, scale to the width used in the README, and reduce
    # to a 256-colour palette.
    subprocess.run([magick, str(out), "-trim", "+repage",
                    "-bordercolor", "white", "-border", str(border),
                    "-resize", f"{width}x", "-colors", "256", "-strip", str(out)],
                   check=True)
    page.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plddt", "chains", "super", "score"])
    ap.add_argument("model", help="predicted mmCIF")
    ap.add_argument("rest", nargs="+", help="[reference.cif for super] out.png")
    ap.add_argument("--fit-to", help="superpose on this reference before rendering, "
                                     "so several figures share an orientation")
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--tilt", type=int, default=0, help="rotation about x, after y")
    ap.add_argument("--zoom", default="1.0")
    ap.add_argument("--width", type=int, default=620, help="output width in pixels")
    ap.add_argument("--border", type=int, default=12)
    ap.add_argument("--canvas", default="1300x1100")
    ap.add_argument("--sticks", action="append", default=[],
                    help="draw this chain as sticks; needed for chains too short to "
                         "draw as a cartoon, which otherwise render as nothing")
    ap.add_argument("--ref-sticks", action="append", default=[],
                    help="super mode: the reference's copy of a --sticks chain")
    ap.add_argument("--spheres", action="append", default=[],
                    help="draw this residue name as spheres, e.g. ZN")
    ap.add_argument("--translucent", action="store_true",
                    help="drop the cartoon's opacity so a buried ligand shows through it")
    ap.add_argument("--colour", action="append", default=[], metavar="KEY:HEX",
                    help="pin a chain id or residue name to a colour, e.g. --colour C:#f6b08c. "
                         "Repeatable. Use it to give two strands of one duplex the same hue so "
                         "they read as a pair rather than as unrelated chains.")
    ap.add_argument("--ref-ligand", metavar="RESN",
                    help="super mode: the reference's name for the ligand, when it differs from "
                         "the prediction's. A crystal often calls it something else, and a "
                         "cartoon draws neither, so without this only the predicted one appears.")
    ap.add_argument("--focus", metavar="RESN",
                    help="super mode: close up on this residue name, e.g. LIG0 for a predicted "
                         "ligand. Draws the protein translucent and the pocket's side chains, "
                         "since an opaque ribbon hides a buried ligand entirely.")
    ap.add_argument("--focus-ref", metavar="RESN",
                    help="the reference's name for the --focus residue, if it differs")
    ap.add_argument("--focus-radius", type=float, default=6.0,
                    help="how far around --focus to draw side chains (default 6 A)")
    ap.add_argument("--residues", metavar="LO-HI",
                    help="score mode: restrict to this residue range, e.g. 1-72 to exclude a "
                         "flexible tail. Repeatable for several ranges.", action="append",
                    default=[])
    ap.add_argument("--ref-chains",
                    help="super mode: use only these reference chains, comma separated, "
                         "mapped onto the prediction's chains in order. A crystal often "
                         "holds several copies of the same complex; without this the extra "
                         "copies are drawn too, unaligned, and the RMSD is against whichever "
                         "copy happens to share chain ids. Example: --ref-chains D,E,F")
    a = ap.parse_args()

    if a.mode == "score":
        if len(a.rest) != 1:
            ap.error("score mode takes: model.cif reference.cif")
        ref_path, out_png = a.rest[0], None
    elif a.mode == "super":
        if len(a.rest) != 2:
            ap.error("super mode takes: model.cif reference.cif out.png")
        ref_path, out_png = a.rest
    else:
        if len(a.rest) != 1:
            ap.error(f"{a.mode} mode takes: model.cif out.png")
        ref_path, out_png = None, a.rest[0]

    pred = load(a.model, b_factor=(a.mode == "plddt"))

    if a.mode == "score":
        ref = load(ref_path)
        ref = ref[ref.res_name != "HOH"]
        cmap = None
        if a.ref_chains:
            want = [c.strip() for c in a.ref_chains.split(",")]
            ref = ref[np.isin(ref.chain_id, want)]
            cmap = dict(zip(sorted(set(pred.chain_id)), want))
        ranges = []
        for spec in a.residues:
            lo, _, hi = spec.partition("-")
            ranges.append((int(lo), int(hi)))
        _score(pred, ref, cmap, ranges)
        return

    tmp = pathlib.Path(out_png).parent

    if a.mode == "super":
        ref = load(ref_path)
        ref = ref[ref.res_name != "HOH"]
        cmap = None
        if a.ref_chains:
            want = [c.strip() for c in a.ref_chains.split(",")]
            ref = ref[np.isin(ref.chain_id, want)]
            cmap = dict(zip(sorted(set(pred.chain_id)), want))
        pred, rmsd, n = superpose(pred, ref, cmap)
        print(f"  superposed: {rmsd:.2f} A over {n} matched atoms")
        models = [to_cif_text(ref, tmp, "_ref"), to_cif_text(pred, tmp, "_pred")]
        if a.focus:
            style = style_focus(a.focus, a.focus_ref or a.focus, a.focus_radius)
        else:
            # anything a cartoon cannot draw, drawn as sticks so it is not silently missing
            ligs = nonpolymer(pred)
            ref_ligs = [a.ref_ligand] if a.ref_ligand else []
            if ligs or ref_ligs:
                print(f"  drawing as sticks: prediction {ligs or '-'}, reference {ref_ligs or '-'}")
            style = style_super(a.sticks, a.ref_sticks, a.spheres, ligs, ref_ligs,
                                a.translucent)
    else:
        if a.fit_to:
            ref = load(a.fit_to)
            ref = ref[ref.res_name != "HOH"]
            pred, rmsd, n = superpose(pred, ref)
            print(f"  fitted to reference: {rmsd:.2f} A over {n} matched atoms")
        models = [to_cif_text(pred, tmp, "_pred")]
        override = {}
        for spec in a.colour:
            if ":" not in spec:
                ap.error(f"--colour expects KEY:HEX, got {spec!r}")
            k, v = spec.split(":", 1)
            override[k.strip()] = v.strip()
        style = (style_plddt() if a.mode == "plddt"
                 else style_chains(sorted(set(pred.chain_id)), a.sticks,
                                   a.translucent, a.spheres, override))

    w, h = (int(x) for x in a.canvas.lower().split("x"))
    page = PAGE
    if a.mode == "super" and a.focus:
        page = PAGE.replace('v.zoomTo();',
                            'v.zoomTo({model:0, resn:"%s"});' % (a.focus_ref or a.focus))
    html = page % dict(w=w, h=h, bands=BANDS if a.mode == "plddt" else "",
                       models=json.dumps(models), bg="white", style=style,
                       rot=a.rotate, tilt=a.tilt, zoom=a.zoom)
    render(html, out_png, (w, h), a.width, a.border)
    for leftover in ("_ref.cif", "_pred.cif"):
        (tmp / leftover).unlink(missing_ok=True)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
