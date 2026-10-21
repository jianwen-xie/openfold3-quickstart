#!/usr/bin/env python3
"""Read training metrics out of a wandb run and print or export them.

Works for runs logged online (``run-*`` directories) and offline
(``offline-run-*``), so the same command serves both paths in the quickstart.

    python read_metrics.py quickstart_run                 # per-step table
    python read_metrics.py quickstart_run --summary       # first -> last
    python read_metrics.py quickstart_run --csv out.csv   # export
    python read_metrics.py quickstart_run --list-keys     # what was logged

Needs the wandb package, so run it inside the project environment:

    pixi run -e openfold3-cuda12 python read_metrics.py quickstart_run
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

DEFAULT_COLUMNS = (
    "step",
    "train/loss",
    "train/distogram_loss",
    "train/diffusion_loss",
)

SUMMARY_KEYS = (
    "train/loss",
    "val/lddt_intra_protein",
    "val/rmsd",
)

PLOT_KEYS = ("train/loss", "val/lddt_intra_protein")

SMOOTH_WINDOW = 10          # running mean, applied once a series is long enough
SMOOTH_MIN_POINTS = 20

# Where a .wandb file hides, relative to whatever the user points us at.
SEARCH_PATTERNS = (
    "run-*.wandb",
    os.path.join("wandb", "*run-*", "run-*.wandb"),
    os.path.join("*run-*", "run-*.wandb"),
)


class MetricsError(Exception):
    """Anything the user can fix by changing the command line."""


def find_run_file(path: str) -> str:
    """Return the newest .wandb file at or under *path*."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise MetricsError(f"no such file or directory: {path}")

    found = []
    for pattern in SEARCH_PATTERNS:
        found.extend(glob.glob(os.path.join(path, pattern)))
    if not found:
        searched = "\n".join(f"    {os.path.join(path, p)}" for p in SEARCH_PATTERNS)
        raise MetricsError(
            f"no wandb run file under {path}. Searched:\n{searched}\n"
            "Point at the output_dir from your config, a wandb run directory, "
            "or a .wandb file directly."
        )
    return max(found, key=os.path.getmtime)


def read_history(run_file: str) -> list[dict]:
    """Decode the binary run file into one dict per history record."""
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal import datastore
    except ImportError as exc:  # pragma: no cover - environment problem
        raise MetricsError(
            f"cannot import wandb ({exc}). Run this inside the project "
            "environment, e.g. pixi run -e openfold3-cuda12 python ..."
        ) from exc

    store = datastore.DataStore()
    try:
        store.open_for_scan(run_file)
    except Exception as exc:
        raise MetricsError(f"cannot read {run_file}: {exc}") from exc

    rows = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        # Every record is kept. Filtering on a training key here would drop
        # each validation record, which carries no train/* key at all.
        rows.append(
            {
                item.key or ".".join(item.nested_key): json.loads(item.value_json)
                for item in record.history.item
            }
        )
    return rows


def with_training_step(rows: list[dict]) -> list[dict]:
    """Return new rows carrying a derived `step`, the real training step.

    wandb's `_step` counts every logged record, so validation and epoch
    aggregates consume numbers that are not training steps. The nth record
    carrying train/loss is training step n. New dicts; the input is untouched,
    and the .wandb file is never written to.
    """
    step = 0
    out = []
    for row in sorted(rows, key=lambda r: r.get("_step", -1)):
        if "train/loss" in row:
            out.append({**row, "step": step})
            step += 1
        else:
            out.append(dict(row))
    return out


def series(rows: list[dict], key: str) -> list[tuple]:
    """Return (step, value) pairs for *key*, ordered by step."""
    return sorted(
        (row["_step"], row[key])
        for row in rows
        if key in row and "_step" in row and isinstance(row[key], (int, float))
    )


def numeric_keys(rows: list[dict]) -> list[str]:
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "_step"
        }
    )


def print_summary(rows: list[dict], keys: tuple) -> None:
    for key in keys:
        points = series(rows, key)
        if not points:
            print(f"  {key:24} not logged")
            continue
        first, last = points[0][1], points[-1][1]
        print(f"  {key:24} {first:9.4f} -> {last:9.4f}   ({len(points)} points)")


def print_table(rows: list[dict], columns: tuple) -> None:
    present = [c for c in columns if any(c in row for row in rows)]
    if not present:
        raise MetricsError(
            f"none of these were logged: {', '.join(columns)}. "
            "Use --list-keys to see what this run has."
        )
    steps = [row for row in rows if any(c in row for c in present if c != "_step")]
    steps.sort(key=lambda r: r.get("_step", -1))

    print("  ".join(f"{c:>22}" for c in present))
    for row in steps:
        cells = []
        for column in present:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:>22.4f}")
            elif value is None:
                cells.append(f"{'':>22}")
            else:
                cells.append(f"{value:>22}")
        print("  ".join(cells))


def write_plot(rows: list[dict], path: str, keys: tuple = PLOT_KEYS) -> None:
    """Draw one panel per key and save a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise MetricsError(
            f"plotting needs matplotlib and numpy ({exc}). Run inside the "
            "project environment, or use --csv and plot elsewhere."
        ) from exc

    panels = [(k, series(rows, k)) for k in keys]
    panels = [(k, pts) for k, pts in panels if pts]
    if not panels:
        raise MetricsError(
            f"none of these were logged: {', '.join(keys)}. "
            "Use --list-keys to see what this run has, then --keys to pick."
        )

    # wandb's _step counts every logged record, validation included, so it is not
    # a training step. The nth train record IS training step n, and validation runs
    # once every len(train)/len(val) steps, so both can be put on a real step axis.
    n_train = len(series(rows, "train/loss"))

    def step_axis(key: str, pts: list) -> "np.ndarray":
        if not key.startswith("val/"):
            return np.arange(len(pts))
        if n_train and len(pts):
            every = n_train / len(pts)
            return np.array([(i + 1) * every for i in range(len(pts))])
        return np.array([p[0] for p in pts])      # fall back to _step

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.1), dpi=150)
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, pts) in zip(axes, panels):
        steps = step_axis(key, pts)
        values = np.array([p[1] for p in pts])
        if len(values) >= SMOOTH_MIN_POINTS:
            # raw trace faint, running mean on top, so noise does not hide the trend
            ax.plot(steps, values, color="#2a78d6", linewidth=1.0, alpha=0.28)
            window = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            ax.plot(
                steps[SMOOTH_WINDOW - 1:],
                np.convolve(values, window, mode="valid"),
                color="#2a78d6",
                linewidth=2.0,
            )
        else:
            ax.plot(steps, values, color="#2a78d6", linewidth=2.0, marker="o",
                    markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        # the cadence is what explains the differing point counts, so say it
        cadence = "once per epoch" if key.startswith("val/") else "every training step"
        ax.set_title(f"logged {cadence}, {len(values)} points", fontsize=10)
        ax.set_ylabel(key)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        if key.endswith("lddt_intra_protein"):
            ax.set_ylim(0, 1)      # full lDDT range; zoomed, noise looks like progress

    # one axis across panels, so validation points line up with the training
    # steps they were measured at
    all_steps = [x for key, pts in panels for x in step_axis(key, pts)]
    for ax in axes:
        ax.set_xlim(min(all_steps), max(all_steps))

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(rows: list[dict], columns: tuple, path: str) -> int:
    present = [c for c in columns if any(c in row for row in rows)]
    steps = [row for row in rows if any(c in row for c in present if c != "_step")]
    steps.sort(key=lambda r: r.get("_step", -1))

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=present, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(steps)
    return len(steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read training metrics out of a wandb run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run",
        nargs="?",
        default="quickstart_run",
        help="output_dir, a wandb run directory, or a .wandb file "
        "(default: quickstart_run)",
    )
    parser.add_argument("--csv", metavar="PATH", help="also write a CSV here")
    parser.add_argument(
        "--plot", metavar="PATH", help="also write a PNG of the curves here"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print first -> last for the headline metrics instead of a table",
    )
    parser.add_argument(
        "--keys",
        metavar="KEY",
        nargs="+",
        help="columns to show, overriding the defaults",
    )
    parser.add_argument(
        "--list-keys", action="store_true", help="list every numeric key logged"
    )
    args = parser.parse_args(argv)

    try:
        run_file = find_run_file(args.run)
        rows = read_history(run_file)
        if not rows:
            raise MetricsError(f"{run_file} holds no history records")

        mode = "offline" if "offline-run-" in run_file else "online"
        print(f"{run_file}  ({len(rows)} records, logged {mode})")
        rows = with_training_step(rows)

        if args.list_keys:
            for key in numeric_keys(rows):
                print(f"  {key}  ({len(series(rows, key))} points)")
            return 0

        columns = tuple(args.keys) if args.keys else None
        if args.summary:
            print_summary(rows, columns or SUMMARY_KEYS)
        else:
            print_table(rows, columns or DEFAULT_COLUMNS)

        if args.csv:
            written = write_csv(rows, columns or DEFAULT_COLUMNS, args.csv)
            print(f"{written} steps -> {args.csv}")
        if args.plot:
            write_plot(rows, args.plot, columns or PLOT_KEYS)
            print(f"plot -> {args.plot}")
    except MetricsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
