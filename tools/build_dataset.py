"""
Build training and testing datasets by chaining matrix export and transformation analysis.

This script will:
  1) Run tools/export_matrices.py to export input/output grids to matrices.
  2) Run tools/find_transformations.py to detect transformation labels and mappings.
  3) Assemble a JSONL dataset with records containing input, transform, and output.

Each JSONL line contains:
{
  "split": str,            # training | test | evaluation
  "task_id": str,
  "subset": str,           # train | test
  "index": int,
  "transform": {           # transformation info
    "label": str | null,
    "scale_k": int | null,
    "tiling_type": str | null,
    "color_mapping": {int:int} | null,
    "is_bijective": bool | null,
    "affine_mod10": {"a": int, "b": int} | null
  },
  "input": [[int,...],...],
  "output": [[int,...],...]     # included only when available
}

Usage examples:
  # Build datasets for training and test splits using CSV format
  python tools/build_dataset.py --split training,test --format csv

  # Build for all splits using NPY
  python tools/build_dataset.py --split all --format npy

  # Quick sample run limited to first 5 tasks per split
  python tools/build_dataset.py --split training --limit 5 --format csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable  # current interpreter/venv


def iter_splits(spec: str) -> Iterable[str]:
    if spec == "all":
        return ("training", "test", "evaluation")
    if "," in spec:
        return tuple(s.strip() for s in spec.split(",") if s.strip())
    return (spec,)


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".csv":
        return np.loadtxt(path, dtype=int, delimiter=",")
    elif path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported matrix format: {path.suffix}")


def run_export(split: str, out_dir: Path, fmt: str, overwrite: bool, limit: Optional[int]) -> None:
    args = [
        PYTHON,
        str(REPO_ROOT / "tools" / "export_matrices.py"),
        "--split", split,
        "--out-dir", str(out_dir),
        "--format", fmt,
    ]
    if overwrite:
        args.append("--overwrite")
    if limit is not None:
        args += ["--limit", str(limit)]
    subprocess.run(args, check=True)


def run_find_transformations(split: str, src_dir: Path, out_json: Path, fmt: str, limit: Optional[int]) -> None:
    args = [
        PYTHON,
        str(REPO_ROOT / "tools" / "find_transformations.py"),
        "--split", split,
        "--src-dir", str(src_dir),
        "--out-json", str(out_json),
    ]
    if fmt in ("csv", "npy"):
        args += ["--format", fmt]
    if limit is not None:
        args += ["--limit", str(limit)]
    subprocess.run(args, check=True)


def build_dataset_for_split(split: str, matrices_root: Path, transform_report: Path, fmt: str, out_path: Path) -> int:
    with transform_report.open("r") as f:
        records = json.load(f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as w:
        for r in records:
            # We require outputs to be present for supervised training
            if not r.get("has_output"):
                continue

            subset = r["subset"]
            idx = r["index"]
            task_id = r["task_id"]

            # Input/Output matrix paths
            subdir = matrices_root / split / task_id / subset
            in_path = subdir / f"{idx}_input.{fmt}"
            # In case fmt=='auto' from report, try csv then npy
            if not in_path.exists():
                alt = subdir / f"{idx}_input.csv"
                if alt.exists():
                    in_path = alt
                else:
                    alt = subdir / f"{idx}_input.npy"
                    if alt.exists():
                        in_path = alt
            out_path_mat = subdir / f"{idx}_output.{fmt}"
            if not out_path_mat.exists():
                alt = subdir / f"{idx}_output.csv"
                if alt.exists():
                    out_path_mat = alt
                else:
                    alt = subdir / f"{idx}_output.npy"
                    if alt.exists():
                        out_path_mat = alt

            if not in_path.exists() or not out_path_mat.exists():
                # Skip incomplete
                continue

            inp = load_matrix(in_path)
            out = load_matrix(out_path_mat)

            transform = {
                "label": r.get("transform_label"),
                "scale_k": r.get("scale_k"),
                "tiling_type": r.get("tiling_type"),
                "color_mapping": r.get("color_mapping"),
                "is_bijective": r.get("is_bijective"),
                "affine_mod10": r.get("affine_mod10"),
            }

            item = {
                "split": split,
                "task_id": task_id,
                "subset": subset,
                "index": idx,
                "transform": transform,
                "input": inp.astype(int).tolist(),
                "output": out.astype(int).tolist(),
            }
            w.write(json.dumps(item))
            w.write("\n")
            written += 1
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build training/testing datasets from exported matrices and transformation analysis")
    ap.add_argument("--split", default="training,test", help="Comma-separated splits or 'all'")
    ap.add_argument("--format", choices=["csv", "npy"], default="csv", help="Matrix file format to export/use")
    ap.add_argument("--matrices-dir", default="artifacts/matrices", help="Where to export matrices")
    ap.add_argument("--transforms-dir", default="artifacts/transformations", help="Where to store transform reports")
    ap.add_argument("--datasets-dir", default="artifacts/datasets", help="Where to store resulting datasets")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite matrices if they exist")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of tasks to process per split")

    args = ap.parse_args(argv)

    matrices_dir = (REPO_ROOT / args.matrices_dir).resolve()
    transforms_dir = (REPO_ROOT / args.transforms_dir).resolve()
    datasets_dir = (REPO_ROOT / args.datasets_dir).resolve()

    total_written = 0
    for split in iter_splits(args.split):
        # 1) Export matrices
        run_export(split, matrices_dir, args.format, args.overwrite, args.limit)

        # 2) Find transformations -> JSON report per split
        transforms_dir.mkdir(parents=True, exist_ok=True)
        report_path = transforms_dir / f"{split}.json"
        run_find_transformations(split, matrices_dir, report_path, args.format, args.limit)

        # 3) Build dataset JSONL
        datasets_dir.mkdir(parents=True, exist_ok=True)
        out_path = datasets_dir / f"{split}.jsonl"
        written = build_dataset_for_split(split, matrices_dir, report_path, args.format, out_path)
        print(f"Split {split}: wrote {written} examples -> {out_path}")
        total_written += written

    print(f"Done. Total examples written: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
