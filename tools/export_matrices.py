"""
Export ARC-AGI grids as numeric matrices.

For each task and example, saves input/output grids to disk as CSV or NPY files.

Usage examples:
  - Export training split to CSVs under artifacts/matrices:
      python tools/export_matrices.py --split training

  - Export evaluation split to NPY files into a custom dir:
      python tools/export_matrices.py --split evaluation --format npy --out-dir ./out

  - Export all splits, overwriting existing files:
      python tools/export_matrices.py --split all --overwrite
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import ARCParser


def to_matrix(grid) -> np.ndarray:
    """Convert a nested list grid into a numpy int array."""
    return np.array(grid, dtype=int)


def save_matrix(arr: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        np.savetxt(path, arr, fmt="%d", delimiter=",")
    elif fmt == "npy":
        np.save(path, arr)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def iter_splits(arg_split: str) -> Iterable[str]:
    if arg_split == "all":
        return ("training", "test", "evaluation")
    return (arg_split,)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export ARC-AGI grids as matrices")
    parser.add_argument("--split", choices=["training", "test", "evaluation", "all"], default="training",
                        help="Dataset split to export")
    parser.add_argument("--out-dir", default="artifacts/matrices", help="Output directory root")
    parser.add_argument("--format", choices=["csv", "npy"], default="csv", help="Output file format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit of tasks to export per split (for quick tests)")

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_dir).resolve()

    total_written = 0
    for split in iter_splits(args.split):
        data_parser = ARCParser()
        data_parser.load_data(split)

        solutions = data_parser.training_solutions or {}

        tasks_iter = data_parser.training_data.items()
        if args.limit is not None:
            tasks_iter = list(tasks_iter)[: args.limit]

        for task_id, task in tasks_iter:
            for i, ex in enumerate(task.get("train", [])):
                inp = to_matrix(ex["input"]) if isinstance(ex, dict) else to_matrix(ex[0])
                out = to_matrix(ex["output"]) if isinstance(ex, dict) else to_matrix(ex[1])

                base = out_root / split / task_id / "train"
                in_path = base / f"{i}_input.{args.format}"
                out_path = base / f"{i}_output.{args.format}"
                if args.overwrite or not in_path.exists():
                    save_matrix(inp, in_path, args.format)
                if args.overwrite or not out_path.exists():
                    save_matrix(out, out_path, args.format)
                total_written += 2

            test_examples = task.get("test", [])
            task_solutions = solutions.get(task_id, {}) if isinstance(solutions, dict) else {}
            sol_tests = task_solutions.get("test", []) if isinstance(task_solutions, dict) else []

            for i, ex in enumerate(test_examples):
                inp = to_matrix(ex["input"]) if isinstance(ex, dict) else to_matrix(ex)
                base = out_root / split / task_id / "test"
                in_path = base / f"{i}_input.{args.format}"
                if args.overwrite or not in_path.exists():
                    save_matrix(inp, in_path, args.format)
                    total_written += 1

                if i < len(sol_tests):
                    out_grid = sol_tests[i]
                    out = to_matrix(out_grid)
                    out_path = base / f"{i}_output.{args.format}"
                    if args.overwrite or not out_path.exists():
                        save_matrix(out, out_path, args.format)
                        total_written += 1

    print(f"Export complete. Files written: {total_written}. Output at: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
