"""Visualize ARC-AGI task example input/output grids in the terminal.

Examples:
  # Show first training example of a task
  python tools/visualize_example.py --split training --task 0ca9ddb6 --subset train --index 0

  # Show a test example (will only show output if solution exists)
  python tools/visualize_example.py --split evaluation --task 0ca9ddb6 --subset test --index 1

Optional flags:
  --show-digits  : overlay numeric color indices inside the colored cells
  --transform    : attempt to print transformation label using ARCAnalyzer
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import ARCParser, ARCAnalyzer  # type: ignore
from analysis.utils import print_side_by_side  # type: ignore


def load_example(parser: ARCParser, split: str, task_id: str, subset: str, index: int):
    task = parser.get_task(task_id)
    if task is None:
        raise SystemExit(f"Task '{task_id}' not found in split '{split}'.")

    if subset not in ("train", "test"):
        raise SystemExit("subset must be 'train' or 'test'")

    examples = task.get(subset, [])
    if index >= len(examples):
        raise SystemExit(f"Index {index} out of range (have {len(examples)} examples in subset '{subset}').")

    ex = examples[index]
    if subset == "train":
        inp = ex["input"] if isinstance(ex, dict) else ex[0]
        out = ex["output"] if isinstance(ex, dict) else ex[1]
    else:  # test subset; output may be missing
        inp = ex["input"] if isinstance(ex, dict) else ex
        # look up solution if available
        sol = parser.get_solution(task_id) or {}
        test_solutions = sol.get("test", []) if isinstance(sol, dict) else []
        out = test_solutions[index] if index < len(test_solutions) else None
    return inp, out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visualize ARC-AGI example grids")
    ap.add_argument("--split", choices=["training", "test", "evaluation"], default="training")
    ap.add_argument("--task", required=True, help="Task ID (hex string)")
    ap.add_argument("--subset", choices=["train", "test"], default="train")
    ap.add_argument("--index", type=int, default=0, help="Example index within the subset")
    ap.add_argument("--show-digits", action="store_true", help="Overlay digits inside colored cells")
    ap.add_argument("--transform", action="store_true", help="Compute and display transformation label")

    args = ap.parse_args(argv)

    parser = ARCParser()
    parser.load_data(args.split)
    inp, out = load_example(parser, args.split, args.task, args.subset, args.index)

    titles = ["INPUT"]
    grids = [inp]
    label = None

    if out is not None:
        titles.append("OUTPUT")
        grids.append(out)
        if args.transform:
            analyzer = ARCAnalyzer(parser)
            label = analyzer.analyze_transformation(inp, out)

    print_side_by_side(grids, titles=titles, show_digit=args.show_digits)

    if label is not None:
        print(f"\nTransformation: {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
