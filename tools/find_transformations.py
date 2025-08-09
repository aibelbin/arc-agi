"""
Analyze transformations between input and output matrices previously exported.

It scans artifacts/matrices (or a custom directory) and, for each example with both
input and output, detects the transformation label using analysis.ARCAnalyzer and
reports additional math-like mappings:
  - integer scaling factor (k) if any
  - tiling type (plain/mirrored)
  - color mapping (input color -> output color) when consistent
  - affine modulo-10 mapping y = (a*x + b) mod 10 if it fits observed pairs

Usage:
  python tools/find_transformations.py --split training --src-dir artifacts/matrices --out-json artifacts/transformations/training.json

You can also run on all splits:
  python tools/find_transformations.py --split all --out-json artifacts/transformations/all.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Ensure repo root on sys.path to import analysis package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import ARCAnalyzer  # type: ignore


@dataclass
class ExampleReport:
    split: str
    task_id: str
    subset: str  # 'train' or 'test'
    index: int
    in_shape: Tuple[int, int]
    out_shape: Optional[Tuple[int, int]]
    transform_label: Optional[str]
    scale_k: Optional[int]
    tiling_type: Optional[str]  # 'plain' | 'mirrored' | None
    color_mapping: Optional[Dict[int, int]]
    is_bijective: Optional[bool]
    affine_mod10: Optional[Dict[str, int]]  # {'a': int, 'b': int}
    has_output: bool


def iter_splits(arg_split: str) -> Iterable[str]:
    if arg_split == "all":
        return ("training", "test", "evaluation")
    return (arg_split,)


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".csv":
        return np.loadtxt(path, dtype=int, delimiter=",")
    elif path.suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"Unsupported matrix format: {path.suffix}")


def extract_index(name: str) -> int:
    # Expect filenames like '0_input.csv' or '12_output.npy'
    m = re.match(r"^(\d+)_", name)
    if not m:
        raise ValueError(f"Cannot extract index from filename: {name}")
    return int(m.group(1))


def compute_color_mapping(a: np.ndarray, b: np.ndarray) -> Optional[Dict[int, int]]:
    if a.shape != b.shape:
        return None
    mapping: Dict[int, int] = {}
    mapped = set()
    for av, bv in zip(a.ravel(), b.ravel()):
        av = int(av)
        bv = int(bv)
        if av in mapping:
            if mapping[av] != bv:
                return None
        else:
            if bv in mapped:
                return None
            mapping[av] = bv
            mapped.add(bv)
    return mapping


def fit_affine_mod10(mapping: Dict[int, int]) -> Optional[Tuple[int, int]]:
    # Try all a,b in 0..9 for small domain, accept if holds for all seen x
    if not mapping:
        return None
    domain = list(mapping.keys())
    codomain = mapping
    for a in range(10):
        for b in range(10):
            ok = True
            for x in domain:
                y = codomain[x]
                if (a * x + b) % 10 != y:
                    ok = False
                    break
            if ok:
                return a, b
    return None


def parse_scale_from_label(label: str) -> Optional[int]:
    m = re.search(r"x(\d+)", label)
    return int(m.group(1)) if m else None


def tiling_type_from_label(label: str) -> Optional[str]:
    if label.startswith("tiling_mirrored"):
        return "mirrored"
    if label.startswith("tiling_x"):
        return "plain"
    return None


def analyze_example(analyzer: ARCAnalyzer, a: np.ndarray, b: Optional[np.ndarray]) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[Dict[int, int]], Optional[bool], Optional[Dict[str, int]]]:
    if b is None:
        return None, None, None, None, None, None

    label = analyzer.analyze_transformation(a, b)
    k = parse_scale_from_label(label) if label else None
    ttype = tiling_type_from_label(label) if label else None

    cmap = compute_color_mapping(a, b)
    is_bijective = None
    affine = None
    if cmap is not None:
        is_bijective = len(set(cmap.values())) == len(cmap)
        aff = fit_affine_mod10(cmap)
        if aff is not None:
            affine = {"a": aff[0], "b": aff[1]}

    return label, k, ttype, cmap, is_bijective, affine


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze transformations between exported matrices")
    ap.add_argument("--split", choices=["training", "test", "evaluation", "all"], default="training")
    ap.add_argument("--src-dir", default="artifacts/matrices", help="Root directory of exported matrices")
    ap.add_argument("--out-json", default=None, help="Path to write JSON report. Default is artifacts/transformations/<split>.json or all.json")
    ap.add_argument("--format", choices=["auto", "csv", "npy"], default="auto", help="Expected matrix file format")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of tasks per split")

    args = ap.parse_args(argv)

    src_root = (REPO_ROOT / args.src_dir).resolve()
    splits = list(iter_splits(args.split))
    out_path: Path
    if args.out_json:
        out_path = (REPO_ROOT / args.out_json).resolve()
    else:
        out_dir = REPO_ROOT / "artifacts" / "transformations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{splits[0] if len(splits)==1 else 'all'}.json"
        out_path = out_dir / out_name

    analyzer = ARCAnalyzer(None)  # parser not needed for single-example analysis

    reports: List[ExampleReport] = []
    for split in splits:
        split_dir = src_root / split
        if not split_dir.exists():
            # Skip missing splits silently
            continue
        task_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if args.limit is not None:
            task_dirs = task_dirs[: args.limit]

        for task_dir in task_dirs:
            for subset in ("train", "test"):
                subdir = task_dir / subset
                if not subdir.exists():
                    continue
                # Build index set based on available input files
                candidates = []
                for p in subdir.iterdir():
                    if not p.is_file():
                        continue
                    if args.format == "auto":
                        if p.name.endswith("_input.csv") or p.name.endswith("_input.npy"):
                            candidates.append(p)
                    elif args.format == "csv" and p.name.endswith("_input.csv"):
                        candidates.append(p)
                    elif args.format == "npy" and p.name.endswith("_input.npy"):
                        candidates.append(p)

                for in_path in sorted(candidates, key=lambda x: extract_index(x.name)):
                    idx = extract_index(in_path.name)
                    # Derive output path with same idx
                    out_ext = ".csv" if in_path.suffix == ".csv" else ".npy"
                    out_path_candidate = subdir / f"{idx}_output{out_ext}"

                    a = load_matrix(in_path)
                    b = load_matrix(out_path_candidate) if out_path_candidate.exists() else None

                    label, k, ttype, cmap, is_bij, affine = analyze_example(analyzer, a, b)

                    rep = ExampleReport(
                        split=split,
                        task_id=task_dir.name,
                        subset=subset,
                        index=idx,
                        in_shape=tuple(a.shape),
                        out_shape=tuple(b.shape) if b is not None else None,
                        transform_label=label,
                        scale_k=k,
                        tiling_type=ttype,
                        color_mapping=cmap,
                        is_bijective=is_bij,
                        affine_mod10=affine,
                        has_output=b is not None,
                    )
                    reports.append(rep)

    # Serialize to JSON (convert numpy types to int)
    def normalize(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {int(k) if isinstance(k, (np.integer, int)) else k: normalize(v) for k, v in obj.items()}
        return obj

    out_data = [normalize(asdict(r)) for r in reports]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Analysis complete. {len(reports)} examples analyzed. Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
