
from __future__ import annotations
import json, argparse, os, math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

# ---------------- Encoding Helpers -----------------

def grid_to_one_hot(grid: List[List[int]], num_channels: int | None = None) -> torch.Tensor:
    """Convert 2D int grid to one-hot (C,H,W). If num_channels None uses max+1.
    Caps channels at 64 (adjustable) to avoid accidental explosion.
    """
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("Grid must be 2D")
    max_color = int(arr.max(initial=0)) if arr.size else 0
    C = num_channels if num_channels is not None else max_color + 1
    C = min(C, 64)
    H, W = arr.shape
    one_hot = torch.zeros((C, H, W), dtype=torch.float32)
    mask = (arr >= 0) & (arr < C)
    idx_y, idx_x = np.where(mask)
    colors = arr[mask]
    one_hot[colors, idx_y, idx_x] = 1.0
    return one_hot

# ---------------- File IO & Processing --------------

def load_json(path: str | Path) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def infer_all_colors(tasks: List[Dict[str, Any]]) -> int:
    col_max = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for phase in ('train','test'):
            for pair in task.get(phase, []):
                # train examples are dicts with input/output; test may be only input
                if isinstance(pair, dict):
                    g_in = pair.get('input')
                    g_out = pair.get('output')
                    for g in (g_in, g_out):
                        if g is not None:
                            arr = np.array(g)
                            if arr.size:
                                col_max = max(col_max, int(arr.max()))
    return col_max + 1


def process_tasks(tasks: List[Dict[str, Any]], out_dir: Path, solutions: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out_meta = {
        'num_tasks': len(tasks),
        'tasks': {},
    }
    # Determine global channel count (optional). We keep per-grid dynamic channels for space; store global suggestion.
    global_C = infer_all_colors(tasks)
    out_meta['suggested_num_channels'] = global_C

    for task in tasks:
        if not isinstance(task, dict):
            # Attempt to expand from solutions map if available
            if solutions and (task in solutions):
                task = solutions[task]
            else:
                continue
        task_id = task.get('id') or task.get('task_id') or task.get('taskId') or 'unknown'
        task_dir = out_dir / f"task_{task_id}"
        (task_dir / 'train').mkdir(parents=True, exist_ok=True)
        (task_dir / 'test').mkdir(parents=True, exist_ok=True)
        t_meta = {'train': [], 'test': []}

        # Train examples
        for i, pair in enumerate(task.get('train', [])):
            if not isinstance(pair, dict):
                continue
            g_in = pair.get('input')
            g_out = pair.get('output')
            if g_in is None or g_out is None:
                continue
            one_in = grid_to_one_hot(g_in)
            one_out = grid_to_one_hot(g_out)
            raw_in = torch.tensor(g_in, dtype=torch.int64)
            raw_out = torch.tensor(g_out, dtype=torch.int64)
            torch.save({'one_hot': one_in, 'raw': raw_in}, task_dir / 'train' / f'example_{i}_input.pt')
            torch.save({'one_hot': one_out, 'raw': raw_out}, task_dir / 'train' / f'example_{i}_output.pt')
            t_meta['train'].append({'i': i, 'H': raw_in.shape[0], 'W': raw_in.shape[1]})

        # Test examples (no outputs unless solutions provided)
        for j, pair in enumerate(task.get('test', [])):
            g_in = pair.get('input') if isinstance(pair, dict) else None
            if g_in is None:
                continue
            one_in = grid_to_one_hot(g_in)
            raw_in = torch.tensor(g_in, dtype=torch.int64)
            torch.save({'one_hot': one_in, 'raw': raw_in}, task_dir / 'test' / f'example_{j}_input.pt')
            # If solutions given try to attach output
            sol_grid = None
            if solutions:
                sol_task = solutions.get(task_id) or solutions.get(str(task_id))
                if sol_task and 'test' in sol_task:
                    if j < len(sol_task['test']):
                        sol_item = sol_task['test'][j]
                        sol_grid = sol_item.get('output') if isinstance(sol_item, dict) else None
                        if sol_grid is not None:
                            one_out = grid_to_one_hot(sol_grid)
                            raw_out = torch.tensor(sol_grid, dtype=torch.int64)
                            torch.save({'one_hot': one_out, 'raw': raw_out}, task_dir / 'test' / f'example_{j}_output.pt')
            t_meta['test'].append({'j': j, 'has_solution': sol_grid is not None})

        out_meta['tasks'][task_id] = t_meta

    return out_meta

# ---------------- CLI ------------------------------

def main():
    """Entry point supporting two modes:
    1. Automatic (no CLI args): processes both training and test splits with hardcoded default paths.
    2. Manual (CLI args provided): legacy behavior using --challenges / --solutions / --out.
    """
    import sys
    if len(sys.argv) == 1:
        # -------- AUTO MODE --------
        base = Path(__file__).resolve().parent.parent  # repo root (.. from tools)
        default_dir = base / 'default'
        train_ch = default_dir / 'arc-agi_training_challenges.json'
        train_sol = default_dir / 'arc-agi_training_solutions.json'
        test_ch = default_dir / 'arc-agi_test_challenges.json'
        missing = [p for p in [train_ch, train_sol, test_ch] if not p.exists()]
        if missing:
            print('[auto] Missing expected files:', ', '.join(str(m) for m in missing))
            print('Place the ARC-AGI JSON files under default/ or run with arguments manually.')
            return
        out_root = base / 'processed'
        # Training split
        print('[auto] Processing training split...')
        train_out = out_root / 'train'
        train_out.mkdir(parents=True, exist_ok=True)
        challenges = load_json(train_ch)
        solutions_raw = load_json(train_sol)
        if isinstance(solutions_raw, list):
            solutions_map = {t.get('id') or t.get('task_id'): t for t in solutions_raw}
        elif isinstance(solutions_raw, dict):
            solutions_map = solutions_raw
        else:
            raise ValueError('Training solutions JSON format not recognized')
        # Normalize challenges list possibly being list of ids
        if isinstance(challenges, list) and challenges and isinstance(challenges[0], str):
            # Replace with full task objects from solutions if present
            expanded = []
            for tid in challenges:
                obj = solutions_map.get(tid) or solutions_map.get(str(tid))
                if obj:
                    expanded.append(obj)
            challenges = expanded
            print(f'[auto] Expanded {len(challenges)} tasks from ID list using solutions map.')
        elif isinstance(challenges, dict):
            # Possibly dict of id->task
            challenges = list(challenges.values())
            print(f'[auto] Converted challenges dict to list of {len(challenges)} tasks.')
        meta_train = process_tasks(challenges, train_out, solutions_map)
        (train_out / 'meta.json').write_text(json.dumps(meta_train, indent=2))
        # Test split
        print('[auto] Processing test split...')
        test_out = out_root / 'test'
        test_out.mkdir(parents=True, exist_ok=True)
        test_tasks = load_json(test_ch)
        if isinstance(test_tasks, list) and test_tasks and isinstance(test_tasks[0], str):
            # Test challenges may be id list without outputs; create empty stubs
            test_tasks = [{'id': tid, 'train': [], 'test': []} for tid in test_tasks]
            print(f'[auto] Expanded test id list into {len(test_tasks)} empty task stubs.')
        elif isinstance(test_tasks, dict):
            test_tasks = list(test_tasks.values())
            print(f'[auto] Converted test challenges dict to list of {len(test_tasks)} tasks.')
        meta_test = process_tasks(test_tasks, test_out, None)
        (test_out / 'meta.json').write_text(json.dumps(meta_test, indent=2))
        print('[auto] Done.')
        print('[auto] Train tasks:', meta_train['num_tasks'], 'Test tasks:', meta_test['num_tasks'])
        return
    # -------- MANUAL MODE --------
    ap = argparse.ArgumentParser()
    ap.add_argument('--challenges', required=True, help='Path to ARC-AGI *_challenges.json')
    ap.add_argument('--solutions', required=False, help='Path to ARC-AGI *_solutions.json (optional)')
    ap.add_argument('--out', required=True, help='Output directory for processed tensors')
    args = ap.parse_args()

    challenges = load_json(args.challenges)
    if not isinstance(challenges, list):
        raise ValueError('Challenges JSON must be a list of task objects')

    solutions = None
    if args.solutions:
        solutions = load_json(args.solutions)
        # Solutions might be a list or dict keyed by id; normalize
        if isinstance(solutions, list):
            solutions = {t.get('id') or t.get('task_id'): t for t in solutions}
        elif not isinstance(solutions, dict):
            raise ValueError('Solutions JSON must be list or dict')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = process_tasks(challenges, out_dir, solutions)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('Wrote tensor dataset to', out_dir)
    print('Tasks:', meta['num_tasks'])
    print('Suggested global channels:', meta['suggested_num_channels'])

if __name__ == '__main__':
    main()
