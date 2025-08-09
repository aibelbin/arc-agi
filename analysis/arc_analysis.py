import json
from pathlib import Path
import numpy as np
from collections import Counter

class ARCParser:
    def __init__(self, data_path: str | None = None):
        # Resolve default data path relative to repository root, not CWD
        default_path = Path(__file__).resolve().parents[1] / "default"
        self.data_path = Path(data_path) if data_path is not None else default_path
        self.training_data = None
        self.training_solutions = None

    def load_data(self, split: str = "training"):
        """
        Load ARC-AGI data for a given split.

        split: 'training' | 'test' | 'evaluation'
        Populates self.training_data and self.training_solutions (empty dict if unavailable).
        """
        split = split.lower()
        if split == "training":
            with open(self.data_path / "arc-agi_training_challenges.json", 'r') as f:
                self.training_data = json.load(f)
            with open(self.data_path / "arc-agi_training_solutions.json", 'r') as f:
                self.training_solutions = json.load(f)
        elif split == "test":
            with open(self.data_path / "arc-agi_test_challenges.json", 'r') as f:
                self.training_data = json.load(f)
            # No official solutions for test split
            self.training_solutions = {}
        elif split == "evaluation":
            with open(self.data_path / "arc-agi_evaluation_challenges.json", 'r') as f:
                self.training_data = json.load(f)
            with open(self.data_path / "arc-agi_evaluation_solutions.json", 'r') as f:
                self.training_solutions = json.load(f)
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_task(self, task_id):
        return self.training_data.get(task_id)

    def get_solution(self, task_id):
        return self.training_solutions.get(task_id)

class ARCAnalyzer:
    def __init__(self, parser):
        self.parser = parser

    @staticmethod
    def _is_color_permutation(a: np.ndarray, b: np.ndarray) -> bool:
        if a.shape != b.shape:
            return False
        mapping: dict[int, int] = {}
        mapped: set[int] = set()
        for av, bv in zip(a.ravel(), b.ravel()):
            av = int(av)
            bv = int(bv)
            if av in mapping:
                if mapping[av] != bv:
                    return False
            else:
                if bv in mapped:
                    return False
                mapping[av] = bv
                mapped.add(bv)
        return True

    @staticmethod
    def _build_mirrored_tiling(a: np.ndarray, k: int) -> np.ndarray:
        rows = []
        for i in range(k):
            tiles = []
            for j in range(k):
                t = a
                if i % 2 == 1:
                    t = np.flipud(t)
                if j % 2 == 1:
                    t = np.fliplr(t)
                tiles.append(t)
            rows.append(np.hstack(tiles))
        return np.vstack(rows)

    def analyze_transformation(self, input_grid, output_grid):
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)

        same_shape = input_array.shape == output_array.shape

        if same_shape:
            if np.array_equal(output_array, np.rot90(input_array)):
                return "rotation_90"
            elif np.array_equal(output_array, np.rot90(input_array, 2)):
                return "rotation_180"
            elif np.array_equal(output_array, np.flipud(input_array)):
                return "flip_vertical"
            elif np.array_equal(output_array, np.fliplr(input_array)):
                return "flip_horizontal"
            elif self._is_color_permutation(input_array, output_array):
                return "color_permutation"
            elif not np.array_equal(input_array, output_array):
                return "other_same_shape"
            else:
                return "identical"

        # Handle enlargements that could be scaling or tiling
        if output_array.size > input_array.size:
            h_out, w_out = output_array.shape
            h_in, w_in = input_array.shape
            if h_out % h_in == 0 and w_out % w_in == 0:
                k_h = h_out // h_in
                k_w = w_out // w_in
                if k_h == k_w:
                    k = k_h
                    # mirrored tiling
                    mirrored = self._build_mirrored_tiling(input_array, k)
                    if np.array_equal(output_array, mirrored):
                        return f"tiling_mirrored_x{k}"
                    # plain tiling
                    tiled = np.tile(input_array, (k, k))
                    if np.array_equal(output_array, tiled):
                        return f"tiling_x{k}"
                    # nearest-neighbor scaling
                    scaled = np.repeat(np.repeat(input_array, k, axis=0), k, axis=1)
                    if np.array_equal(output_array, scaled):
                        return f"scaling_nn_x{k}"
                    # generic scaling fallback
                    return f"scaling_x{k}"

        return "complex_transformation"
    
    def get_dataset_summary(self):
        summary = {
            "total_tasks": len(self.parser.training_data),
            "transformation_types": Counter(),
            "grid_sizes": Counter(),
            "complexity_levels": {"simple": 0, "medium": 0, "complex": 0}
        }
        
        for task_id, task in self.parser.training_data.items():
            for example in task['train']:
                input_shape = np.array(example['input']).shape
                output_shape = np.array(example['output']).shape
                
                summary["grid_sizes"][f"{input_shape[0]}x{input_shape[1]}"] += 1
                
                transform_type = self.analyze_transformation(example['input'], example['output'])
                summary["transformation_types"][transform_type] += 1
                
                max_size = max(input_shape[0], input_shape[1], output_shape[0], output_shape[1])
                if max_size <= 5:
                    summary["complexity_levels"]["simple"] += 1
                elif max_size <= 15:
                    summary["complexity_levels"]["medium"] += 1
                else:
                    summary["complexity_levels"]["complex"] += 1
        
        return summary
