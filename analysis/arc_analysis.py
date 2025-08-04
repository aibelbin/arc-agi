import json
import numpy as np
from collections import Counter

class ARCParser:
    def __init__(self, data_path="../default/"):
        self.data_path = data_path
        self.training_data = None
        self.training_solutions = None
        
    def load_data(self):
        with open(f"{self.data_path}/arc-agi_training_challenges.json", 'r') as f:
            self.training_data = json.load(f)
        with open(f"{self.data_path}/arc-agi_training_solutions.json", 'r') as f:
            self.training_solutions = json.load(f)
    
    def get_task(self, task_id):
        return self.training_data.get(task_id)
    
    def get_solution(self, task_id):
        return self.training_solutions.get(task_id)

class ARCAnalyzer:
    def __init__(self, parser):
        self.parser = parser
    
    def analyze_transformation(self, input_grid, output_grid):
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        same_shape = input_array.shape == output_array.shape
        same_size = input_array.size == output_array.size
        
        if same_shape:
            if np.array_equal(output_array, np.rot90(input_array)):
                return "rotation_90"
            elif np.array_equal(output_array, np.rot90(input_array, 2)):
                return "rotation_180"
            elif np.array_equal(output_array, np.flipud(input_array)):
                return "flip_vertical"
            elif np.array_equal(output_array, np.fliplr(input_array)):
                return "flip_horizontal"
            elif not np.array_equal(input_array, output_array):
                return "color_change"
            else:
                return "identical"
        
        if output_array.size > input_array.size:
            h_out, w_out = output_array.shape
            h_in, w_in = input_array.shape
            if h_out % h_in == 0 and w_out % w_in == 0:
                scale = h_out // h_in
                if scale == w_out // w_in:
                    return f"scaling_x{scale}"
        
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
