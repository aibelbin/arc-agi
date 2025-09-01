# tools/json_to_tensors.py (FINAL - Structure-Aware Version)
import json
from pathlib import Path
import torch
import numpy as np

def grid_to_one_hot(grid, num_channels=10):
    """Converts a 2D grid of integers to a one-hot tensor."""
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2: return torch.zeros((num_channels, 0, 0))
    H, W = arr.shape
    one_hot = torch.zeros((num_channels, H, W), dtype=torch.float32)
    for c in range(num_channels):
        one_hot[c, :, :] = torch.from_numpy((arr == c).astype(np.float32))
    return one_hot

def save_tensor(grid, path):
    """Saves a grid as a dictionary containing both one-hot and raw tensors."""
    torch.save({
        'one_hot': grid_to_one_hot(grid),
        'raw': torch.tensor(grid, dtype=torch.int64)
    }, path)

def main():
    # --- Setup Paths ---
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'default'
    output_dir = base_dir / 'processed'

    train_challenges_path = data_dir / 'arc-agi_training_challenges.json'
    train_solutions_path = data_dir / 'arc-agi_training_solutions.json'
    
    if not train_challenges_path.exists() or not train_solutions_path.exists():
        print(f"[ERROR] Data not found in the 'default' folder.")
        return

    print("Loading JSON files...")
    with open(train_challenges_path, 'r') as f:
        challenges_data = json.load(f)
    with open(train_solutions_path, 'r') as f:
        solutions_data = json.load(f)
    
    # --- Create a direct lookup map for solutions ---
    solutions_map = {str(k): v for k, v in solutions_data.items()}
    print(f"Created a solution lookup map with {len(solutions_map)} entries.")

    print("\nProcessing training tasks...")
    train_output_dir = output_dir / 'train'
    train_output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    for task_id, task_data in challenges_data.items():
        task_id_str = str(task_id)
        task_dir = train_output_dir / f"task_{task_id_str}"
        train_dir = task_dir / 'train'
        test_dir = task_dir / 'test'
        
        # Process training examples (these are always correct)
        train_pairs = task_data.get('train', [])
        for j, pair in enumerate(train_pairs):
            train_dir.mkdir(parents=True, exist_ok=True)
            save_tensor(pair['input'], train_dir / f"example_{j}_input.pt")
            save_tensor(pair['output'], train_dir / f"example_{j}_output.pt")

        # --- NEW LOGIC for processing test examples ---
        solution_grids = solutions_map.get(task_id_str)
        if not solution_grids:
            failure_count += 1
            continue
            
        test_pairs = task_data.get('test', [])
        # Ensure the number of test inputs matches the number of solutions
        if len(test_pairs) != len(solution_grids):
            failure_count += 1
            continue

        for j, pair in enumerate(test_pairs):
            input_grid = pair['input']
            # Directly get the solution grid from the list
            # It might have an 'output' key or it might be the grid itself
            solution_item = solution_grids[j]
            output_grid = solution_item.get('output') if isinstance(solution_item, dict) else solution_item

            test_dir.mkdir(parents=True, exist_ok=True)
            save_tensor(input_grid, test_dir / f"example_{j}_input.pt")
            save_tensor(output_grid, test_dir / f"example_{j}_output.pt")

        success_count += 1
        print(f"  [OK] Successfully processed task {task_id_str}")

    print("\n--- Preprocessing Summary ---")
    print(f"Successfully processed {success_count} tasks.")
    print(f"Skipped {failure_count} tasks due to missing solutions.")
    print("Your data is now ready for training.")


if __name__ == '__main__':
    main()