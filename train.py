# train.py (Complete Final Version)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

# ======================================================================================
# SECTION 1: DATA LOADING
# ======================================================================================
class ARCTaskDataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.task_names = sorted([d for d in os.listdir(processed_dir) if d.startswith('task_')])
        print(f"Found {len(self.task_names)} tasks in {processed_dir}")

    def __len__(self):
        return len(self.task_names)

    def __getitem__(self, idx):
        task_name = self.task_names[idx]
        task_path = os.path.join(self.processed_dir, task_name)
        train_pairs = self._load_pairs(os.path.join(task_path, 'train'))
        test_input_path = os.path.join(task_path, 'test', 'example_0_input.pt')
        test_output_path = os.path.join(task_path, 'test', 'example_0_output.pt')

        if not os.path.exists(test_output_path):
            # This check is important to ensure data integrity
            raise FileNotFoundError(f"Required solution file not found: {test_output_path}")

        test_input_tensor = torch.load(test_input_path)['one_hot']
        test_output_tensor = torch.load(test_output_path)['one_hot']

        return {
            "train_inputs": [pair['input'] for pair in train_pairs],
            "train_outputs": [pair['output'] for pair in train_pairs],
            "test_input": test_input_tensor,
            "test_output": test_output_tensor,
            "task_name": task_name # For debugging
        }

    def _load_pairs(self, directory):
        pairs = []
        example_idx = 0
        while True:
            input_path = os.path.join(directory, f'example_{example_idx}_input.pt')
            output_path = os.path.join(directory, f'example_{example_idx}_output.pt')
            if not os.path.exists(input_path) or not os.path.exists(output_path):
                break
            pairs.append({
                'input': torch.load(input_path)['one_hot'],
                'output': torch.load(output_path)['one_hot']
            })
            example_idx += 1
        return pairs

# ======================================================================================
# SECTION 2: CUSTOM BATCHING (WITH GRID PADDING)
# ======================================================================================
def arc_collate_fn(batch):
    """
    New collate_fn that pads both the number of training pairs AND the grid dimensions.
    """
    # --- Find max dimensions for padding ---
    max_train_pairs = max(len(item['train_inputs']) for item in batch)
    max_h = 0
    max_w = 0
    for item in batch:
        for grid in item['train_inputs'] + item['train_outputs'] + [item['test_input'], item['test_output']]:
            if grid.ndim == 3: # Ensure grid is not empty
                max_h = max(max_h, grid.shape[1])
                max_w = max(max_w, grid.shape[2])

    # --- Pad and batch all tensors ---
    batched_train_inputs, batched_train_outputs = [], []
    batched_test_inputs, batched_test_outputs = [], []
    
    num_colors = batch[0]['train_inputs'][0].shape[0] if batch[0]['train_inputs'] else 10
    
    # Helper function to pad a single grid
    def pad_grid(grid):
        if grid.ndim < 3: return torch.zeros((num_colors, max_h, max_w))
        h, w = grid.shape[1], grid.shape[2]
        # (padding_left, padding_right, padding_top, padding_bottom)
        padding = (0, max_w - w, 0, max_h - h)
        return F.pad(grid, padding, "constant", 0)

    for item in batch:
        # 1. Pad train pairs to max_train_pairs
        num_pair_padding = max_train_pairs - len(item['train_inputs'])
        padding_grid = torch.zeros((num_colors, max_h, max_w))
        
        # 2. Pad each grid within the pairs
        padded_train_in = [pad_grid(g) for g in item['train_inputs']] + [padding_grid] * num_pair_padding
        padded_train_out = [pad_grid(g) for g in item['train_outputs']] + [padding_grid] * num_pair_padding
        
        batched_train_inputs.append(torch.stack(padded_train_in))
        batched_train_outputs.append(torch.stack(padded_train_out))
        
        # 3. Pad test grids
        batched_test_inputs.append(pad_grid(item['test_input']))
        batched_test_outputs.append(pad_grid(item['test_output']))

    return {
        'train_inputs': torch.stack(batched_train_inputs),
        'train_outputs': torch.stack(batched_train_outputs),
        'test_input': torch.stack(batched_test_inputs),
        'test_output': torch.stack(batched_test_outputs)
    }

# ======================================================================================
# SECTION 3: THE MODEL ARCHITECTURE
# ======================================================================================
class ARCSolver(nn.Module):
    def __init__(self, num_colors=10, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_colors, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, embed_dim)
        )
        self.reasoner_ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.decoder_start = nn.Linear(embed_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, num_colors, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, train_inputs, train_outputs, test_input):
        B, N, C, H, W = train_inputs.shape # Get padded dimensions
        
        train_in_embed = self.encoder(train_inputs.view(B * N, C, H, W)).view(B, N, -1)
        train_out_embed = self.encoder(train_outputs.view(B * N, C, H, W)).view(B, N, -1)
        test_in_embed = self.encoder(test_input).view(B, 1, -1)
        
        transformation_vectors = train_out_embed - train_in_embed
        avg_transformation = transformation_vectors.mean(dim=1, keepdim=True)
        reasoned_embed = test_in_embed + avg_transformation
        
        decoder_input = self.reasoner_ffn(reasoned_embed.squeeze(1))
        x = self.decoder_start(decoder_input).view(-1, 128, 4, 4)
        x = self.decoder(x)
        
        # Interpolate to the final padded size of the batch
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

# ======================================================================================
# SECTION 4: MAIN TRAINING SCRIPT (WITH MODEL SAVING)
# ======================================================================================
if __name__ == '__main__':
    DATA_DIR = 'processed/train'
    MODEL_SAVE_PATH = 'arc_solver_model.pth' # The name of the file your model will be saved to
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = ARCTaskDataset(processed_dir=DATA_DIR)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=arc_collate_fn)
    
    model = ARCSolver().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(data_loader):
            try:
                train_inputs = batch['train_inputs'].to(device)
                train_outputs = batch['train_outputs'].to(device)
                test_input = batch['test_input'].to(device)
                true_test_output = batch['test_output'].to(device)
                
                optimizer.zero_grad()
                predicted_test_output = model(train_inputs, train_outputs, test_input)
                
                # The label must be long integers for CrossEntropyLoss
                true_test_output_indices = true_test_output.argmax(dim=1).long()
                
                loss = loss_fn(predicted_test_output, true_test_output_indices)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error on batch {i}: {e}")
                continue # Skip to the next batch

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} --- Average Loss: {avg_loss:.4f}")
        
    print("Training finished!")

    # --- Save the trained model's state dictionary ---
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully!")