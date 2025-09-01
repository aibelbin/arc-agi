# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

# ======================================================================================
# SECTION 1: DATA LOADING (Your dataset.py code)
# ======================================================================================
# This section handles loading the .pt tensor files you created.

class ARCTaskDataset(Dataset):
    """
    Custom PyTorch Dataset to load one entire ARC task at a time.
    Each item from this dataset is a dictionary containing all the 
    tensors for a single task.
    """
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
        test_pairs = self._load_pairs(os.path.join(task_path, 'test'))

        return {
            "train_inputs": [pair['input'] for pair in train_pairs],
            "train_outputs": [pair['output'] for pair in train_pairs],
            "test_input": test_pairs[0]['input'],
            "test_output": test_pairs[0]['output'] # This is the ground truth label
        }

    def _load_pairs(self, directory):
        pairs = []
        example_idx = 0
        while True:
            input_path = os.path.join(directory, f'example_{example_idx}_input.pt')
            output_path = os.path.join(directory, f'example_{example_idx}_output.pt')

            if not os.path.exists(input_path):
                break 

            input_tensor = torch.load(input_path)['one_hot']
            output_tensor = torch.load(output_path)['one_hot']
            
            pairs.append({'input': input_tensor, 'output': output_tensor})
            example_idx += 1
        return pairs

# ======================================================================================
# SECTION 2: CUSTOM BATCHING (The collate_fn)
# ======================================================================================
# This function tells the DataLoader how to combine multiple tasks into a single batch,
# even if they have different numbers of training examples. It does this by "padding"
# the shorter tasks with empty tensors until they are all the same size.

def arc_collate_fn(batch):
    """
    Custom collate function to handle padding for ARC tasks.
    """
    max_train_pairs = max(len(item['train_inputs']) for item in batch)
    
    # Assuming all grids have the same dimensions
    C, H, W = batch[0]['train_inputs'][0].shape
    
    batched_train_inputs, batched_train_outputs = [], []
    batched_test_inputs, batched_test_outputs = [], []
    
    padding_tensor = torch.zeros((C, H, W), dtype=torch.float32)
    
    for item in batch:
        num_padding = max_train_pairs - len(item['train_inputs'])
        
        padded_train_inputs = item['train_inputs'] + [padding_tensor] * num_padding
        padded_train_outputs = item['train_outputs'] + [padding_tensor] * num_padding
        
        batched_train_inputs.append(torch.stack(padded_train_inputs))
        batched_train_outputs.append(torch.stack(padded_train_outputs))
        batched_test_inputs.append(item['test_input'])
        batched_test_outputs.append(item['test_output'])

    return {
        'train_inputs': torch.stack(batched_train_inputs),
        'train_outputs': torch.stack(batched_train_outputs),
        'test_input': torch.stack(batched_test_inputs),
        'test_output': torch.stack(batched_test_outputs)
    }

# ======================================================================================
# SECTION 3: THE MODEL ARCHITECTURE
# ======================================================================================
# This is a simplified Encoder-Reasoner-Decoder model.
# - Encoder: Converts each grid image into a compact feature vector (embedding).
# - Reasoner: Uses attention to compare the embeddings and figure out the rule.
# - Decoder: Converts the final vector back into an output grid image.

class ARCSolver(nn.Module):
    def __init__(self, num_colors=10, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- ENCODER ---
        # A simple CNN to process 30x30 grids
        self.encoder = nn.Sequential(
            nn.Conv2d(num_colors, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Shrink to 1x1
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

        # --- REASONER ---
        # A simple attention mechanism to find the transformation rule
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.reasoner_ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

        # --- DECODER ---
        # A "deconvolution" network to rebuild the grid
        self.decoder_start = nn.Linear(embed_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            # Upsamples to 8x8, then 15x15 (almost 16x16), then we resize
            nn.ConvTranspose2d(64, num_colors, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, train_inputs, train_outputs, test_input):
        # Shape of inputs: (Batch, NumPairs, C, H, W)
        B, N, C, H, W = train_inputs.shape
        
        # --- 1. Encode all grids ---
        # Reshape to process all grids in one go: (Batch * NumPairs, C, H, W)
        train_inputs_flat = train_inputs.view(B * N, C, H, W)
        train_outputs_flat = train_outputs.view(B * N, C, H, W)
        test_input_flat = test_input # Shape: (Batch, C, H, W)
        
        # Get embeddings
        train_in_embed = self.encoder(train_inputs_flat).view(B, N, -1)
        train_out_embed = self.encoder(train_outputs_flat).view(B, N, -1)
        test_in_embed = self.encoder(test_input_flat).view(B, 1, -1)
        
        # --- 2. Reason about the transformation ---
        # We form a sequence for the attention layer: [in1, out1, in2, out2, ..., test_in]
        # We want the model to learn the pattern from in->out pairs
        # and apply it to test_in.
        
        # Simple approach: average the transformations
        transformation_vectors = train_out_embed - train_in_embed # Learn the "difference"
        avg_transformation = transformation_vectors.mean(dim=1, keepdim=True) # (B, 1, embed_dim)
        
        # Apply the learned transformation to the test input
        reasoned_embed = test_in_embed + avg_transformation
        
        # --- 3. Decode the result ---
        decoder_input = self.reasoner_ffn(reasoned_embed.squeeze(1)) # Shape: (B, embed_dim)
        x = self.decoder_start(decoder_input)
        x = x.view(-1, 128, 4, 4) # Reshape for deconvolution
        x = self.decoder(x)

        # The output size might be slightly off (e.g., 28x28 or 30x30). We resize to be sure.
        # This gives us the final prediction grid. The loss function wants (B, C, H, W).
        output_grid = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return output_grid

# ======================================================================================
# SECTION 4: MAIN TRAINING SCRIPT
# ======================================================================================

if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = 'processed/train'
    BATCH_SIZE = 8 # Adjust based on your GPU memory
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EMBED_DIM = 256 # Internal size of the model's feature vectors
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Create Dataset and DataLoader ---
    dataset = ARCTaskDataset(processed_dir=DATA_DIR)
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=arc_collate_fn
    )
    
    # --- Initialize Model, Loss, and Optimizer ---
    model = ARCSolver(num_colors=10, embed_dim=EMBED_DIM).to(device)
    # CrossEntropyLoss is used for classification problems (classifying each pixel's color)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train() # Set the model to training mode
        total_loss = 0
        
        for batch in data_loader:
            # Move all tensors in the batch to the selected device (GPU/CPU)
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            test_input = batch['test_input'].to(device)
            # This is the ground truth, our target for the loss function
            true_test_output = batch['test_output'].to(device)
            
            # --- Forward Pass ---
            optimizer.zero_grad() # Reset gradients
            predicted_test_output = model(train_inputs, train_outputs, test_input)
            
            # --- Calculate Loss ---
            # The model predicts one-hot grids (B, C, H, W), but the loss function
            # wants class indices (B, H, W). .argmax() converts it.
            true_test_output_indices = true_test_output.argmax(dim=1)
            loss = loss_fn(predicted_test_output, true_test_output_indices)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} --- Average Loss: {avg_loss:.4f}")

    print("Training finished!")
    # TODO: Add saving the model weights and an evaluation loop