

import torch

# Node features and corresponding batch numbers
node_features = torch.tensor([[1], [2], [3], [4], [5], [2], [5]], dtype=torch.float)
batch_numbers = torch.tensor([0, 0, 0, 0, 0, 1, 1])

# Step 1: Determine the number of batches and the maximum length
batch_size = batch_numbers.max().item() + 1
batch_counts = torch.bincount(batch_numbers)
max_length = batch_counts.max().item()

# Step 2: Initialize the padded tensor
padded_tensor = torch.zeros(batch_size, max_length, node_features.size(1))

# Step 3: Create an index tensor for positions within each batch
# Sort batch_numbers to group indices for each batch together
_, inverse_indices = torch.sort(batch_numbers)
# Create cumulative counts for each batch
positions = torch.cumsum(torch.ones_like(batch_numbers), dim=0) - 1
# Adjust positions so that they reset for each new batch
positions -= positions.clone().index_select(0, inverse_indices).cumsum(0)[inverse_indices]

# Step 4: Place node features into the padded tensor
padded_tensor[batch_numbers, positions] = node_features

print(padded_tensor)