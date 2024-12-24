import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import generate_three_nodes_dataset
from models import NonLinearSAGE
from torch_geometric.loader import DataLoader


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.data)


# Create a dataset
data_list = generate_three_nodes_dataset(300)
data_loader = DataLoader(data_list, batch_size=1028, shuffle=True)

# Create a grid of w1 and w2 values
w1_range = np.linspace(-1, 1, 100)
w2_range = np.linspace(-1, 1, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Initialize your model
model = NonLinearSAGE(activation="gaussian")
criterion = torch.nn.CrossEntropyLoss()

Loss = np.zeros((len(w1_range), len(w2_range)))

max_loss = 0
max_loss_w_1_weight = 0
max_loss_w_2_weight = 0

# Calculate loss over the grid
for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        # Set model parameters to specific values in the grid
        model.conv.lin_l.weight.data = torch.tensor([[W1[i, j]]], dtype=torch.float32)
        model.conv.lin_r.weight.data = torch.tensor([[W2[i, j]]], dtype=torch.float32)
        total_loss = 0
        for batch in data_loader:
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
        Loss[i, j] = total_loss / len(data_loader)
        if Loss[i, j] > max_loss:
            max_loss = Loss[i, j]
            max_loss_w_1_weight = W1[i, j]
            max_loss_w_2_weight = W2[i, j]
    print(f"Finished {i+1}/{len(w1_range)} iterations")

print(f"Max loss: {max_loss:.4f}")
print(f"Max loss w1 weight: {max_loss_w_1_weight:.4f}")
print(f"Max loss w2 weight: {max_loss_w_2_weight:.4f}")


# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(W1, W2, Loss, cmap="viridis", edgecolor="none")

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("Loss")

# high resolution plot
plt.savefig("three_neighbour_classifier_gaussian_activation.svg", dpi=900)
