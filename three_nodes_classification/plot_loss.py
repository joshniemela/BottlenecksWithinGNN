import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import generate_three_nodes_dataset
from models import GCN, SAGE, NonLinearSAGE
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use('Qt5Agg')

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.data)

# Create a dataset
data_list = generate_three_nodes_dataset(300)
data_loader = DataLoader(data_list, batch_size=1028, shuffle=True)

# Create a grid of w1 and w2 values
w1_range = np.linspace(-3, 3,200)
w2_range = np.linspace(-3, 3, 200)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Initialize your model
model = NonLinearSAGE()
criterion = torch.nn.CrossEntropyLoss()

Loss = np.zeros((len(w1_range), len(w2_range)))

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

    print(f"Finished {i+1}/{len(w1_range)} iterations")

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Loss, cmap='viridis', edgecolor='none')

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('3D Loss Surface')

# high resolution plot
plt.savefig('loss_3d.png', dpi=900)

plt.ion()
plt.show(block=True)
