from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from models import GCN
from torch.optim import Adam
from torch.nn import MSELoss

# Load the QM9 dataset
dataset = QM9(root="data/")

# extract target 7, 8, 9
dataset.data.y = dataset.data.y[:, [7, 8, 9]]

# construct trainin and evaluation datasets
train_dataset = dataset[: dataset.data.y.size(0) * 4 // 5]
eval_dataset = dataset[dataset.data.y.size(0) * 4 // 5 :]


# dataloader
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=2048, shuffle=False)


model = GCN(
    input_dim=11,
    hidden_dim=32,
    output_dim=3,
    num_layers=3,  # arbitraryly chosen
)

# optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# loss
mse_loss = MSELoss()

# training loop
for epoch in range(100):
    model.train()
    for data in train_loader:
        out = model(data)
        loss = mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    mean_loss = 0
    for data in eval_loader:
        out = model(data)
        loss = mse_loss(out, data.y)
        mean_loss += loss.item()
    print(f"Epoch {epoch}, loss: {mean_loss / len(eval_loader)}")
