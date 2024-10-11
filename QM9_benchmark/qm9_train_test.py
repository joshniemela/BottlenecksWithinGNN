from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from models import GCN_regressor as GCN
from torch.optim import Adam
from torch.nn import MSELoss
from torch import cuda
from torch.optim.lr_scheduler import StepLR

BATCH_SIZE = 2048 * 4

device = "cuda" if cuda.is_available() else "cpu"

# Load the QM9 dataset
dataset = QM9(root="data/").to(device)

# construct trainin and evaluation datasets
train_dataset = dataset[: dataset.y.size(0) * 4 // 5]
eval_dataset = dataset[dataset.y.size(0) * 4 // 5 :]


# dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loss
mse_loss = MSELoss().to(device)

scores = {}
for config in [False, True]:
    model = GCN(
        input_dim=11,
        hidden_dim=32,
        output_dim=3,
        num_layers=3,  # arbitraryly chosen
        use_fully_adj=config,
    ).to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=0.01)

    # training loop
    for epoch in range(250):
        model.train()
        for data in train_loader:
            out = model(data)
            loss = mse_loss(out, data.y[:, 7:10])  # we use only the targets 7, 8, 9
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        mean_loss = 0
        for data in eval_loader:
            out = model(data)
            loss = mse_loss(out, data.y[:, 7:10])
            mean_loss += loss.item()
        print(f"Epoch {epoch}, loss: {mean_loss / BATCH_SIZE}")
    mean_loss = mean_loss / len(eval_loader)
    scores[config] = mean_loss

print(scores)
