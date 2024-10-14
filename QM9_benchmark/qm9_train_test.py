from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from models import GCN_regressor as GCN
from torch.optim import Adam
from torch.nn import MSELoss
from torch import cuda
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

BATCH_SIZE = 2048 * 4

device = "cuda" if cuda.is_available() else "cpu"

# Load the QM9 dataset
dataset = QM9(root="data/").to(device)


# use only the first 10k samples
dataset = dataset[:6000]

# normalize the targets
y_mean = dataset.data.y.mean(dim=0, keepdim=True)
y_std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - y_mean) / y_std

# construct trainin and evaluation datasets
train_dataset = dataset[: dataset.y.size(0) * 4 // 5]
eval_dataset = dataset[dataset.y.size(0) * 4 // 5 :]


# dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# loss
mse_loss = MSELoss().to(device)

scores = {}
for config in [None, "mlp", "lstm"]:
    model = GCN(
        input_dim=11,
        hidden_dim=16,
        output_dim=3,
        num_layers=5,  # arbitraryly chosen
        use_fully_adj=False,
        aggregator_mode=config,
    ).to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=1)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # training loop
    for epoch in range(200):
        model.train()
        for data in train_loader:
            out = model(data)
            loss = mse_loss(out, data.y[:, 7:10])  # we use only the targets 7, 8, 9
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step(loss.item())

        model.eval()
        mean_loss = 0
        for data in eval_loader:
            out = model(data)
            loss = mse_loss(out, data.y[:, 7:10])
            mean_loss += loss.item()
        print(f"Epoch {epoch}, loss: {mean_loss}")
        # we print 5 random examples from predictions and targets
        print(out[1] - data.y[1, 7:10])
    mean_loss = mean_loss / len(eval_loader)
    scores[config] = mean_loss

print(scores)
