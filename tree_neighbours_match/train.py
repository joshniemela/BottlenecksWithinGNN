import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch import cuda
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dataset as ds
from models import GCN

epochs = 1000
accum_grad = 1
eval_every = 1000
patience = 20

device = "cuda" if cuda.is_available() else "cpu"
device = "cpu"

model = GCN(9, 32, 9, 4).to(device)
criterion = F.cross_entropy
optimiser = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimiser, mode="max", threshold_mode="abs", factor=0.5, patience=10
)
print("Starting training")

train_data, test_data = ds.train_test_split(ds.generate_tree(10000, 3, device))
train_loader = DataLoader(train_data, batch_size=516, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, pin_memory=True)

best_test_acc = 0.0
best_train_acc = 0.0
best_epoch = 0
epochs_no_improve = 0


for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_num_examples = 0
    train_correct = 0
    optimiser.zero_grad()
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        total_num_examples += batch.num_graphs
        total_loss += loss.item() * batch.num_graphs
        _, train_pred = out.max(dim=1)
        train_correct += train_pred.eq(batch.y).sum().item()

        loss = loss / accum_grad
        loss.backward()
        if (i + 1) % accum_grad == 0:
            optimiser.step()
            optimiser.zero_grad()

    avg_training_loss = total_loss / total_num_examples
    train_acc = train_correct / total_num_examples
    scheduler.step(train_acc)

    model.eval()
    with torch.no_grad():

        total_correct = 0
        total_examples = 0
        for batch in test_loader:
            batch = batch.to(device)
            _, pred = model(batch).max(dim=1)
            total_correct += pred.eq(batch.y).sum().item()
            total_examples += batch.y.size(0)
        test_acc = total_correct / total_examples

    cur_lr = [g["lr"] for g in optimiser.param_groups]

    new_best_str = ""
    stopping_threshold = 0.0001
    stopping_value = 0

    if train_acc > best_train_acc + stopping_threshold:
        best_train_acc = train_acc
        best_test_acc = test_acc
        best_epoch = epoch
        epochs_no_improve = 0
        stopping_value = train_acc
        new_best_str = " (new best train)"
    else:
        epochs_no_improve += 1
    print(
        f"Epoch {epoch}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Train acc: {train_acc:.4f}, Test accuracy: {test_acc:.4f}{new_best_str}"
    )
    if stopping_value == 1.0:
        break
    if epochs_no_improve >= patience:
        print(f"{patience} epochs without train acc improvement, stopping. ")
        break
print(f"Best train acc: {best_train_acc}, epoch: {best_epoch}")
