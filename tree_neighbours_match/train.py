import torch
import torch.nn as nn
from torch.optim import Adam
from torch import cuda
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dataset as ds
from models import GCN


# This should be set to an arbitrary high but not infinite value
max_epochs = 1000

early_stop_patience = 20

device = "cuda" if cuda.is_available() else "cpu"

# Debugger is more happy if we use the CPU
device = "cpu"

model = GCN(9, 32, 9, 4).to(device)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimiser = Adam(model.parameters(), lr=0.02)
scheduler = ReduceLROnPlateau(
    optimiser, mode="max", threshold_mode="abs", factor=0.75, patience=5
)
print("Starting training")

train_data, test_data = ds.train_test_split(ds.generate_tree(10000, 3, device))
train_loader = DataLoader(train_data, batch_size=2048, pin_memory=True)
eval_loader = DataLoader(test_data, batch_size=128, pin_memory=True)

total_train_samples = len(train_loader.dataset)
total_eval_samples = len(eval_loader.dataset)
best_test_acc = 0.0
best_train_acc = 0.0
best_epoch = 0
epochs_no_improve = 0


for epoch in range(max_epochs):
    # Training step
    model.train()
    total_loss = 0
    train_correct = 0
    optimiser.zero_grad()
    for batch in train_loader:
        batch = batch.to(device)
        train_preds = model(batch)
        loss = criterion(train_preds, batch.y)
        total_loss += loss.item()
        train_pred = train_preds.argmax(dim=1)
        train_correct += train_pred.eq(batch.y).sum().item()

        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

    avg_training_loss = total_loss / total_train_samples
    train_acc = train_correct / total_train_samples
    scheduler.step(train_acc)

    # Evaluation step
    model.eval()
    with torch.no_grad():
        total_test_correct = 0
        for batch in eval_loader:
            batch = batch.to(device)
            eval_preds = model(batch).argmax(dim=1)
            total_test_correct += eval_preds.eq(batch.y).sum().item()
        eval_acc = total_test_correct / total_eval_samples

    cur_lr = [g["lr"] for g in optimiser.param_groups]

    new_best_str = ""
    stopping_threshold = 0.0001

    if train_acc > best_train_acc + stopping_threshold:
        best_train_acc = train_acc
        best_test_acc = eval_acc
        best_epoch = epoch
        epochs_no_improve = 0
        new_best_train = True
    else:
        epochs_no_improve += 1
        new_best_train = False

    print(
        f"Epoch {epoch}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Train acc: {train_acc:.4f}, Test accuracy: {eval_acc:.4f}",
        "(new best train)" if new_best_train else "",
    )

    if epochs_no_improve >= early_stop_patience or train_acc == 1.0:
        print(f"{early_stop_patience} epochs without train acc improvement, stopping. ")
        break
print(f"Best train acc: {best_train_acc}, epoch: {best_epoch}")
