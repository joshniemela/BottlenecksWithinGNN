"""
This module contains the code for training and evaluating a GCN (Graph Convolutional Network) model using the wandb (Weights and Biases) integration and sweep for hyperparameter search.

Functions:
- train_eval_model: Trains and evaluates the GCN model.
- train_eval_gcn_with_wandb: Wrapper function for wandb integration and sweep.
"""

import dataset as ds
from models import GCN
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from bayes_grid_search import BayesGridSearch


# device = "cuda" if cuda.is_available() else "cpu"

# Debugger is more happy if we use the CPU
device = "cpu"


def black_box_function(
    log_C, log_epochs, log_lr, mlp_aggregation, tree_depth, use_fully_adj
):
    hidden_dim = 128
    num_trees = 40000
    batch_size = 1024
    epochs = round(10**log_epochs)

    model = GCN(
        2**tree_depth + 1,
        hidden_dim,
        2**tree_depth + 1,
        tree_depth + 1,
        use_fully_adj=use_fully_adj,
        mlp=mlp_aggregation,
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimiser = Adam(model.parameters(), lr=10**log_lr, weight_decay=10**log_C)

    train_data, val_data = ds.train_test_split(
        ds.generate_tree(num_trees, tree_depth, device)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    total_val_samples = len(val_data)
    for epoch in range(epochs):
        # Training step
        model.train()
        optimiser.zero_grad()
        for batch in train_loader:
            batch = batch.to(device)
            train_preds = model(batch)
            loss = criterion(train_preds, batch.y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        print(f"Epoch {epoch} of {epochs} complete")

    model.eval()
    with torch.no_grad():
        total_test_correct = 0
        for batch in val_loader:
            batch = batch.to(device)
            val_preds = model(batch).argmax(dim=1)
            total_test_correct += val_preds.eq(batch.y).sum().item()
        val_acc = total_test_correct / total_val_samples

    epoch_loss = 4e-7 * epochs
    return val_acc - epoch_loss


p_bounds = {"log_epochs": (1, 3.5), "log_lr": (-4, -1), "log_C": (-4, -2)}

grid_params = {
    "tree_depth": (2, 5),
    "use_fully_adj": {True, False},
    "mlp_aggregation": {True, False},
}


if __name__ == "__main__":
    optimizer = BayesGridSearch(grid_params, p_bounds)
    results = optimizer.run(black_box_function)

    # get the next result until we run out of results
    while True:
        try:
            result = next(results)
            print(result)
        except StopIteration:
            break
