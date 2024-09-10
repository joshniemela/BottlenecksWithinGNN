import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import cuda
from torch_geometric.loader import DataLoader
import dataset as ds
from models import GCN


device = "cuda" if cuda.is_available() else "cpu"

# Debugger is more happy if we use the CPU
device = "cpu"


def train_eval_model(
    model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, config: dict
):

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimiser = Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="max",
        threshold_mode="abs",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )

    max_epochs = config["epochs"]
    early_stop_patience = config["early_stop_patience"]
    early_stop_grace_period = config["early_stop_grace_period"]
    stopping_threshold = config["stopping_threshold"]
    total_train_samples = len(train_loader.dataset)
    total_eval_samples = len(eval_loader.dataset)

    # Some variables to keep track of the best model
    # for the purpose of early stopping
    best_train_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    print("Starting training")

    # Code takes inspiration from https://github.com/tech-srl/bottleneck
    # By Uri Alon and Eran Yahav
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

        # Log metrics (log only if wandb is in use)
        if config.get("use_wandb", False):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_training_loss,
                    "train_accuracy": train_acc,
                    "eval_accuracy": eval_acc,
                    "learning_rate": optimiser.param_groups[0]["lr"],
                }
            )

        if train_acc > best_train_acc + stopping_threshold:
            best_train_acc = train_acc
            best_epoch = epoch
            epochs_no_improve = 0
            new_best_train = True
        else:
            epochs_no_improve += 1
            new_best_train = False

        cur_lr = [g["lr"] for g in optimiser.param_groups]
        print(
            f"Epoch {epoch}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Train acc: {train_acc:.4f}, Test accuracy: {eval_acc:.4f}",
            "(new best train)" if new_best_train else "",
        )

        if (
            epochs_no_improve >= early_stop_patience and epoch > early_stop_grace_period
        ) or train_acc == 1.0:
            print(
                f"{early_stop_patience} epochs without train acc improvement, stopping. "
            )
            break
    print(f"Best train acc: {best_train_acc}, epoch: {best_epoch}")


# Wrapper function for wandb integration and sweep
def train_eval_gcn_with_wandb():
    wandb.init()

    # Define model
    model = GCN(
        2**wandb.config.tree_depth + 1,
        wandb.config.hidden_dim,
        2**wandb.config.tree_depth + 1,
        wandb.config.num_layers,
    )

    # Prepare data (replace with actual data loading mechanism)
    train_data, test_data = ds.train_test_split(
        ds.generate_tree(wandb.config.num_trees, wandb.config.tree_depth, device)
    )
    train_loader = DataLoader(
        train_data, batch_size=wandb.config.batch_size, pin_memory=True
    )
    eval_loader = DataLoader(
        test_data, batch_size=wandb.config.batch_size, pin_memory=True
    )

    # Configuration for training
    config = {
        "epochs": wandb.config.epochs,
        "lr": wandb.config.lr,
        "batch_size": wandb.config.batch_size,
        "hidden_dim": wandb.config.hidden_dim,
        "num_layers": wandb.config.num_layers,
        "early_stop_patience": wandb.config.early_stop_patience,
        "early_stop_grace_period": wandb.config.early_stop_grace_period,
        "stopping_threshold": wandb.config.stopping_threshold,
        "scheduler_factor": wandb.config.scheduler_factor,
        "scheduler_patience": wandb.config.scheduler_patience,
        "use_wandb": True,  # Flag to enable wandb logging
    }

    # Run general training function
    train_eval_model(model, train_loader, eval_loader, config)

    # Finish wandb run
    wandb.finish()


# Sweep Configuration for hyperparameter search
sweep_config = {
    "method": "bayes",
    "metric": {"name": "train_accuracy", "goal": "maximize"},
    "parameters": {
        "tree_depth": {"values": [3]},
        "num_trees": {"values": [10000]},
        "epochs": {"values": [1000]},
        "lr": {"values": [0.01]},
        "batch_size": {"values": [2048]},
        "hidden_dim": {"values": [32]},
        "num_layers": {"values": [4]},
        "early_stop_patience": {"values": [20]},
        "early_stop_grace_period": {"values": [50]},
        "stopping_threshold": {"values": [0.0001]},
        "scheduler_factor": {"values": [0.75]},
        "scheduler_patience": {"values": [5]},
    },
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="GCN_Model")

# Run the sweep
wandb.agent(sweep_id, function=train_eval_gcn_with_wandb, count=1)
