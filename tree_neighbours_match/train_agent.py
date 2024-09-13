"""
This module contains the code for training and evaluating a GCN (Graph Convolutional Network) model using the wandb (Weights and Biases) integration and sweep for hyperparameter search.

Functions:
- train_eval_model: Trains and evaluates the GCN model.
- train_eval_gcn_with_wandb: Wrapper function for wandb integration and sweep.
"""

from models import model_dict
import dataset as ds
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import cuda
from torch_geometric.loader import DataLoader
import wandb
import yaml
import os
import argparse

device = "cuda" if cuda.is_available() else "cpu"


# Debugger is more happy if we use the CPU
# device = "cpu"


def train_eval_model(
    model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, config: dict
):
    """
    Trains and evaluates a model using the given train and evaluation data loaders.
    Args:
        model (nn.Module): The model to train and evaluate.
        train_loader (DataLoader): The data loader for training data.
        eval_loader (DataLoader): The data loader for evaluation data.
        config (dict): A dictionary containing configuration parameters.
    Returns:
        None
    """

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

    max_epochs = config["max_epochs"]
    early_stop_patience = config["early_stop_patience"]
    early_stop_grace_period = config["early_stop_grace_period"]
    stopping_threshold = config["stopping_threshold"]
    total_train_samples = len(train_loader.dataset)
    total_eval_samples = len(eval_loader.dataset)

    # Some variables to keep track of the best model
    # for the purpose of early stopping
    best_train_acc = 0.0
    epochs_no_improve = 0

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
        else:
            print(
                {
                    "depth": config["num_layers"] - 1,
                    "epoch": epoch,
                    "train_loss": avg_training_loss,
                    "train_accuracy": train_acc,
                    "eval_accuracy": eval_acc,
                    "learning_rate": optimiser.param_groups[0]["lr"],
                }
            )

        if train_acc > best_train_acc + stopping_threshold:
            best_train_acc = train_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (
            epochs_no_improve >= early_stop_patience and epoch > early_stop_grace_period
        ) or train_acc == 1.0:
            break

    return model, train_acc, eval_acc


def create_dataset_train_eval_model():
    # Load YAML file
    with open(
        os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        + "/sweep_config.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file).get("parameters")

    # for each parameter extract the list of values
    for key, value in config.items():
        config[key] = value["values"]

    # For each config with more than one list element, choose the first element
    for key, value in config.items():
        if isinstance(value, list):
            config[key] = value[0]

    # Define model
    model = model_dict[config["model_type"]](
        2 ** config["tree_depth"] + 1,
        config["hidden_dim"],
        2 ** config["tree_depth"] + 1,
        config["tree_depth"] + 1,  # number of layers
        use_fully_adj=config["fully_adjacent_last_layer"],
        mlp=config["mlp_aggregation"],
    ).to(device)

    # Prepare data (replace with actual data loading mechanism)
    train_data, test_data = ds.train_test_split(
        ds.generate_tree(config["num_trees"], config["tree_depth"], device)
    )

    # Print devices
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], pin_memory=False
    )
    eval_loader = DataLoader(
        test_data, batch_size=config["batch_size"], pin_memory=False
    )

    # Configuration for training
    training_config = {
        "max_epochs": config["max_epochs"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["tree_depth"] + 1,
        "early_stop_patience": config["early_stop_patience"],
        "early_stop_grace_period": config["early_stop_grace_period"],
        "stopping_threshold": config["stopping_threshold"],
        "scheduler_factor": config["scheduler_factor"],
        "scheduler_patience": config["scheduler_patience"],
        "use_wandb": False,  # Flag to enable wandb logging
    }

    # Run general training function
    train_eval_model(model, train_loader, eval_loader, training_config)

    # Finish wandb run
    wandb.finish()


model_dict = {"GAT": GAT, "GCN": GCN, "GGNN": GGNN, "GIN": GIN}

# Sweep Configuration for hyperparameter search
sweep_config = {
    "method": "grid",
    "metric": {"name": "train_accuracy", "goal": "maximize"},
    "parameters": {
        "model_type": {"values": ["GCN", "GGNN", "GIN", "GAT"]},
        "tree_depth": {"values": [2, 3, 4]},
        "num_trees": {"values": [40000]},
        "epochs": {"values": [1000]},
        # Initial learning rate (intentionally too high)
        "lr": {"values": [0.01]},
        # FIXME: setting this to 4096 causes a bizarre error
        "batch_size": {"values": [2048]},
        "hidden_dim": {"values": [32]},
        "early_stop_patience": {"values": [20]},
        "early_stop_grace_period": {"values": [50]},
        "stopping_threshold": {"values": [0.0001]},
        # If scheduler runs out of patience, it multiplies current LR by this factor
        "scheduler_factor": {"values": [0.75]},
        # Number of epochs to wait before reducing LR
        "scheduler_patience": {"values": [5]},
        "fully_adjacent_last_layer": {"values": [False]},
    },
}

parser = argparse.ArgumentParser("Sweep agent")
parser.add_argument("--id", type=str, default="None")
parser.add_argument("--project", type=str, default="Uncategorized")
parser.add_argument("--key", type=str, default="None")
args = parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.id == "None":
        raise ValueError("Please provide a sweep ID")
    if args.key != "None":
        wandb.login(key=args.key)

    # Run the sweep
    wandb.agent(args.id, function=train_eval_gcn_with_wandb, project=args.project)
