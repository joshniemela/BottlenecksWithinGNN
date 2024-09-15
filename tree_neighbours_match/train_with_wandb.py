from train import train_eval_model

import dataset as ds
from models import model_dict
from torch import cuda
from torch_geometric.loader import DataLoader
import wandb


device = "cuda" if cuda.is_available() else "cpu"


# Wrapper function for wandb integration and sweep
def train_eval_gcn_with_wandb():
    """
    Train and evaluate a GCN model using wandb for logging.
    This function initializes a wandb run, defines a GCN model, prepares the data, and runs the training and evaluation process.
    The configuration for training is set using the wandb.config parameters.
    Args:
        None
    Returns:
        None
    """

    wandb.init()

    # Define model
    model = model_dict[wandb.config.model_type](
        2**wandb.config.tree_depth + 1,
        wandb.config.hidden_dim,
        2**wandb.config.tree_depth + 1,
        wandb.config.tree_depth + 1,  # number of layers
        use_fully_adj=wandb.config.fully_adjacent_last_layer,
    ).to(device)

    # Prepare data (replace with actual data loading mechanism)
    train_data, test_data = ds.train_test_split(
        ds.generate_tree(wandb.config.num_trees, wandb.config.tree_depth, device)
    )
    # print devices
    train_loader = DataLoader(
        train_data, batch_size=wandb.config.batch_size, pin_memory=False
    )
    eval_loader = DataLoader(
        test_data, batch_size=wandb.config.batch_size, pin_memory=False
    )

    # Configuration for training
    config = {
        "max_epochs": wandb.config.max_epochs,
        "lr": wandb.config.lr,
        "batch_size": wandb.config.batch_size,
        "hidden_dim": wandb.config.hidden_dim,
        "num_layers": wandb.config.tree_depth + 1,
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


if __name__ == "__main__":
    # Run the sweep
    train_eval_gcn_with_wandb()
