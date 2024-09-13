import wandb

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

if __name__ == "__main__":
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="TreeBottleneckReproduction")
    print(f"Starting sweep with ID: {sweep_id}")
