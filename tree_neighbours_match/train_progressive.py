import os
import yaml
from torch import cuda
from torch_geometric.loader import DataLoader
import dataset as ds
import torch
from train import train_eval_model
from models import model_dict


device = "cuda" if cuda.is_available() else "cpu"

num_threads = 14
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

if __name__ == "__main__":
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

    # we do some override of the depth here since it is not a parameter for a progressive model
    config["tree_depth"] = 5

    # Define model
    model = model_dict[config["model_type"]](
        2 ** config["tree_depth"] + 1,
        config["hidden_dim"],
        2 ** config["tree_depth"] + 1,
        (config["tree_depth"] + 1) * 1,  # number of layers
        use_fully_adj=config["fully_adjacent_last_layer"],
        mlp=config["mlp_aggregation"],
    ).to(device)

    training = True

    while training:
        if config["tree_depth"] > 2:
            # Save the original model's state_dict
            previous_model_weights = model.state_dict()

            # Create a new model with more layers
            model = model_dict[config["model_type"]](
                2 ** config["tree_depth"] + 1,
                config["hidden_dim"],
                2 ** config["tree_depth"] + 1,
                (config["tree_depth"] + 1) * 1,  # number of layers
                use_fully_adj=config["fully_adjacent_last_layer"],
                mlp=config["mlp_aggregation"],
            ).to(device)

            # load from checkpoint if exists
            if os.path.exists("model_depth_{}.pt".format(config["tree_depth"])):
                model.load_state_dict(
                    torch.load("model_depth_{}.pt".format(config["tree_depth"]))
                )
                print("Loaded model from checkpoint")

        # Prepare data
        train_data, test_data = ds.train_test_split(
            ds.generate_tree(config["num_trees"] // 4, config["tree_depth"], device)
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
            "batch_size": 16 * config["batch_size"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["tree_depth"] + 1,
            "early_stop_patience": config["early_stop_patience"],
            "early_stop_grace_period": config["early_stop_grace_period"],
            "stopping_threshold": config["stopping_threshold"],
            "scheduler_factor": config["scheduler_factor"],
            "scheduler_patience": config["scheduler_patience"],
            "use_wandb": False,  # Flag to enable wandb logging
        }

        # Run general training function, through the output explicitly
        model, train_acc, _ = train_eval_model(
            model, train_loader, eval_loader, training_config
        )

        # We start a low depth, it is likely we converge to a solution here,
        # if we solve the depth, we continue to a deeper depth
        if train_acc < 0.99:
            training = False
            print("Training stopped as accuracy is below 95%")
            # save the model

        torch.save(model.state_dict(), "model_depth_{}.pt".format(config["tree_depth"]))
        config["tree_depth"] += 1
