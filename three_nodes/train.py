from dataset import generate_three_nodes_dataset
from models import GCN
from torch_geometric.loader import DataLoader
from safetensors.torch import save_model
import uuid
import torch
import csv
from pathlib import Path

from torch.optim.adam import Adam
import argparse

# Hyperparameters
learning_rate = 0.1
num_epochs = 100


# Training function
def train(model, data_loader, optimizer, criterion):
    """Train the model on the given data loader.
    This function retuns the average loss value over the training data.
    """
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# Main function
def main(normalise=False, num_runs=15):
    # Generate a dataset of 1000 graphs
    train_data_list = generate_three_nodes_dataset(1000)
    train_data = DataLoader(train_data_list, batch_size=512, shuffle=True)
    test_data_list = generate_three_nodes_dataset(1000)
    test_data = DataLoader(test_data_list, batch_size=512, shuffle=True)

    print("Training model, normalise =", normalise)
    runs = []
    for _ in range(num_runs):
        # Create model
        model = GCN(normalise=normalise)

        # Define optimizer and loss function
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        for epoch in range(num_epochs):
            loss = train(model, train_data, optimizer, criterion)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

        # Calculate the MSE loss on the test data
        total_loss = 0
        with torch.no_grad():
            for batch in test_data:
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()

        runs.append(
            {
                "test_loss": total_loss / len(test_data),
                "normalise": normalise,
                "weight": model.conv.lin.weight.item(),
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "model": model,
            }
        )

    return runs


def save_runs(runs):
    csv_file_path = "results/runs.csv"
    csv_file = Path(csv_file_path)
    if not csv_file.exists():
        with open(csv_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "test_loss",
                    "normalise",
                    "weight",
                    "learning_rate",
                    "epochs",
                ]
            )

    with open(csv_file_path, "a") as f:
        writer = csv.writer(f)
        for run in runs:
            run_id = str(uuid.uuid4())
            writer.writerow(
                [
                    run_id,
                    run["test_loss"],
                    run["normalise"],
                    run["weight"],
                    run["learning_rate"],
                    run["epochs"],
                ]
            )
            save_model(run["model"], f"results/{run_id}.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GCN model on a three nodes dataset"
    )
    parser.add_argument(
        "--normalise", action="store_true", help="Apply normalisation to the model"
    )
    args = parser.parse_args()
    run_results = main(normalise=args.normalise)
    save_runs(run_results)
