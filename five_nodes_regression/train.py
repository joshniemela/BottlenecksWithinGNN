import argparse
import csv
from pathlib import Path
import time
import uuid

from tqdm import tqdm
from dataset import generate_five_nodes_dataset
from models import GCN, SAGE
from torch_geometric.loader import DataLoader
from safetensors.torch import save_model
import torch
from torch.optim import AdamW
from torch.optim import Adam

# Set the seed for reproducibility
# torch.manual_seed(29)


# Training function
def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# Evaluation function to compute accuracy
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            diff = torch.abs(outputs - batch.y)
            predicted = diff < 0.5
            correct += predicted.sum().item()
            total += outputs.size(0)
    accuracy = 100 * correct / total
    return accuracy


# Main function
def main(
    model_type,
    samples,
    batch_size,
    learning_rate,
    num_epochs,
    num_runs,
    normalise=False,
):
    runs = []
    accuracies = []

    print(f"Training model: {model_type}, Normalise: {normalise}")

    for run in range(num_runs):
        train_data_list = generate_five_nodes_dataset(samples)
        test_data_list = generate_five_nodes_dataset(samples)
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)
        # Select model type based on user input
        if model_type == "GCN":
            model = GCN(normalise=normalise)
        elif model_type == "SAGE":
            model = SAGE(normalise=normalise)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'GCN', 'SAGE', or 'NonLinearSAGE'."
            )

        # Define optimizer and loss function
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        start = time.time()
        with tqdm(total=num_epochs, desc="Epochs", unit="epoch") as pbar:
            for _ in range(num_epochs):
                loss = train(model, train_loader, optimizer, criterion)

                # Update the progress bar
                pbar.set_postfix(loss=loss)
                pbar.update(1)
        end = time.time()

        # Evaluate accuracy on test set
        accuracy = evaluate(model, test_loader)
        accuracies.append(accuracy)
        print(f"Run {run+1}/{num_runs}, Accuracy: {accuracy:.2f}%")

        # prinitng the model parameters
        for name, param in model.named_parameters():
            print(name, param)

        runs.append(
            {
                "model": model,
                "dataset_name": "ThreeNodesClassification",
                "epochs": num_epochs,
                "accuracy": accuracy,
                "time_taken": end - start,
            }
        )

        save_runs(runs)

        runs = []

    # Metrics computation
    mean_accuracy = sum(accuracies) / len(accuracies)
    median_accuracy = sorted(accuracies)[len(accuracies) // 2]
    std_dev_accuracy = torch.std(torch.tensor(accuracies)).item()

    print("\nFinal Metrics Across Multiple Runs:")
    print(f"Best Accuracy: {max(accuracies):.2f}%")
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"Median Accuracy: {median_accuracy:.2f}%")
    print(f"Standard Deviation of Accuracy: {std_dev_accuracy:.2f}")


def save_runs(runs):
    csv_file_path = f"results/runs.csv"
    csv_file = Path(csv_file_path)
    if not csv_file.exists():
        with open(csv_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "dataset_name",
                    "epochs",
                    "accuracy",
                    "time_taken",
                ]
            )

    with open(csv_file_path, "a") as f:
        writer = csv.writer(f)
        for run in runs:
            run_id = str(uuid.uuid4())
            writer.writerow(
                [
                    run_id,
                    run["dataset_name"],
                    run["epochs"],
                    run["accuracy"],
                    run["time_taken"],
                ]
            )
            save_model(run["model"], f"results/{run_id}.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model on a dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=["GCN", "SAGE"],
        help="Model type to use",
    )
    parser.add_argument(
        "--normalise", action="store_true", help="Apply normalisation to the model"
    )
    parser.add_argument(
        "--samples", type=int, default=2000, help="Number of samples in the dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1028, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of training and evaluation runs"
    )
    args = parser.parse_args()

    # Pass parsed arguments to the main function
    main(
        model_type=args.model,
        normalise=args.normalise,
        samples=args.samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        num_runs=args.runs,
    )
