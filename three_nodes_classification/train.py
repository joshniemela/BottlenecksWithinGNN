import argparse
from dataset import generate_three_nodes_dataset
from models import GCN, SAGE, NonLinearSAGE
from torch_geometric.loader import DataLoader
import torch
from torch.optim import AdamW

# Set the seed for reproducibility
torch.manual_seed(0)


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
            predicted = (outputs > 0.5).float()
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Main function
def main(
    model_type="GCN",
    normalise=False,
    samples=1000,
    batch_size=64,
    learning_rate=0.01,
    num_epochs=100,
    num_runs=5,
):
    # Generate training and test datasets
    train_data_list = generate_three_nodes_dataset(samples)
    test_data_list = generate_three_nodes_dataset(samples)
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

    accuracies = []

    print(f"Training model: {model_type}, Normalise: {normalise}")

    for run in range(num_runs):
        # Select model type based on user input
        if model_type == "GCN":
            model = GCN(normalise=normalise)
        elif model_type == "SAGE":
            model = SAGE(normalise=normalise)
        elif model_type == "NonLinearSAGE":
            model = NonLinearSAGE(normalise=normalise)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'GCN', 'SAGE', or 'NonLinearSAGE'."
            )

        # Define optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, criterion)

        # Evaluate accuracy on test set
        accuracy = evaluate(model, test_loader)
        accuracies.append(accuracy)
        print(f"Run {run+1}/{num_runs}, Accuracy: {accuracy:.2f}%")

        # prinitng the model parameters
        for name, param in model.named_parameters():
            print(name, param)

    # Metrics computation
    mean_accuracy = sum(accuracies) / len(accuracies)
    median_accuracy = sorted(accuracies)[len(accuracies) // 2]
    std_dev_accuracy = torch.std(torch.tensor(accuracies)).item()

    print("\nFinal Metrics Across Multiple Runs:")
    print(f"Best Accuracy: {max(accuracies):.2f}%")
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"Median Accuracy: {median_accuracy:.2f}%")
    print(f"Standard Deviation of Accuracy: {std_dev_accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model on a dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=["GCN", "SAGE", "NonLinearSAGE"],
        help="Model type to use",
    )
    parser.add_argument(
        "--normalise", action="store_true", help="Apply normalisation to the model"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples in the dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1028, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of training epochs"
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
