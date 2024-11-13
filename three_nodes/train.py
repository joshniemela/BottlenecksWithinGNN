from dataset import generate_three_nodes_dataset
from models import GCN
from torch_geometric.loader import DataLoader
import torch

from torch.optim.adam import Adam
import argparse

# Hyperparameters
learning_rate = 0.1
num_epochs = 100

# set the seed for reproducibility
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


# Main function
def main(normalize=False):
    # Generate a dataset of 1000 graphs
    data_list = generate_three_nodes_dataset(1000)
    data = DataLoader(data_list, batch_size=512, shuffle=True)
    data_list = generate_three_nodes_dataset(1000)
    test_data = DataLoader(data_list, batch_size=512, shuffle=True)
    mse_losses = []

    print("Training model, normalise =", normalize)
    for _ in range(5):
        # Create model
        model = GCN(normalise=normalize)

        # Define optimizer and loss function
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        for epoch in range(num_epochs):
            loss = train(model, data, optimizer, criterion)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

        # Calculate the MSE loss on the test data
        total_loss = 0
        with torch.no_grad():
            for batch in test_data:
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
        mse_losses.append(total_loss / len(test_data))

    # print model parameters
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(name, param)

    # print the mean, median and standard deviation of the MSE loss
    print("MSE losses:")
    print(f"Max MSE loss of the model: {max(mse_losses)}")
    print(f"Mean MSE loss of the model: {sum(mse_losses) / len(mse_losses)}")
    print(f"Median MSE loss of the model: {sorted(mse_losses)[len(mse_losses) // 2]}")
    print(f"Standard deviation of the MSE loss: {torch.std(torch.tensor(mse_losses))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GCN model on a three nodes dataset"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Apply normalization to the model"
    )
    args = parser.parse_args()
    main(normalize=args.normalize)
