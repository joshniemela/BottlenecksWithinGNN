from dataset import generate_three_nodes_dataset
from models import GCN
from torch_geometric.loader import DataLoader
import torch

import torch.optim as optim
from torch.optim.adam import Adam

# Hyperparameters
learning_rate = 0.02
num_epochs = 200


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
def main():
    # Generate a dataset of 1000 graphs
    data_list = generate_three_nodes_dataset(1000)
    data = DataLoader(data_list, batch_size=512, shuffle=True)

    # Create model
    model = GCN()

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print("Training model without normalisation...")
    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

    # print model parameters
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(name, param)

    model = GCN(normalise=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print("Training model with normalisation...")
    for epoch in range(num_epochs):
        loss = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

    # print model parameters
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(name, param)


if __name__ == "__main__":
    main()
