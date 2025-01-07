import argparse
import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import GCNWithTOGL  # Assuming the model is saved in a separate file
from dataset import (
    generate_tree,
    train_test_split,
)  # Assuming the data functions are saved separately


def train_gcn_togl(args):
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )

    # Data Preparation
    print("Generating data...")
    data_list = generate_tree(args.num_trees, args.tree_depth, device=device)
    train_data, test_data = train_test_split(data_list, train_ratio=args.train_ratio)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = GCNWithTOGL(
        2**args.tree_depth + 1, args.output_classes, args.hidden_channels, args.tree_depth + 1
    ).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Training Loop
    print("Training started...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss / len(train_loader):.4f}"
        )

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

    # Save Model
    if args.save_model:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GCN with TOGL on synthetic binary tree data."
    )

    # Dataset arguments
    parser.add_argument(
        "--num_trees", type=int, default=1000, help="Number of trees to generate."
    )
    parser.add_argument(
        "--tree_depth", type=int, default=3, help="Depth of the binary trees."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training dataset ratio."
    )

    # Model arguments
    parser.add_argument(
        "--output_classes", type=int, default=10, help="Number of output classes."
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=32, help="Number of hidden channels."
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")

    # Save arguments
    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="gcn_with_togl.pth",
        help="Path to save the model.",
    )

    args = parser.parse_args()
    train_gcn_togl(args)
