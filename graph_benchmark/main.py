import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from dataset import CitationDataset


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (Cora by default)
    dataset = CitationDataset("Cora")
    data = dataset.get_data().to(device)

    # Initialize model
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
    ).to(device)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Training loop
    for epoch in range(500):
        loss = train(model, data, optimizer)
        if epoch % 20 == 0:
            test_acc = test(model, data)
            print(f"Epoch: {epoch:04d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
