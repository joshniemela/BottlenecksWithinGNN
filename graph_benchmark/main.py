import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.transforms import NormalizeFeatures, Compose
from dataset import CitationDataset
from torch_geometric.data import Data
from fully_adjacent import fully_connect
from torch_geometric.nn.aggr import SoftmaxAggregation, SetTransformerAggregation
from torch import nn


class GCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        use_fully_adj=False,
    ):
        super(GCN, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_hidden_layers

        self.input_layer = GCNConv(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = GCNConv(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.use_fully_adj:
            if data.full_edge_index is None:
                raise ValueError(
                    "full_edge_index is None, please run fully_connect on the dataset"
                )
            full_edge_index = data.full_edge_index

        x = self.input_layer(x, edge_index)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)

        return self.output_layer(
            x, full_edge_index if self.use_fully_adj else edge_index
        )


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (Cora by default)
    dataset = CitationDataset("Cora", transform=fully_connect)

    data = dataset.get_data().to(device)

    # Initialize model
    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=128,
        output_dim=dataset.num_classes,
        num_hidden_layers=2,
        use_fully_adj=True,
    ).to(device)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Training loop
    for epoch in range(1000):
        loss = train(model, data, optimizer)
        if epoch % 20 == 0:
            test_acc = test(model, data)
            print(f"Epoch: {epoch:04d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
