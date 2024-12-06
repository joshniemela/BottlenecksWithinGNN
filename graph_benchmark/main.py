import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.transforms import NormalizeFeatures, Compose
from dataset import CitationDataset
from torch_geometric.data import Data
from fully_adjacent import fully_connect
from torch_geometric.nn.aggr import SoftmaxAggregation, SetTransformerAggregation
from torch import nn
from new_fully_adjacent import GlobalSAGEConv
import csv
from pathlib import Path


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

        if self.use_fully_adj:
            self.output_layer = GlobalSAGEConv(hidden_dim, output_dim, bias=True)
        else:
            self.output_layer = GCNConv(hidden_dim, output_dim, bias=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_layer(x, edge_index)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)

        return self.output_layer(x, edge_index)


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


def add_run_to_csv(
    dataset_name, use_fully_adj, num_hidden_layers, learning_rate, score, time_taken
):
    csv_file_path = f"results/{dataset_name}.csv"
    csv_file = Path(csv_file_path)

    if not csv_file.exists():
        with open(csv_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Dataset",
                    "Use Fully Adj",
                    "Num Hidden Layers",
                    "Learning Rate",
                    "Score",
                    "Time Taken",
                ]
            )

    with open(csv_file, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                dataset_name,
                use_fully_adj,
                num_hidden_layers,
                learning_rate,
                score,
                time_taken,
            ]
        )


import time
import random


def sweep_all():
    param_tuples = []
    for dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        for use_fully_adj in [True, False]:
            for num_hidden_layers in [1, 2, 3]:
                for learning_rate in [0.001, 0.002, 0.005, 0.001]:
                    for i in range(15):
                        param_tuples.append(
                            (
                                dataset_name,
                                use_fully_adj,
                                num_hidden_layers,
                                learning_rate,
                            )
                        )

    # randomise
    random.shuffle(param_tuples)

    j = 0
    start = time.time()
    for param_tuple in param_tuples:
        sweep(*param_tuple)
        j += 1
        # compute eta till finished
        eta = (time.time() - start) / (j + 1) * (1080 - j) / 60
        print(f"{j} of 1080 done, eta: {eta:.2f} minutes")


def sweep(dataset_name, use_fully_adj, num_hidden_layers, learning_rate):
    print(
        f"Sweeping {dataset_name} with FA:{use_fully_adj} and {num_hidden_layers} hidden layers and learning rate {learning_rate}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (Cora by default)
    dataset = CitationDataset(dataset_name)

    data = dataset.get_data().to(device)

    # Initialize model
    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=128,
        output_dim=dataset.num_classes,
        num_hidden_layers=num_hidden_layers,
        use_fully_adj=use_fully_adj,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
    )

    best_acc = 0
    epochs_no_improvement = 0
    # Training loop
    start_time = time.time()
    for epoch in range(2000):
        loss = train(model, data, optimizer)
        test_acc = test(model, data)
        if test_acc > best_acc:
            best_acc = test_acc
            epochs_no_improvement = 0
            # print(f"Epoch: {epoch:04d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}")
        else:
            epochs_no_improvement += 1
        if epochs_no_improvement == 300:
            break

    time_taken = time.time() - start_time
    print(
        f"Test Acc: {best_acc:.4f}, Params: {dataset_name}, {use_fully_adj}, {num_hidden_layers}, {learning_rate}"
    )
    add_run_to_csv(
        dataset_name,
        use_fully_adj,
        num_hidden_layers,
        learning_rate,
        best_acc,
        time_taken,
    )


if __name__ == "__main__":
    sweep_all()
