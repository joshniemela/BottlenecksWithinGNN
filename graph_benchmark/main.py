import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.transforms import NormalizeFeatures, Compose
from dataset import CitationDataset
from torch_geometric.data import Data
from torch_geometric.nn.aggr import SoftmaxAggregation, SetTransformerAggregation
from torch import nn
from fully_adjacent import GlobalSAGEConv
import csv
from pathlib import Path
from bayes_opt import BayesianOptimization
from safetensors import safe_open
from safetensors.torch import save_file


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


def val(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
    return val_acc


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


def evaluate_model(model, optimizer, data, epochs, num_runs=15):
    runs = []
    for _ in range(num_runs):
        start = time.time()
        for epoch in range(epochs):
            train(model, data, optimizer)
        end = time.time()
        test_acc = test(model, data)
        runs.append((test_acc, end - start, model.state_dict()))


def sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (Cora by default)
    dataset = CitationDataset("Cora")

    data = dataset.get_data().to(device)

    pbounds = {
        "log_epochs": (2, 4),
        "log_learning_rate": (-3, -0.5),
        "log_C": (-4, 0),
    }

    def base_model(num_hidden_layers):
        model = GCN(
            input_dim=dataset.num_features,
            hidden_dim=16,
            output_dim=dataset.num_classes,
            num_hidden_layers=round(num_hidden_layers),
            use_fully_adj=False,
        )
        return model

    def objective(log_epochs, log_learning_rate, log_C):
        model = base_model(1)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=10**log_learning_rate, weight_decay=10**log_C
        )
        for _ in range(round(10**log_epochs)):
            train(model, data, optimizer)
        val_acc = val(model, data)

        return val_acc

    bo = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
    )
    bo.maximize(
        init_points=5,
        n_iter=10,
    )

    print(bo.max)
    max_params = bo.max["params"]

    model = base_model(1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=10 ** max_params["log_learning_rate"],
        weight_decay=10 ** max_params["log_C"],
    )

    print(evaluate_model(model, optimizer, data, 10))


if __name__ == "__main__":
    sweep()
