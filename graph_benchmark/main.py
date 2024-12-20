import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset import CitationDataset
from torch import nn
from fully_adjacent import GlobalSAGEConv
import csv
from pathlib import Path
from bayes_opt import BayesianOptimization
from safetensors.torch import save_model
import time
import uuid


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


def base_model(dataset, num_hidden_layers, use_fully_adj):
    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=128,
        output_dim=dataset.num_classes,
        num_hidden_layers=num_hidden_layers,
        use_fully_adj=use_fully_adj,
    )
    return model


def evaluate_model(num_hidden_layers, use_fully_adj, max_params, dataset, num_runs=15):
    epochs = round(10 ** max_params["log_epochs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset.get_data().to(device)

    runs = []
    for _ in range(num_runs):
        model = base_model(dataset, num_hidden_layers, use_fully_adj).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=10 ** max_params["log_learning_rate"],
            weight_decay=10 ** max_params["log_C"],
        )

        print("Starting run")
        start = time.time()
        for _ in range(epochs):
            train(model, data, optimizer)
        end = time.time()
        test_acc = test(model, data)
        runs.append(
            {
                "test_acc": test_acc,
                "time_taken": end - start,
                "model": model,
            }
        )
        print(f"Test accuracy: {test_acc:.4f}, time: {end - start:.2f}s")
    return runs


def bayesian_sweep(num_hidden_layers, dataset_name, use_fully_adj, num_runs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CitationDataset(dataset_name)

    data = dataset.get_data().to(device)

    pbounds = {
        "log_epochs": (2, 3.5),
        "log_learning_rate": (-4, -1),
        "log_C": (-4, -2),
    }

    def objective(log_epochs, log_learning_rate, log_C):
        model = base_model(dataset, num_hidden_layers, use_fully_adj).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=10**log_learning_rate, weight_decay=10**log_C
        )
        for _ in range(round(10**log_epochs)):
            train(model, data, optimizer)
        val_acc = val(model, data)

        epoch_loss = 4e-7 * 10**log_epochs
        return val_acc - epoch_loss

    bo = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
    )
    bo.maximize(
        init_points=5,
        n_iter=15,
    )

    max_params = bo.max["params"]

    runs_data = evaluate_model(
        num_hidden_layers,
        use_fully_adj,
        max_params,
        dataset,
        num_runs,
    )
    for run in runs_data:
        run["dataset_name"] = dataset_name
        run["use_fully_adj"] = use_fully_adj
        run["num_hidden_layers"] = num_hidden_layers
        run["log_learning_rate"] = max_params["log_learning_rate"]
        run["log_C"] = max_params["log_C"]
        run["log_epochs"] = max_params["log_epochs"]
    return runs_data


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
                    "use_fully_adj",
                    "num_hidden_layers",
                    "learning_rate",
                    "epochs",
                    "C",
                    "score",
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
                    run["use_fully_adj"],
                    run["num_hidden_layers"],
                    10 ** run["log_learning_rate"],
                    round(10 ** run["log_epochs"]),
                    10 ** run["log_C"],
                    run["test_acc"],
                    run["time_taken"],
                ]
            )
            save_model(run["model"], f"results/{run_id}.safetensors")


if __name__ == "__main__":
    for dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        for use_fully_adj in [False, True]:
            for num_hidden_layers in [1, 2, 3]:
                runs = bayesian_sweep(
                    num_hidden_layers, dataset_name, use_fully_adj, 15
                )
                save_runs(runs)
