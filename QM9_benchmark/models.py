import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv
from gcn_mlp import GCNMLPConv, GCNLSTMConv


class GCN_regressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        use_fully_adj=False,
        aggregator_mode=None,
    ):
        super(GCN_regressor, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_layers

        self.embedding_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if aggregator_mode == "mlp":
                self.layers.append(GCNMLPConv(hidden_dim, hidden_dim))
            elif aggregator_mode == "lstm":
                self.layers.append(GCNLSTMConv(hidden_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.embedding_layer(x)  # Embed

        for i, layer in enumerate(self.layers):
            if self.use_fully_adj and i == len(self.layers) - 1:
                edge_index = self.fully_adj_edges(x, batch)
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norms[i](x)

        x = self.output_layer(x)
        x = global_mean_pool(x, batch)
        return x

    def fully_adj_edges(self, x, batch):
        # Get unique batch indices
        unique_batches = batch.unique()

        # Prepare lists to hold the edge indices
        edge_indices = []

        # Iterate over each unique batch
        for b in unique_batches:
            # Get the indices of the nodes in the current batch
            mask = batch == b
            indices = mask.nonzero(as_tuple=True)[
                0
            ]  # Get the indices of the nodes in the batch

            # Create edges for the fully connected subgraph of the current batch
            num_nodes = indices.size(0)
            if num_nodes > 1:
                # Create a fully connected graph for the current batch
                row_indices = indices.repeat(num_nodes)
                col_indices = indices.view(-1, 1).expand(-1, num_nodes).flatten()
                edge_indices.append(torch.stack([row_indices, col_indices], dim=0))

        # Concatenate all edge indices
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=x.device
            )  # No edges case

        return edge_index
