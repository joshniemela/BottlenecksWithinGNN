import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        use_fully_adj=False,
        mlp=False,
    ):
        super(GCN, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_layers

        self.embedding_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if mlp:
                self.layers.append(GCNMLPConv(hidden_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Assuming x contains 5 features in the first 5 columns
        x = self.embedding_layer(x)  # Embed into 32-dimensional space

        for i, layer in enumerate(self.layers):
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norms[i](x)

        x = self.output_layer(x)
        x = global_mean_pool(x, batch)
        return x
