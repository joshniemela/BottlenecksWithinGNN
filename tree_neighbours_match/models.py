import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()

        self.classes_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hidden_dim
        )
        self.neighbor_num_embedding = nn.Embedding(
            num_embeddings=input_dim + 1, embedding_dim=hidden_dim
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        num_nodes = (x.shape[0] / (data.batch[-1] + 1)).int().item()
        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True
        roots_mask = root_mask.repeat(batch[-1] + 1)

        x_classes, x_neighbours = x[:, 0], x[:, 1]
        x_classes = self.classes_embedding(x_classes)
        x_neighbours = self.neighbor_num_embedding(x_neighbours)
        x = x_classes + x_neighbours

        for i in range(len(self.layers)):
            x_new = self.layers[i](x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norms[i](x)

        root_nodes = x[roots_mask]
        logits = self.output_layer(root_nodes)

        return logits
