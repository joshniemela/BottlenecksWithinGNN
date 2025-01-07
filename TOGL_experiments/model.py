"""Debug script for benchmarking some calculations."""

import time
import torch
import sys

import torch.nn as nn
from torch.nn import functional as F
from torch_topological.nn.graphs import TOGL
from torch_geometric.nn import GCNConv


class GCNWithTOGL(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super(GCNWithTOGL, self).__init__()
        self.num_layers = num_layers

        self.classes_embedding = nn.Embedding(
            num_embeddings=in_channels, embedding_dim=hidden_channels
        )
        self.neighbour_num_embedding = nn.Embedding(
            num_embeddings=in_channels + 1, embedding_dim=hidden_channels
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GCNConv(
                    hidden_channels,
                    hidden_channels,
                )
            )

        self.layer_norm = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norm.append(nn.LayerNorm(hidden_channels))

        self.tolg_layer = TOGL(
            hidden_channels, 16, hidden_channels, hidden_channels, "mean"
        )

        self.output_layer = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        num_nodes = (x.shape[0] / (data.batch[-1] + 1)).int().item()
        num_batches = batch[-1] + 1

        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True
        roots_mask = root_mask.repeat(num_batches)

        x_classes, x_neighbours = x[:, 0], x[:, 1]
        x_classes = self.classes_embedding(x_classes)
        x_neighbours = self.neighbour_num_embedding(x_neighbours)
        x = x_classes + x_neighbours

        for i, layer in enumerate(self.layers[:1]):
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norm[i](x)

        x = self.tolg_layer(x, data)

        for i, layer in enumerate(self.layers[1:]):
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norm[i](x)

        root_nodes = x[roots_mask]
        logits = self.output_layer(root_nodes)

        return logits
