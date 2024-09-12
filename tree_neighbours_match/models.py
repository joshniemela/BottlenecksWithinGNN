import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv


def create_fully_adjacent_edges(num_nodes, num_trees):
    """
    Creates an edge list which connects every single node to the root for each tree
    Args:
        num_nodes (int): The number of nodes in each binary tree.
        num_trees (int): The number of binary trees (batch size).
    Returns:
        torch.Tensor: A tensor of shape (2, num_nodes * num_trees) representing the edge list.
    """
    # We need to connect all nodes to a root, therefore we can just count from 1 to num_nodes for each tree
    source_nodes = torch.arange(num_nodes * num_trees)
    # Floor division to get the root node for each tree
    target_nodes = num_nodes * (source_nodes // num_nodes)
    return torch.stack([source_nodes, target_nodes])


class GCN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, use_fully_adj=False
    ):
        super(GCN, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_layers

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

        # Total number of nodes in a single tree
        num_nodes = (x.shape[0] / (data.batch[-1] + 1)).int().item()
        num_batches = batch[-1] + 1

        # We set the roots mask to be 1, 0,..., 1, 0...
        # where 1 is the root and 0 is not a root.
        # this is needed because batching means we have multiple graphs in one batch
        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True
        roots_mask = root_mask.repeat(num_batches)

        x_classes, x_neighbours = x[:, 0], x[:, 1]
        x_classes = self.classes_embedding(x_classes)
        x_neighbours = self.neighbor_num_embedding(x_neighbours)
        x = x_classes + x_neighbours

        for i, layer in enumerate(self.layers):
            # Use fully adjacent edges if specified
            edge_index = (
                create_fully_adjacent_edges(num_nodes, num_batches).to(x.device)
                if self.use_fully_adj and i + 1 == self.num_layers
                else edge_index
            )
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norms[i](x)

        root_nodes = x[roots_mask]
        logits = self.output_layer(root_nodes)

        return logits


class GGNN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, use_fully_adj=False
    ):
        super(GGNN, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_layers

        self.classes_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hidden_dim
        )
        self.neighbor_num_embedding = nn.Embedding(
            num_embeddings=input_dim + 1, embedding_dim=hidden_dim
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedGraphConv(hidden_dim, 1))

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Total number of nodes in a single tree
        num_nodes = (x.shape[0] / (data.batch[-1] + 1)).int().item()
        num_batches = batch[-1] + 1

        # We set the roots mask to be 1, 0,..., 1, 0...
        # where 1 is the root and 0 is not a root.
        # this is needed because batching means we have multiple graphs in one batch
        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True
        roots_mask = root_mask.repeat(num_batches)

        x_classes, x_neighbours = x[:, 0], x[:, 1]
        x_classes = self.classes_embedding(x_classes)
        x_neighbours = self.neighbor_num_embedding(x_neighbours)
        x = x_classes + x_neighbours

        for i, layer in enumerate(self.layers):
            # Use fully adjacent edges if specified
            edge_index = (
                create_fully_adjacent_edges(num_nodes, num_batches).to(x.device)
                if self.use_fully_adj and i + 1 == self.num_layers
                else edge_index
            )
            x = layer(x, edge_index)

        root_nodes = x[roots_mask]
        logits = self.output_layer(root_nodes)

        return logits


class GIN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, use_fully_adj=False
    ):
        super(GIN, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_layers

        self.classes_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hidden_dim
        )
        self.neighbor_num_embedding = nn.Embedding(
            num_embeddings=input_dim + 1, embedding_dim=hidden_dim
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    )
                )
            )

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Total number of nodes in a single tree
        num_nodes = (x.shape[0] / (data.batch[-1] + 1)).int().item()
        num_batches = batch[-1] + 1

        # We set the roots mask to be 1, 0,..., 1, 0...
        # where 1 is the root and 0 is not a root.
        # this is needed because batching means we have multiple graphs in one batch
        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True
        roots_mask = root_mask.repeat(num_batches)

        x_classes, x_neighbours = x[:, 0], x[:, 1]
        x_classes = self.classes_embedding(x_classes)
        x_neighbours = self.neighbor_num_embedding(x_neighbours)
        x = x_classes + x_neighbours

        for i, layer in enumerate(self.layers):
            # Use fully adjacent edges if specified
            edge_index = (
                create_fully_adjacent_edges(num_nodes, num_batches).to(x.device)
                if self.use_fully_adj and i + 1 == self.num_layers
                else edge_index
            )
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
            x = self.layer_norms[i](x)

        root_nodes = x[roots_mask]
        logits = self.output_layer(root_nodes)

        return logits


class GAT(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, use_fully_adj=False
    ):
        super(GAT, self).__init__()
        self.use_fully_adj = use_fully_adj
        self.num_layers = num_layers

        self.classes_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hidden_dim
        )
        self.neighbor_num_embedding = nn.Embedding(
            num_embeddings=input_dim + 1, embedding_dim=hidden_dim
        )

        num_heads = 4
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
            )

        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Total number of nodes in a single tree
        num_nodes = (x.shape[0] / (data.batch[-1] + 1)).int().item()
        num_batches = batch[-1] + 1

        # We set the roots mask to be 1, 0,..., 1, 0...
        # where 1 is the root and 0 is not a root.
        # this is needed because batching means we have multiple graphs in one batch
        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True
        roots_mask = root_mask.repeat(num_batches)

        x_classes, x_neighbours = x[:, 0], x[:, 1]
        x_classes = self.classes_embedding(x_classes)
        x_neighbours = self.neighbor_num_embedding(x_neighbours)
        x = x_classes + x_neighbours

        for i, layer in enumerate(self.layers):
            # Use fully adjacent edges if specified
            edge_index = (
                create_fully_adjacent_edges(num_nodes, num_batches).to(x.device)
                if self.use_fully_adj and i + 1 == self.num_layers
                else edge_index
            )
            x_new = layer(x, edge_index)
            x = x + x_new
            x = self.layer_norms[i](x)

        root_nodes = x[roots_mask]
        logits = self.output_layer(root_nodes)

        return logits
