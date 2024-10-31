import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self
    ):
        super(GCN, self).__init__()
        self.conv = GCNConv()

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
