import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv


class GCN(nn.Module):
    def __init__(self, normalise=False, bias=True):
        super(GCN, self).__init__()
        self.conv = GCNConv(1, 1, normalize=normalise, bias=bias)
        self.interediate_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )
        self.conv2 = GCNConv(1, 1, normalize=normalise, bias=bias)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x_prev = self.conv(x, edge_index)

        x = self.interediate_mlp(torch.cat([x_prev, x_prev], dim=1))

        x = x.view(-1, 5)

        return x[:, 0]
