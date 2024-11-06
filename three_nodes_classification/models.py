import torch.nn as nn
import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, normalise=False):
        super(GCN, self).__init__()
        self.conv = GCNConv(1, 1, normalize=normalise, bias=True)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        return x[:, 0]


class SAGE(nn.Module):
    def __init__(self, normalise=False):
        super(SAGE, self).__init__()
        self.conv = SAGEConv(1, 1, normalize=normalise, bias=True, aggr="sum")

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        return x[:, 0]


class NonLinearSAGE(nn.Module):
    def __init__(self, normalise=False):
        super(NonLinearSAGE, self).__init__()
        self.conv = SAGEConv(1, 1, normalize=normalise, bias=False, aggr="sum")

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        # apply gaussian function
        out = torch.exp(-torch.pow(x[:, 0], 2))

        return out
