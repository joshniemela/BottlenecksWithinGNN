import torch.nn as nn
import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, normalise=False, bias=True):
        super(GCN, self).__init__()
        self.conv = GCNConv(1, 1, normalize=normalise, bias=bias)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        return x[:, 0]


class SAGE(nn.Module):
    def __init__(self, normalise=False):
        super(SAGE, self).__init__()
        self.conv = SAGEConv(1, 1, normalize=normalise, bias=False, aggr="sum")

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        return x[:, 0]


class NonLinearSAGE(nn.Module):
    def __init__(self, normalise=False, activation="mlp"):
        super(NonLinearSAGE, self).__init__()
        self.conv = SAGEConv(1, 1, normalize=normalise, bias=False, aggr="sum")

        if activation == "gaussian":
            self.activation = lambda x: torch.exp(-torch.pow(x, 2))
        elif activation == "simple":
            self.activation = lambda x: 1 - torch.pow(x, 2)
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "mlp":
            self.activation = nn.Sequential(
                nn.Linear(1, 2),
                nn.ReLU(),
                nn.Linear(2, 1),
            ) 

        # self.conv.lin_r.weight.data.fill_(1)
        # self.conv.lin_l.weight.data.fill_(-1)
        # self.conv.lin_r.weight.requires_grad = False
        # self.conv.lin_l.weight.requires_grad = False

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        out = self.activation(x[:, 0].view(-1, 1))

        return out.view(-1)
