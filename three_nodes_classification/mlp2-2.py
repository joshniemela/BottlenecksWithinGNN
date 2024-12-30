import torch.nn as nn
import torch
from torch_geometric.nn.conv import SAGEConv

class NonLinearSAGE(nn.Module):
    def __init__(self, normalise=False):
        super(NonLinearSAGE, self).__init__()
        self.conv = SAGEConv(1, 1, normalize=normalise, bias=False, aggr="sum")

        self.activation = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        ) 

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        out = self.activation(x[:, 0].view(-1, 1))

        return out.view(-1)
