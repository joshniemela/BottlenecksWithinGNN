import torch.nn as nn
from torch_geometric.nn.conv import GraphConv, GCNConv


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv = GCNConv(1, 1, normalize=False, bias=False)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)

        x = x.view(-1, 3)

        return x[:, 0]
