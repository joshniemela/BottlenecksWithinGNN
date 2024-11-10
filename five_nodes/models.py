import torch.nn as nn
from torch_geometric.nn.conv import GraphConv, GCNConv


class GCN(nn.Module):
    def __init__(self, normalise=False):
        super(GCN, self).__init__()
        self.conv = GCNConv(1, 1, normalize=normalise, bias=False)
        self.conv2 = GCNConv(1, 1, normalize=normalise, bias=False)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)
        x = self.conv2(x, edge_index)

        x = x.view(-1, 5)

        return x[:, 0]
