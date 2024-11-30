import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch.nn import Linear, Parameter
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from typing import Final, Tuple, Union
from torch_geometric.nn.pool import global_mean_pool
from torch import tanh
from torch import nn


class NonLinearWeighter(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, bias=True):
        super(NonLinearWeighter, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=bias),
            nn.Tanh(),  # Nonlinear activation
            nn.Linear(hidden_dim, out_channels, bias=bias),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.model:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x):
        return self.model(x)


class GlobalSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.neighbour_lin = Linear(in_channels, out_channels, bias=False)
        self.self_lin = Linear(in_channels, out_channels, bias=bias)

        # Global node representation
        self.global_weighter = NonLinearWeighter(
            2 * in_channels, 512, in_channels, bias=bias
        )
        self.projector = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.neighbour_lin.reset_parameters()
        self.self_lin.reset_parameters()
        self.global_weighter.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        out = self.propagate(edge_index, x=x, size=size)
        out = self.neighbour_lin(out)

        x_r = x[1]
        if x_r is not None:
            out += self.self_lin(x_r)

        # global_node = global_mean_pool(x, batch)
        global_node = torch.mean(x, dim=0, keepdim=True)

        # Each node gets the global node added scaled by the weighter
        global_weights = self.global_weighter(
            torch.cat([x, global_node.expand_as(x)], dim=-1)
        )

        out = out + self.projector(global_weights)

        return out
