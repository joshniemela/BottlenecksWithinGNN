from torch_geometric.nn import MessagePassing
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.aggr import MLPAggregation, LSTMAggregation


class GCNLSTMConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            aggr=LSTMAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        )
        self.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = sort_edge_index(edge_index, num_nodes=x.size(0), sort_by_row=False)

        norm = edge_index.new_ones((edge_index.size(1),))
        out = self.propagate(edge_index, x=x, norm=norm)

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNMLPConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            aggr=MLPAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=32,
                max_num_elements=10,
                num_layers=2,
            )
        )
        # self.lin = Linear(in_channels, out_channels, bias=False)
        # self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin.reset_parameters()
    #     self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index = sort_edge_index(edge_index, num_nodes=x.size(0), sort_by_row=False)

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # no normalization
        norm = edge_index.new_ones((edge_index.size(1),))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out  # + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
