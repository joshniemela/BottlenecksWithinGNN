from model import TOGL

import torch


x = torch.rand(7, 2)
edge_list = torch.tensor(
    [
        [2, 4, 1, 4, 5, 4, 6, 1, 2, 3, 2, 6, 3, 5],
        [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6],
    ]
)

togl = TOGL(2, 2, 2, 2, "sum")

nfiltrations = togl.n_filtrations
print(togl.generate_persistence_diagram_dim0(x, edge_list))
