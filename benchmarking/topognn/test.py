from model import TOGL

import torch
from torch_geometric.datasets import Planetoid


x = torch.rand(6, 2)
edge_list = torch.tensor(
    [
        [2, 4, 1, 4, 5, 4, 6, 1, 2, 3, 2, 6, 3, 5],
        [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6],
    ]
)

togl = TOGL(2, 2, 2, 2, "sum")

print(togl.generate_persistence_diagram_dim0(x, edge_list))

dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
data = dataset[0]

print(data)

togl = TOGL(500, 2, 2, 2, "sum")
print(togl.generate_persistence_diagram_dim0(data.x, data.edge_index))
