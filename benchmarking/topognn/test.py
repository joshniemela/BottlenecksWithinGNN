from model import TOGL

import torch
from torch_geometric.datasets import Planetoid


x = torch.rand(2, 6)
edge_list = torch.tensor(
    [
        [1, 3, 0, 3, 4, 3, 5, 0, 1, 2, 1, 5, 2, 4],
        [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    ]
)

togl = TOGL(2, 2, 2, 2, "sum")

print(togl.generate_persistence_diagram_dim0(x, edge_list))

dataset = Planetoid(root="/tmp/PubMed", name="PubMed")
data = dataset[0]

print(data)

togl = TOGL(500, 2, 2, 2, "sum")
print(togl.generate_persistence_diagram_dim0(data.x, data.edge_index))
