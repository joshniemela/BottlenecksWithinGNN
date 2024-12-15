from model import TOGL, GCNWithTOGL

import torch
from torch_geometric.datasets import Planetoid


x = torch.rand(6, 2)
edge_list = torch.tensor(
    [
        [1, 3, 0, 3, 4, 3, 5, 0, 1, 2, 1, 5, 2, 4],
        [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    ]
)

togl = TOGL(2, 2, 6, "sum")

print(togl(x, edge_list))
print(togl(x, edge_list))

dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
data = dataset[0]

print(data)

# togl = TOGL(500, 2, 2, 2, "sum")
# print(togl(data.x, data.edge_index))
# print(togl(data.x, data.edge_index))



input_dim = 500 # Number of input features per node
hidden_dim = 32  # Hidden layer dimension
output_dim = 3  # Number of output classes
n_layers = 3     # Number of GCN layers
n_filtrations = 2  # Number of filtrations in TOGL
model = GCNWithTOGL(input_dim, hidden_dim, output_dim, n_layers, n_filtrations)

model(data)
model(data)