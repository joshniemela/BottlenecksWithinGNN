import torch
from torch_geometric.data import Data


def generate_three_nodes_dataset(n: int, device="cpu") -> list[Data]:
    # Index 0 is our root, 1 and 2 are our two nodes on the sides
    edge_index = torch.tensor([[1, 2], [0, 0]])

    node_values = torch.empty(n, 3, dtype=torch.float, device=device)

    # Set the two side nodes to something between 0 and 10
    node_values[:, 1:] = torch.randint(10, (n, 2)).float()

    # Set the root node to the sum of the two side nodes
    node_values[:, 0] = torch.sum(node_values[:, 1:], 1)

    # we replace half of the root values with random values
    mask = torch.rand(n) > 0.5
    node_values[mask, 0] = torch.randint(10 * 2 - 1, (mask.sum(),)).float()

    # We need to reshape the tensor to have the correct shape
    node_values = node_values.reshape(n, 3, 1)

    # If node 0 is the sum of the other two nodes it is class 1, otherwise it is class 0
    # allow epsilon difference
    eps = 0.01
    y = torch.abs(node_values[:, 0] - (node_values[:, 1] + node_values[:, 2])) < eps
    # make 2d where the second dimension is 1 if the value is true and dimension 0  is 1 if the value is false
    y = y.to(torch.float)

    # We may use the data_list directly in a DataLoader object,
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    data_list = [
        Data(x=node_values[i], y=y[i], edge_index=edge_index) for i in range(n)
    ]

    return data_list
