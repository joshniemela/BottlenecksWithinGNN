import torch
from torch_geometric.data import Data


def generate_three_nodes_dataset(n: int, device="cpu") -> list[Data]:
    # Index 0 is our root, 1 and 2 are our two nodes on the sides
    edge_index = torch.tensor([[1, 2], [0, 0]])

    node_values = torch.empty(n, 3, dtype=torch.int, device=device)

    # Our root node is set to 0, this is just to give us a neutral node
    node_values[:, 0] = 0

    # Set the two side nodes to something between 0 and 1000
    node_values[:, 1:] = torch.randint(1000, (n, 2))

    node_values = node_values.reshape(n, 3, 1).float()

    # Find the sum of each graph, we skip the root, this is technically unnessecary since its set to 0
    y = torch.sum(node_values[:, 1:], 1)

    # We may use the data_list directly in a DataLoader object,
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    data_list = [
        Data(x=node_values[i], y=y[i], edge_index=edge_index) for i in range(n)
    ]

    return data_list
