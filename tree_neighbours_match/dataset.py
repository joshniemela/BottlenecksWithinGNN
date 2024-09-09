""" Generate binary trees with random values and random number of neighbours. """

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def generate_tree(n: int, depth: int, device="cpu") -> list[Data]:
    """
    Generate n binary trees with specified depth and number of trees.

    Args:
        depth (int): The depth of the binary trees.
        n (int): The number of binary trees to generate.

    Returns:
        Data: A Data object containing the generated binary trees.

    """
    # Generate n binary trees with 2^depth nodes
    num_nodes = 2 ** (depth + 1) - 1
    num_last_layer = 2**depth
    halfway_index = num_nodes // 2

    node_values = torch.empty(n, num_nodes, 2, dtype=torch.int, device=device)

    # Set the first half in each tree to 0s, this is the part of the tree with no leaves
    node_values[:, :halfway_index, :] = 0

    # Set the second half first slice to the values from 1 to num_last_layer + 1.
    # (We are enumerating the nodes on the last layer)
    node_values[:, halfway_index:, 0] = torch.arange(
        1, num_last_layer + 1, device=device
    )
    # Set the second half second slice to a random permutation from 1 to n
    # (We are setting the number of neighbours of the nodes on the last layer)
    node_values[:, halfway_index:, 1] = (
        torch.randint(
            torch.iinfo(torch.int64).max, (n, num_last_layer), device=device
        ).argsort()  # Random permutation
        + 1
    )

    # Pick a random row from the second of the now random tree
    random_row_indexes = torch.randint(
        halfway_index, num_nodes, (n, 1), device=device
    ).squeeze()
    random_rows = node_values[torch.arange(n, device=device), random_row_indexes]

    # We now set the y value to the class of this row
    # and we set the number of neighbours of the root node to the neighours of the row
    y = random_rows[:, 0]
    node_values[:, 0, 1] = random_rows[:, 1]

    # Create the edge list (it is a tree so we know it is n-1 edges)
    edge_index = torch.empty((2, num_nodes - 1), dtype=torch.int, device=device)
    edge_index[0, :] = torch.arange(1, num_nodes, device=device)
    edge_index[1, :] = (edge_index[0, :] - 1) // 2
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]
    edge_index = edge_index.repeat(n, 1, 1)

    # We may use the data_list directly in a DataLoader object,
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    data_list = [
        Data(x=node_values[i], y=y[i], edge_index=edge_index[i]) for i in range(n)
    ]

    return data_list


if __name__ == "__main__":
    # Generate 2 binary trees with depth 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    data = generate_tree(36000, 4, DEVICE)
    print(data[0].x, data[0].y, data[0].edge_index)
