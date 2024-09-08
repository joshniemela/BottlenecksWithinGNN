import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def generate_tree(depth: int) -> Data:
    # Generate a binary tree with 2^depth nodes
    num_nodes = 2 ** (depth + 1) - 1
    num_last_layer = 2**depth
    halfway_index = num_nodes // 2

    node_values = torch.empty(num_nodes, 2, dtype=torch.int)

    # Set the first half to 0s, this is the part of the tree with no leaves
    node_values[:halfway_index, :] = 0

    # Set the second half first slice to 1 up to n
    node_values[halfway_index:, 0] = torch.arange(1, num_last_layer + 1)
    # Set the second half second slice to a random permutation from 1 to n
    node_values[halfway_index:, 1] = torch.randperm(num_last_layer) + 1

    # Pick a random row from the second of the now random tree
    random_row = node_values[torch.randint(halfway_index, num_nodes, (1, 1)).squeeze()]
    print(random_row)

    # We now set the y value to the class of this row
    # and we set the number of neighbours of the root node to the neighours of the row
    y = random_row[0]
    node_values[0, 1] = random_row[1]

    # Create the edge list (it is a tree so we know it is n-1 edges)
    edge_index = torch.empty((2, num_nodes - 1), dtype=torch.int)
    edge_index[0, :] = torch.arange(1, num_nodes)
    edge_index[1, :] = (edge_index[0, :] - 1) // 2
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]

    return Data(x=node_values, y=y, edge_index=edge_index)
