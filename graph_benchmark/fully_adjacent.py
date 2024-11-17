import torch
from torch_geometric.data import Data


def find_neighbours(edge_index: torch.Tensor, idx: int) -> torch.Tensor:
    """Finds the neighbours of a node in an edge index"""
    return edge_index[1, edge_index[0] == idx]


def fully_connect(data: Data) -> Data:
    assert data.edge_index is not None
    assert data.num_nodes is not None

    # An index of what connected component a node belongs to
    connected_components = torch.zeros(data.num_nodes, dtype=torch.int)
    connected_components.fill_(-1)  # -1 is not-visited

    # Initialise with no components
    component_id = -1  # This gets incremented to 0 on the first iteration
    stack = []

    # DFS begins
    for node in range(data.num_nodes):
        if connected_components[node] == -1:
            # We haven't visited this node yet
            component_id += 1
            stack.append(node)

        while stack:
            stack_node = stack.pop()
            if connected_components[stack_node] != -1:
                continue

            connected_components[stack_node] = component_id

            # We all the not-yet-visited neighbours of this node to the stack
            neighbours = find_neighbours(data.edge_index, stack_node)
            for neighbour in neighbours:
                if connected_components[neighbour] == -1:
                    stack.append(neighbour)
    # DFS ends

    n_components = component_id + 1

    full_edges = []
    for component in range(n_components):
        component_nodes = torch.where(connected_components == component)[0].int()

        # This computes the cartesian product of a connected component
        i, j = torch.meshgrid(component_nodes, component_nodes, indexing="ij")
        edges = torch.stack((i.flatten(), j.flatten()), dim=0)
        full_edges.append(edges)

    data.full_edge_index = torch.cat(full_edges, dim=1).int()
    return data
