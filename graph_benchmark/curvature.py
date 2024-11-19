import torch
from torch import Tensor
from torch_geometric.data import Data


def find_4_cycles(edge_index, k):
    i, j = edge_index[:, k]
    i_neighbours = edge_index[1, torch.where(edge_index[0] == i)[0]]
    triangle_nodes = find_3_cycles(edge_index, k)
    # Do not include j in the neighbours
    i_neighbours = i_neighbours[torch.where(i_neighbours != j)[0]]
    # Do not include any diagonal
    if len(triangle_nodes) > 0:
        i_neighbours = i_neighbours[torch.where(i_neighbours != triangle_nodes)[0]]
    # only include the neighbour if it has a neighbour that goes to j but not to the triangle nodes


def compute_neighbour_dict(edge_index: torch.Tensor) -> dict:
    neighbour_dict = {}
    for i in range(edge_index.max() + 1):
        neighbour_dict[i] = set()
    for i, j in zip(edge_index[0], edge_index[1]):
        neighbour_dict[int(i)].add(int(j))
    return neighbour_dict


class NeighbourDict:
    def __init__(self, edge_index: torch.Tensor):
        self.neighbour_dict = {}
        for i in range(edge_index.max().item() + 1):
            self.neighbour_dict[i] = set()
        for i, j in zip(edge_index[0], edge_index[1]):
            self.neighbour_dict[int(i)].add(int(j))

    def __getitem__(self, i):
        return self.neighbour_dict[i]

    def add(self, i, j):
        self.neighbour_dict[i].add(j)

    def remove(self, i, j):
        self.neighbour_dict[i].remove(j)

    def three_cycles(self, i, j):
        """
        Returns the neighbours of i that are also neighbours of j (3-cycles)
        """
        return self.neighbour_dict[i] & self.neighbour_dict[j]

    def neighbours(self, nodes: set):
        """
        Return all the nodes reachable from the given nodes
        """
        if len(nodes) == 0:
            return set()
        return set.union(*[self.neighbour_dict[i] for i in nodes])

    def four_cycles(self, i, j):
        """
        Returns the neighbours of i that are also neighbours of j (4-cycles)
        """
        # Bad nodes are all triangle nodes, i and j
        bad_nodes = self.three_cycles(i, j) | {i, j}

        i_neighbours = self.neighbour_dict[i]
        candidates = i_neighbours - bad_nodes

        four_cycle_nodes = set()
        for k in candidates:
            candidate_neighbours = self.neighbour_dict[k] - bad_nodes
            neighbours = self.neighbours(candidate_neighbours)
            if j in neighbours:
                four_cycle_nodes.add(k)

        return four_cycle_nodes


def sdrf(data: Data) -> Data:
    assert data.edge_index is not None
    # dict is keyed on node contains [neighbour_nodess...]
    neighbour_dict = compute_neighbour_dict(data.edge_index)
