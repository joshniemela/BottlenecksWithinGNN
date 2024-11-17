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

    def degree(self, i):
        return len(self.neighbour_dict[i])

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

    def max_degeneracy(self, i, j):
        """
        Returns the maximal number of 4-cycles that traverse some common node on the path
        from i to j
        This is assumed to only be run if there are 4-cycles
        """

        i_four_cycle_nodes = self.four_cycles(i, j)
        j_four_cycle_nodes = self.four_cycles(j, i)

        max_degeneracy = 0

        # This will only check if a neighbour of i is appeaing multiple times
        for k in i_four_cycle_nodes:
            neighbours = self.neighbour_dict[k]
            # count how many neighbours of k are also in j_four_cycle_nodes
            degeneracy = len(neighbours & j_four_cycle_nodes)
            if degeneracy > max_degeneracy:
                max_degeneracy = degeneracy

        # We will therefore also have to check from the other direction
        # This works since we know the degeneracy should be symmetric
        for k in j_four_cycle_nodes:
            neighbours = self.neighbour_dict[k]
            degeneracy = len(neighbours & i_four_cycle_nodes)
            if degeneracy > max_degeneracy:
                max_degeneracy = degeneracy

        return max_degeneracy

    def ricci_curvature(self, i, j):
        """
        Returns the Ricci curvature of the edge (i, j)
        """
        return (
            2 / self.degree(i)
            + 2 / self.degree(j)
            - 2
            + 2 * len(self.three_cycles(i, j)) / max(self.degree(i), self.degree(j))
            + len(self.three_cycles(i, j)) / min(self.degree(i), self.degree(j))
            + (
                1
                / (self.max_degeneracy(i, j) * max(self.degree(i), self.degree(j)))
                * (len(self.four_cycles(i, j)) - len(self.four_cycles(j, i)))
                if len(self.four_cycles(i, j)) > 0
                else 0
            )
        )


def ricci_curvatures(edge_index: torch.Tensor) -> Tensor:
    neighbour_dict = NeighbourDict(edge_index)
    # for every edge in the edge_index, we compute the ricci curvature
    ricci_curvatures = []
    for i, j in zip(edge_index[0], edge_index[1]):
        ricci_curvatures.append(neighbour_dict.ricci_curvature(i.item(), j.item()))
    return torch.tensor(ricci_curvatures)


import time, dataset

citation_data = dataset.CitationDataset("Cora")
start = time.time()
ricci_curvatures(citation_data.get_data().edge_index)
print(time.time() - start)


def sdrf(data: Data) -> Data:
    assert data.edge_index is not None
    # dict is keyed on node contains [neighbour_nodess...]
    neighbour_dict = compute_neighbour_dict(data.edge_index)
