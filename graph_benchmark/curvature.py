import torch
from copy import deepcopy
from torch import Tensor
from torch_geometric.data import Data


from contextlib import contextmanager


@contextmanager
def temporary_edge(nd, i, j):
    original_neighbors_i = nd.neighbour_dict[i].copy()
    original_neighbors_j = nd.neighbour_dict[j].copy()
    nd.add(i, j)
    try:
        yield
    finally:
        nd.neighbour_dict[i] = original_neighbors_i
        nd.neighbour_dict[j] = original_neighbors_j
        nd.memo = {}  # Clear memo cache


class NeighbourDict:
    def __init__(self, edge_index: torch.Tensor):
        self.neighbour_dict = {}
        self.memo = {}
        self.cache_misses = 0
        self.cache_hits = 0

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
        if (i, j) in self.memo:
            self.cache_hits += 1
            return self.memo[(i, j)]
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

        self.memo[(i, j)] = four_cycle_nodes
        self.cache_misses += 1

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
        if min(self.degree(i), self.degree(j)) == 1:
            return 0
        if len(self.four_cycles(i, j)) != 0:
            last_term = (
                1
                / (self.max_degeneracy(i, j) * max(self.degree(i), self.degree(j)))
                * (len(self.four_cycles(i, j)) - len(self.four_cycles(j, i)))
            )
        else:
            last_term = 0

        return (
            2 / self.degree(i)
            + 2 / self.degree(j)
            - 2
            + 2 * len(self.three_cycles(i, j)) / max(self.degree(i), self.degree(j))
            + len(self.three_cycles(i, j)) / min(self.degree(i), self.degree(j))
            + last_term
        )


class SDRF:
    def __init__(self, edge_index: torch.Tensor):
        self.nd = NeighbourDict(edge_index)
        self.ricci_curvature = {}
        self.pending_edges = []
        for i, j in zip(edge_index[0], edge_index[1]):
            self.pending_edges.append((i.item(), j.item()))

    def sync(self):
        """
        Synchronises the Ricci curvatures of the graph
        This must be called after modifying the graph
        """
        # reset the memo to avoid caching the wrong values
        # TODO: make so we don't nuke the entire cache each time
        while len(self.pending_edges) > 0:
            i, j = self.pending_edges.pop()
            self.ricci_curvature[i, j] = self.nd.ricci_curvature(i, j)
        self.nd.memo = {}

    def add_edge(self, i, j):
        # All the neighbours of neighbours of i or j might have changed due to this modification
        # TODO: figure out what the receptive field is of add and remove
        neighbours = self.nd.neighbours({i, j})
        # add all the edges that are connected to these neighbours
        for k in neighbours:
            for l in self.nd[k]:
                self.pending_edges.append((k, l))

        # This edge now exists and has an undefined Ricci curvature
        self.nd.add(i, j)
        self.pending_edges.append((i, j))

    def remove_edge(self, i, j):
        # All the neighbours of neighbours of i or j might have changed due to this modification
        # TODO: figure out what the receptive field is of add and remove
        neighbours = self.nd.neighbours({i, j})
        # add all the edges that are connected to these neighbours
        for k in neighbours:
            for l in self.nd[k]:
                self.pending_edges.append((k, l))

        # This edge no longer exists
        self.nd.remove(i, j)
        del self.ricci_curvature[i, j]

    def receptive_field(self, i, r):
        """
        Returns the set of all reachable nodes from i within radius r
        """
        reachable_nodes = {i}
        for _ in range(r):
            reachable_nodes = self.nd.neighbours(reachable_nodes)
        return reachable_nodes

    def preprocess(self, r, temperature, c_max, max_iter):
        """
        Performs the SDRF algorithm on the graph given the receptive field radius,
        temperature, uppper bound on the Ricci curvature, and max iterations
        """
        for _ in range(max_iter):
            # Find the edge with minimal Ricci curvature
            min_rc = float("inf")
            min_edge = (None, None)
            for i, j in self.ricci_curvature:
                if self.ricci_curvature[i, j] < min_rc:
                    min_rc = self.ricci_curvature[i, j]
                    min_edge = (i, j)

            print(f"New minimum: {min_edge}, Ricci curvature: {min_rc}")

            #  x is our vector of improvements
            x = []
            edges = []
            i_receptive_field = self.receptive_field(min_edge[0], 1)
            j_receptive_field = self.receptive_field(min_edge[1], 1)
            print(f"length of i_receptive_field: {len(i_receptive_field)}")
            print(f"length of j_receptive_field: {len(j_receptive_field)}")
            for n in i_receptive_field:
                for m in j_receptive_field:
                    if n != m:
                        edges.append((n, m))
            print(f"length of edges: {len(edges)}")

            i, j = min_edge

            for k, l in edges:
                old_rc = self.nd.ricci_curvature(k, l)
                with temporary_edge(self.nd, k, l):
                    new_rc = self.nd.ricci_curvature(k, l)
                    x.append(new_rc - old_rc)

            return x


import time, dataset

citation_data = dataset.CitationDataset("PubMed")
sdrf = SDRF(citation_data.get_data().edge_index)
start = time.time()
sdrf.sync()
print(time.time() - start)
