from typing import List
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time
import torch.nn as nn

class PersistentHomologyFiltrationUnionFind:
    def __init__(self, sorted_indices: torch.Tensor) -> None:
        self.parent = torch.arange(sorted_indices.shape[1]).repeat(
            sorted_indices.shape[0], 1
        )
        self.rank = torch.zeros_like(self.parent)
        row_indices = (
            torch.arange(sorted_indices.shape[1])
            .view(1, -1)
            .repeat(sorted_indices.shape[0], 1)
        )
        self.rank.scatter_(1, sorted_indices, row_indices)

    def find(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of the `find` operation for multiple nodes.

        Args:
            nodes: A 1D tensor of node indices to find the root for.

        Returns:
            A 1D tensor of root nodes corresponding to the input nodes.
        """
        while True:
            # Identify the parents of the current nodes
            parents = self.parent[nodes[0], nodes[1]]

            # Check if all nodes are their own parents (i.e., they are roots)
            done = parents == nodes[1, :]

            if done.all():
                break

            # Update the parent for path compression
            self.parent[nodes] = self.parent[parents]

            # Move to the next set of nodes (their parents)
            nodes = parents

        return nodes

    def union(self, nodes1: torch.Tensor, nodes2: torch.Tensor):
        roots1 = self.find(nodes1).clone()  # Find roots for all nodes in nodes1
        roots2 = self.find(nodes2).clone()  # Find roots for all nodes in nodes2

        # Create a mask for where the roots are different
        different_roots_mask = roots1[1, :] != roots2[1, :]

        if different_roots_mask.any():  # Check if there are any different roots
            # Get the ranks of the roots
            ranks1 = self.rank[roots1[0], roots1[1]]
            ranks2 = self.rank[roots2[0], roots2[1]]

            # Create a mask for the union by rank
            rank_mask = ranks1 < ranks2

            # Update parents based on the rank mask
            self.parent[roots2[0, :][rank_mask], roots2[1, :][rank_mask]] = roots1[1, :][
                rank_mask
            ]  # root2 becomes child of root1 where rank1 < rank2
            self.parent[roots1[0, :][~rank_mask], roots1[1, :][~rank_mask]] = roots2[1, :][
                ~rank_mask
            ]  # root1 becomes child of root2 where rank1 >= rank2

        return roots1, roots2, roots1[1, :]


class TOGL(nn.Module):
    """
    Implementation of TOGL, a topological graph layer.

    Some caveats: this implementation only focuses on a set function
    aggregation of topological features.
    """

    def __init__(
        self,
        n_features: int,
        n_filtrations: int,
        hidden_dim: int,
        out_dim: int,
        aggregation_fn,
    ):
        super().__init__()
        self.n_filtrations = n_filtrations

        # Neural network to compute filtration values
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_filtrations),
        )

    def filter_graph(self, X: torch.Tensor) -> torch.Tensor:
        """
        Filters the node features.
        Args:
            X: torch.tensor of shape (n_nodes, n_features)
        Returns:
            torch.tensor of shape (n_nodes, n_filtrations)
        """
        return self.filtrations(X)

    def generate_persistence_diagram_dim0(
        self, X: torch.Tensor, edge_list: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Generates 0-dimensional persistence diagrams using the updated PersistenceDiagram class.

        Args:
            X: Node features of shape (n_filtrations, n_nodes).
            edge_list: Edge list of shape (2, n_edges).

        Returns:
            List of tensors, each representing the persistence diagram for one filtration.
        """
        n_nodes = X.shape[1]
        n_filtrations = X.shape[0]
        persistence_diagrams = []

        _, indices = torch.sort(
            X, dim=1
        )  # Sort nodes by filtration value for current filtration

        uf = PersistentHomologyFiltrationUnionFind(
            indices
        )  # Initialize Union-Find for connected components
        pd = PersistenceDiagram(
            n_nodes, n_filtrations
        )  # Initialize the persistence diagram

        active_nodes = torch.zeros_like(X, dtype=torch.bool)
        pbar = tqdm(total=n_nodes, desc="Filtration", leave=True)

        for step, vertecies in enumerate(indices.t()):

            active_nodes[torch.arange(vertecies.shape[0]), vertecies] = True
            pd.step_counter = step

            for filtration, vertex in enumerate(vertecies):
                neighbors = edge_list[1][
                    edge_list[0] == vertex
                ] 

                active_neighbors = neighbors[active_nodes[filtration][neighbors]]

                for neighbor in active_neighbors:
                    neighbor_index = torch.tensor([ [filtration], [neighbor] ])
                    vertex_index = torch.tensor([ [filtration], [vertex] ])
                    pd.merge_components(vertex_index, neighbor_index, indices, uf)

            pbar.update(1)

        pbar.close()

        # Finalize the persistence diagram and convert component lifetimes to tensor
        pd.finalize()
        persistence_diagrams.append(pd.component_lifetimes.clone())

        return persistence_diagrams


class PersistenceDiagram:
    def __init__(self, n_nodes: int, n_filtrations: int) -> None:
        """
        Initialize the persistence diagram.

        Args:
            n_nodes: Number of nodes in the graph.
            n_filtrations: Number of filtration dimensions to track.
        """
        # Lifetimes for each dimension and filtration
        self.component_lifetimes = torch.zeros(
            (n_filtrations, n_nodes, 2), dtype=torch.long
        )
        self.component_lifetimes[:, :, 0] = (
            torch.arange(n_nodes).unsqueeze(0).repeat(n_filtrations, 1)
        )
        self.step_counter = torch.zeros(1, dtype=torch.long)

    def merge_components(
        self,
        nodes1: torch.Tensor,
        nodes2: torch.Tensor,
        indices: torch.Tensor,
        union_find: PersistentHomologyFiltrationUnionFind,
    ):
        """
        Vectorized version of merge_components to handle batches of nodes.

        Args:
            nodes1: A tensor of node indices in the first component.
            nodes2: A tensor of node indices in the second component.
            indices: A tensor containing sorted node indices by filtration order.
            union_find: A PersistentHomologyFiltrationUnionFind instance.
        """

        # Update the Union-Find structure
        roots1, roots2, new_roots = union_find.union(nodes1, nodes2)

        # Create a mask for where the roots are different
        different_roots_mask = roots1 != roots2

        if different_roots_mask.any():  # Check if there are any different roots

            # Determine absorbed roots
            absorbed_roots = torch.where(new_roots == roots2, roots1, roots2)

            # Update the persistence diagram
            absorbed_indexes = torch.where(indices == absorbed_roots.unsqueeze(1))
            self.component_lifetimes[absorbed_indexes] = self.step_counter

    def finalize(self):
        """
        Finalize the persistence diagram by marking all remaining components with death at infinity.
        """
        self.component_lifetime[self.component_lifetime[:, :, 1] == 0] = (
            self.step_counter
        )
