from typing import List
from torch import nn
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time

class PersistentHomologyFiltrationUnionFind:
    def __init__(self, size: int, indices: torch.Tensor) -> None:
        self.parent = torch.arange(size)  # Initialize the Union-Find parent array
        self.rank = torch.zeros(size, dtype=torch.long)
        self.rank[indices] = torch.arange(size)

    def find(self, node: torch.Tensor) -> torch.Tensor:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])  # Path compression
        return self.parent[node]

    def union(self, node1: torch.Tensor, node2: torch.Tensor):
        root1 = self.find(node1).clone()
        root2 = self.find(node2).clone()

        if (root1 != root2):
            # Union by rank, but rank is based on the sorted indices
            if (self.rank[root1] < self.rank[root2]):
                self.parent[root2] = root1
            else:  # there is no need to check for equality, since the indices are unique
                self.parent[root1] = root2
        
        return root1, root2, self.parent[root1]


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

    def generate_persistence_diagram_dim0(self, X: torch.Tensor, edge_list: torch.Tensor) -> List[torch.Tensor]:
        """
        Generates 0-dimensional persistence diagrams using the updated PersistenceDiagram class.

        Args:
            X: Node features of shape (n_nodes, n_filtrations).
            edge_list: Edge list of shape (2, n_edges).

        Returns:
            List of tensors, each representing the persistence diagram for one filtration.
        """
        n_nodes = X.shape[0]
        persistence_diagrams = []

        for i in range(self.n_filtrations):
            _, indices = torch.sort(X[:, i])  # Sort nodes by filtration value for current filtration
            uf = PersistentHomologyFiltrationUnionFind(n_nodes, indices)  # Initialize Union-Find for connected components
            pd = PersistenceDiagram(n_nodes)  # Initialize the persistence diagram

            active_nodes = torch.zeros(n_nodes, dtype=torch.bool, device=X.device)
            pbar = tqdm(total=n_nodes, desc=f"Filtration {i}", leave=True)

            for step, vertex in enumerate(indices):
                active_nodes[vertex] = True
                pd.step_counter = step

                neighbors = edge_list[1][edge_list[0] == vertex] # Get neighbors of the current node
                active_neighbors = neighbors[active_nodes[neighbors]] # Filter out nodes not yet added to the filtration
                
                for neighbor in active_neighbors:
                    pd.merge_components(vertex, neighbor, indices, uf)

                pbar.update(1)

            pbar.close()

            # Finalize the persistence diagram and convert component lifetimes to tensor
            pd.finalize()
            persistence_diagrams.append(pd.component_lifetimes.clone())

        return persistence_diagrams


class PersistenceDiagram:
    def __init__(self, n_nodes: int):
        self.component_lifetimes = torch.stack((torch.arange(n_nodes), torch.zeros(n_nodes)), dim=1)  # key: root node, value: (birth, death)
        self.current_components = {}  # key: node, value: root node
        self.step_counter = 0  # Track the current step in the filtration


    def merge_components(self, node1: torch.Tensor, node2: torch.Tensor, indices: torch.Tensor, union_find: PersistentHomologyFiltrationUnionFind):
        """
        Merge two components. The component of node2 is absorbed into node1's component.
        """

        # Update the Union-Find structure
        root1, root2, new_root = union_find.union(node1, node2)

        if root1 == root2:
            return

        # Determine which component survives
        absorbed_root = root1 if new_root == root2 else root2

        # Update death time for the absorbed component
        self.component_lifetimes[torch.where(indices == absorbed_root)[0], 1] = self.step_counter


    def finalize(self):
        """
        Finalize the persistence diagram by marking all remaining components with death at infinity.
        """
        self.component_lifetimes[self.component_lifetimes[:,1] == 0, 1] = float("inf")
