from typing import List
from torch import nn
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time

class UnionFind:
    def __init__(self, size, indices):
        self.parent = list(range(size))
        self.rank = [0] * size

        # Initialize the Union-Find rank based on the sorted indices
        for i, idx in enumerate(indices):
            self.rank[idx] = i

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])  # Path compression
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 != root2:
            # Union by rank
            if self.rank[root1] < self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1


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

    def dfs(
        self,
        vertex: int,
        visited: torch.Tensor,
        component: List[int],
        graph_edges: torch.Tensor,
    ):
        visited[vertex - 1] = True
        component.append(vertex)
        for edge in graph_edges:
            if edge[0] == vertex and not visited[edge[1] - 1]:
                self.dfs(edge[1], visited, component, graph_edges)
            if edge[1] == vertex and not visited[edge[0] - 1]:
                self.dfs(edge[0], visited, component, graph_edges)

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
            uf = UnionFind(n_nodes, indices)          # Initialize Union-Find for connected components
            pd = PersistenceDiagram()        # Initialize the persistence diagram

            active_nodes = set()  # Keep track of vertices added to the filtration
            pbar = tqdm(total=n_nodes, desc=f"Filtration {i}", leave=True)

            for step, idx in enumerate(indices):
                vertex = idx.item()          # Current vertex being added
                active_nodes.add(vertex)
                pd.step_counter = step

                # Add edges and merge components
                neighbors = edge_list[0][edge_list[1] == vertex]
                for neighbor in neighbors:
                    if neighbor.item() in active_nodes:
                        pd.merge_components(vertex, neighbor.item(), uf)

                # Collect connected components
                components = {}
                for node in active_nodes:
                    root = uf.find(node)
                    if root not in components:
                        components[root] = []
                    components[root].append(node)

                # Update persistence diagram
                pd.step(list(components.values()), step, uf)
                pbar.update(1)

            pbar.close()

            # Finalize the persistence diagram and convert component lifetimes to tensor
            pd.finalize()
            longevity_tensor = torch.tensor(
                [v for v in pd.component_lifetimes.values()],
                dtype=torch.float32,
            )
            persistence_diagrams.append(longevity_tensor)

        return persistence_diagrams


class PersistenceDiagram:
    def __init__(self):
        self.component_lifetimes = {}  # key: root node, value: (birth, death)
        self.current_components = {}  # key: node, value: root node
        self.step_counter = 0  # Track the current step in the filtration

    def add_component(self, node):
        """
        Add a new component for a node.
        """
        self.component_lifetimes[node] = (self.step_counter, float("inf"))
        self.current_components[node] = node

    def merge_components(self, node1, node2, union_find):
        """
        Merge two components. The component of node2 is absorbed into node1's component.
        """
        root1 = union_find.find(node1)
        root2 = union_find.find(node2)

        if root1 != root2:
            # Determine which component survives based on Union-Find rules
            surviving_root = min(root1, root2, key=lambda x: union_find.rank[x])
            absorbed_root = root2 if surviving_root == root1 else root1

            # Update death time for the absorbed component
            if absorbed_root in self.component_lifetimes:
                self.component_lifetimes[absorbed_root] = (
                    self.component_lifetimes[absorbed_root][0],
                    self.step_counter,
                )
            else:
                self.component_lifetimes[absorbed_root] = (
                    self.step_counter,
                    self.step_counter,
                )

            # Update the Union-Find structure
            union_find.union(root1, root2)

    def step(self, connected_components: List[List[int]], step: int, union_find):
        """
        Update the persistence diagram at the current step.
        """
        self.step_counter = step

        # Mark live components as active or merged
        active_roots = set()
        for component in connected_components:
            root = union_find.find(component[0])
            active_roots.add(root)
            if root not in self.component_lifetimes:
                self.add_component(root)

        # Update the lifetime of absorbed components
        for root in list(self.component_lifetimes.keys()):
            if root not in active_roots:
                if self.component_lifetimes[root][1] == float("inf"):
                    self.component_lifetimes[root] = (
                        self.component_lifetimes[root][0],
                        step,
                    )

    def finalize(self):
        """
        Finalize the persistence diagram by marking all remaining components with death at infinity.
        """
        for root, (birth, death) in self.component_lifetimes.items():
            if death == float("inf"):
                self.component_lifetimes[root] = (birth, float("inf"))
