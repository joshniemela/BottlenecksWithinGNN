from typing import List
from torch import nn
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time


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

    def generate_persistence_diagram_dim0(self, X: torch.Tensor, edge_list: torch.Tensor) -> torch.Tensor:
        """
        Generates the 0-dimensional persistence diagrams.
        Args:
            X: torch.tensor of shape (n_nodes, n_filtrations)
            edge_list: torch.tensor of shape (2, n_edges)
        Returns:
            List[torch.Tensor] with each tensor representing the persistence diagram for one filtration.
        """
        n_nodes = X.shape[0]
        persistence_diagrams = []

        for i in range(self.n_filtrations):
            # Sort nodes by filtration value for the current filtration
            start_time = time.time()
            _, indices = torch.sort(X[:, i])

            # Initialize empty graph
            graph_vertices = torch.zeros(n_nodes, dtype=torch.bool)
            graph_edges = torch.empty(0, 2, dtype=torch.long)

            persistence_diagram = PersistenceDiagram()
            step = 0

            part_times = {'sort_nodes': 0, 'add_vertex': 0, 'update_edges': 0, 'connected_components': 0}
            total_time = 0

            pbar = tqdm(total=n_nodes, desc=f"Filtration {i}", position=0, leave=True)
            for j in range(n_nodes):
                iteration_start = time.time()

                # Add the vertex to the graph
                vertex_start = time.time()
                vertex = indices[j].item() + 1
                graph_vertices[vertex - 1] = True
                part_times['add_vertex'] += time.time() - vertex_start

                # Add edges to the graph if both vertices are present
                edges_start = time.time()
                mask = edge_list[1] == vertex
                neighbors = edge_list[0][mask]
                for neighbor in neighbors:
                    if graph_vertices[neighbor - 1]:
                        graph_edges = torch.cat(
                            [graph_edges, torch.tensor([[vertex, neighbor]])]
                        )
                part_times['update_edges'] += time.time() - edges_start

                # Compute connected components
                cc_start = time.time()
                uf = UnionFind(n_nodes)
                for edge in graph_edges:
                    uf.union(edge[0] - 1, edge[1] - 1)  # Union using 0-based indexing
                components = uf.connected_components()
                part_times['connected_components'] += time.time() - cc_start

                # Update persistence diagram with current components
                persistence_diagram.step(components, step)
                step += 1

                iteration_end = time.time()
                total_time += iteration_end - iteration_start

                # Update the progress bar with runtime proportions
                time_percentages = {
                    k: (v / total_time) * 100 if total_time > 0 else 0
                    for k, v in part_times.items()
                }
                pbar.set_postfix({
                    f"{k} %": f"{v:.2f}" for k, v in time_percentages.items()
                })
                pbar.update(1)

            pbar.close()

            # Finalize the persistence diagram
            persistence_diagram.finalize()

            # Convert the component longevity dictionary to a tensor
            longevity_tensor = torch.tensor(
                [v for v in persistence_diagram.component_longivity.values()],
                dtype=torch.float32,
            )
            persistence_diagrams.append(longevity_tensor)

        return persistence_diagrams


class PersistenceDiagram:
    def __init__(self):
        self.live_component_dict = (
            {}
        )  # key: component-associated vertex, value: list of vertices in the component
        self.component_longivity = (
            {}
        )  # key: component-associated vertex, value: longevity tuple (birth, death)
        self.dead_component_dict = {}

    def step(self, connected_components: List[List[int]], step: int):
        # Update the birth and death times for each component
        for component in connected_components:
            root_vertex = component[0].item()
            if root_vertex not in self.live_component_dict:
                self.live_component_dict[root_vertex] = component
                self.component_longivity[root_vertex] = (step, step)
            else:
                self.component_longivity[root_vertex] = (
                    self.component_longivity[root_vertex][0],
                    step,
                )

        # Mark components as dead if they are no longer present
        live_vertices = set(self.live_component_dict.keys())
        current_vertices = {comp[0].item() for comp in connected_components}
        dead_vertices = live_vertices - current_vertices

        for vertex in dead_vertices:
            self.dead_component_dict[vertex] = self.live_component_dict[vertex]
            del self.live_component_dict[vertex]

        return self.component_longivity

    def finalize(self):
        # Mark all remaining components with longevity (birth, inf)
        for vertex in self.live_component_dict:
            self.component_longivity[vertex] = (
                self.component_longivity[vertex][0],
                float("inf"),
            )


class UnionFind:
    def __init__(self, n):
        self.parent = torch.arange(n, dtype=torch.long)  # Parent pointers
        self.rank = torch.zeros(n, dtype=torch.long)     # Rank for union by rank

    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by rank
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def connected_components(self):
        # Collect all components based on their root
        component_dict = {}
        for node in range(len(self.parent)):
            root = self.find(node)
            if root not in component_dict:
                component_dict[root] = []
            component_dict[root].append(node + 1)  # Convert to 1-based indexing
        return list(component_dict.values())
