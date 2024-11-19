from typing import List
from torch import nn
import torch
from torch_geometric.data import Data


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
        visited[vertex] = True
        component.append(vertex)
        for edge in graph_edges:
            if edge[0] == vertex and not visited[edge[1]]:
                self.dfs(edge[1], visited, component, graph_edges)
            if edge[1] == vertex and not visited[edge[0]]:
                self.dfs(edge[0], visited, component, graph_edges)

    def generate_persistence_diagram_dim0(
        self, X: torch.Tensor, edge_list: torch.Tensor
    ) -> torch.Tensor:
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
            _, indices = torch.sort(X[:, i])

            # Initialize empty graph
            graph_vertices = torch.zeros(n_nodes, dtype=torch.bool)
            graph_edges = torch.empty(0, 2, dtype=torch.long)

            persistence_diagram = PersistenceDiagram()
            step = 0

            for j in range(n_nodes):
                vertex = indices[j].item()

                # Add the vertex to the graph
                graph_vertices[vertex] = True

                # Add edges to the graph if both vertices are present
                mask = edge_list[1] == vertex
                neighbors = edge_list[0][mask]

                # Iterate through the neighbors
                for neighbor in neighbors:
                    if graph_vertices[neighbor]:
                        graph_edges = torch.cat(
                            [graph_edges, torch.tensor([[vertex, neighbor]])]
                        )

                # Compute connected components using DFS
                visited = torch.zeros(n_nodes, dtype=torch.bool)
                components = []
                print(indices[: j + 1])
                for k in indices[: j + 1]:
                    if not visited[k]:
                        component = []
                        self.dfs(k, visited, component, graph_edges)
                        if component:
                            components.append(component)

                # Update persistence diagram with current components
                persistence_diagram.step(components, step)
                step += 1

            # Finalize the persistence diagram
            persistence_diagram.finalize()

            # Convert the component longevity dictionary to a tensor
            longevity_tensor = torch.tensor(
                [v for v in persistence_diagram.component_longivity.values()],
                dtype=torch.float32,
            )
            persistence_diagrams.append(longevity_tensor)

        return torch.concat(persistence_diagrams, dim=0)


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
            root_vertex = component[0]
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
        current_vertices = {comp[0] for comp in connected_components}
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
