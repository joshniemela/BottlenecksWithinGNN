from typing import List
from torch import nn
from torch_geometric.data import Data
import torch


class TOGL(nn.Module):
    """Implementation of TOGL, a topological graph layer.

    Some caveats: this implementation only focuses on a set function
    aggregation of topological features. At the moment, it is not as
    powerful and feature-complete as the original implementation.
    """

    def __init__(
        self,
        n_features,
        n_filtrations,
        hidden_dim,
        out_dim,
        aggregation_fn,
    ):
        super().__init__()

        self.n_filtrations = n_filtrations

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_filtrations),
        )

    def filter_graph(self, X):
        """
        Args: X: torch.tensor of shape (n_nodes, n_features)
        output: torch.tensor of shape (n_nodes, n_filtrations)
        """
        return self.filtrations(X)

    def generate_persistence_diagram_dim0(self, X, edge_list):
        """
        Args: X: torch.tensor of shape (n_nodes, n_filtrations)
        output: torch.tensor of shape (n_nodes, persistence_dim0_for_filtrations)
        """

        n_nodes = X.shape[0]

        for i in range(self.n_filtrations):
            _, indices = torch, sorted(X[:, i])

            # initialize empty graph
            graph_vertices = []
            graph_edges = []

            for j in range(n_nodes):
                # we add the vertex to the graph
                graph_vertices.append(indices[j])

                # we add the edge to the graph if both vertices are in the graph
                if j in edge_list[1]:
                    index = edge_list[1].index(j)
                    if edge_list[0][index] in graph_vertices:
                        graph_edges.append((j, edge_list[0][index]))
