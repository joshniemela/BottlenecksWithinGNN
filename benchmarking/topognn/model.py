import hashlib
from typing import List
from torch import nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import aggr
from torch_geometric.nn import GCNConv
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
        aggregation_fn,
    ):
        super().__init__()
        self.n_filtrations = n_filtrations
        self.hidden_dim = hidden_dim

        # Neural network to compute filtration values
        self.filtrations_nn = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_filtrations),
        )
        
        theta_nn = nn.Linear(2*n_filtrations, hidden_dim)
        rho_nn = nn.Linear(1,n_features)
        self.deepSetLayer = aggr.DeepSetsAggregation(theta_nn, rho_nn)

        self.cache = {}

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
        persistence_diagrams = torch.zeros(X.shape[1], X.shape[0], 2)

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

            # Finalize the persistence diagram
            pd.finalize()
            persistence_diagrams[i, indices, :] = pd.component_lifetimes

        return persistence_diagrams.permute(1, 0, 2).reshape(n_nodes, 2*self.n_filtrations)
    
    def compute_hash(self, edge_list):
        # Hash the weights of the filtration neural network
        weights = b''.join(p.data.cpu().numpy().tobytes() for p in self.filtrations_nn.parameters())
        edge_list_bytes = edge_list.cpu().numpy().tobytes()
        combined = weights + edge_list_bytes
        return hashlib.sha256(combined).hexdigest()

    def forward(self, X, edge_list):

        # Compute hash key for caching
        cache_key = self.compute_hash(edge_list)

        # Check cache
        if cache_key not in self.cache:
            # Generate persistence diagrams if not cached
            filtrations = self.filtrations_nn(X)
            persitance_diagrams = self.generate_persistence_diagram_dim0(filtrations, edge_list)
            x_hat =  self.deepSetLayer.forward(persitance_diagrams, torch.zeros(self.hidden_dim, dtype=int), dim=1)
            self.cache[cache_key] = x_hat 

        return X +  self.cache[cache_key]


class GCNWithTOGL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_filtrations):
        super(GCNWithTOGL, self).__init__()
        self.n_layers = n_layers

        # Define GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # TOGL layer
        self.togl = TOGL(
            n_features=hidden_dim,
            n_filtrations=n_filtrations,
            hidden_dim=hidden_dim,
            aggregation_fn=aggr.MeanAggregation(),  # Aggregation function
        )

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass through GCN layers
        for i in range(self.n_layers):
            x = self.gcn_layers[i](x, edge_index)
            x = torch.relu(x)

        x = self.togl(x, edge_index)

        # Final output layer
        x = self.output_layer(x)

        return x
