from typing import List
from torch import nn
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time

class PersistentHomologyFiltrationUnionFind:
    def __init__(self, indices: torch.Tensor) -> None:
        self.parent = torch.arange(indices.shape[1]).repeat(indices.shape[0])
        self.rank = torch.zeros(indices.shape[0] * indices.shape[1], dtype=torch.long)
        self.rank[indices] = torch.arange(indices.shape[1]).repeat(indices.shape[0])

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
            parents = self.parent[nodes]
            
            # Check if all nodes are their own parents (i.e., they are roots)
            done = parents == nodes
            
            if done.all():
                break

            # Update the parent for path compression
            self.parent[nodes] = self.parent[parents]
            
            # Move to the next set of nodes (their parents)
            nodes = parents

        return nodes

    def union(self, nodes1: torch.Tensor, nodes2: torch.Tensor):
        roots1 = self.find(nodes1).clone()  # Find roots for all nodes in node1
        roots2 = self.find(nodes2).clone()  # Find roots for all nodes in node2

        # Create a mask for where the roots are different
        different_roots_mask = roots1 != roots2

        if different_roots_mask.any():  # Check if there are any different roots
            # Get the ranks of the roots
            ranks1 = self.rank[roots1]
            ranks2 = self.rank[roots2]

            # Create a mask for the union by rank
            rank_mask = ranks1 < ranks2 

            # Update parents based on the rank mask
            self.parent[roots2[rank_mask]] = roots1[rank_mask]  # root2 becomes child of root1 where rank1 < rank2
            self.parent[roots1[~rank_mask]] = roots2[~rank_mask]  # root1 becomes child of root2 where rank1 >= rank2
        
        return roots1, roots2, self.parent[roots1]


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
    def __init__(self, n_nodes: int, n_filtrations: int) -> None:
        """
        Initialize the persistence diagram.

        Args:
            n_nodes: Number of nodes in the graph.
            n_filtrations: Number of filtration dimensions to track.
        """
        # Lifetimes for each dimension and filtration
        self.component_lifetimes = torch.zeros((n_filtrations, n_nodes, 2), dtype=torch.long)
        self.component_lifetimes[:, :, 0] = torch.arange(n_nodes).unsqueeze(0).repeat(n_filtrations, 1)
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
            absorbed_roots = torch.where(
                new_roots == roots2, roots1, roots2
            )

            absorbed_roots_mask = absorbed_roots == roots1:
            self.






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
