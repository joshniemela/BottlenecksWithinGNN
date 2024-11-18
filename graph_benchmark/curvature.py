import torch
from torch_geometric.data import Data


"""
    Find the number of triangles based at i ~ j, where i and j are the two nodes
    that are connected by the k-th edge in the edge_index.
"""


def n_triangles(edge_index, k):
    i = edge_index[0, k]
    j = edge_index[1, k]
    i_neighbours = edge_index[1, torch.where(edge_index[0] == i)[0]]
    j_neighbours = torch.where(edge_index[1] == j)[0]

    return torch.isin(i_neighbours, j_neighbours).sum()


# Instead of finding the count, find the nodes that are neighbours of i and j
def find_3_cycles(edge_index, k):
    i, j = edge_index[:, k]
    i_neighbours = edge_index[1, torch.where(edge_index[0] == i)[0]]
    j_neighbours = edge_index[1, torch.where(edge_index[0] == j)[0]]
    print(i_neighbours)
    print(j_neighbours)
    # find intersection of both neighbours
    return i_neighbours[torch.isin(i_neighbours, j_neighbours)]


"""
// k is the index of the edge in the edge list
// edge_idx is a matrix of size 2 x k
fun find_4_cycles(k: int, edge_idx: EdgeIndex): List[int] {
  let i = edge_idx[0, k];
  let j = edge_idx[1, k];

  let i_neighbours = neighbours(i, edge_idx);

  let triangle_nodes = find_3_cycles(k, edge_idx);

  i_neighbours
      .filter(|n| n != j)
      .filter(|n| !triangle_nodes.contains(n))
      .filter(|n| all(neighbours(n, edge_idx), |m| !triangle_nodes.contains(m)))
}
"""


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
