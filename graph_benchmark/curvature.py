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


def n_4cycles(edge_index, k):
    i = edge_index[0, k]
    j = edge_index[1, k]
    i_neighbours = edge_index[1, torch.where(edge_index[0] == i)[0]]
    j_neighbours = torch.where(edge_index[1] == j)[0]
