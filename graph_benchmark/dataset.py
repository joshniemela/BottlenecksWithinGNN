from torch_geometric.datasets import Planetoid
import torch


dataset = Planetoid("./datasets", "PubMed")
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class CitationDataset:
    """Handles loading of citation network datasets (Cora, CiteSeer, PubMed)"""
    
    def __init__(self, name, root='./data', transform=NormalizeFeatures()):
        """
        Args:
            name: Dataset name ('Cora', 'CiteSeer', or 'PubMed')
            root: Root directory for data storage
            transform: Data transform (default: NormalizeFeatures)
        """
        self.name = name
        self.dataset = Planetoid(root=root, name=name, transform=transform)
        self.data = self.dataset[0]
        
    def get_data(self):
        return self.data
    
    @property
    def num_features(self):
        return self.dataset.num_features
    
    @property
    def num_classes(self):
        return self.dataset.num_classes
