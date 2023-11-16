import torch 
from packages.RatSPN.spn.rat_spn import make_rat

class WeightedMean(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, weight_dims, normalize_dim):
        super(WeightedMean, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(size=tuple(weight_dims)))
        self.normalize_dim = normalize_dim
        self.softmax = torch.nn.Softmax(dim=normalize_dim)
        
    def forward(self, x):
        x = (x*self.softmax(self.weights)).sum(self.normalize_dim )
        return x


class RatSPN(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0):
        super(RatSPN, self).__init__()
        self.spn = make_rat(num_features=num_features, classes=classes, leaves=leaves, sums=sums, num_splits=num_splits, dropout=dropout)
        
    def forward(self, x):
        return self.spn(x.permute(1,0,2))
    
class EinsumNet(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0):
        super(WeightedMean, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(size=tuple(weight_dims)))
        self.normalize_dim = normalize_dim
        self.softmax = torch.nn.Softmax(dim=normalize_dim)
        
    def forward(self, x):
        x = (x*self.softmax(self.weights)).sum(self.normalize_dim )
        return x


class FlowCircuit(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, weight_dims, normalize_dim):
        super(WeightedMean, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(size=tuple(weight_dims)))
        self.normalize_dim = normalize_dim
        self.softmax = torch.nn.Softmax(dim=normalize_dim)
        
    def forward(self, x):
        x = (x*self.softmax(self.weights)).sum(self.normalize_dim )
        return x
