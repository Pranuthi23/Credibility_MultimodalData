import torch 
from packages.RatSPN.spn.rat_spn import make_rat
from packages.EinsumNet.simple_einet.einet import EinetConfig, Einet
from packages.EinsumNet.simple_einet.einet_mixture import EinetMixture
from packages.EinsumNet.simple_einet.layers.distributions.binomial import Binomial
from packages.EinsumNet.simple_einet.layers.distributions.categorical import Categorical, ConditionalCategorical
from packages.EinsumNet.simple_einet.layers.distributions.normal import RatNormal, Normal
from packages.EinsumNet.simple_einet.layers.distributions.dirichlet import Dirichlet
from packages.EinsumNet.simple_einet.layers.distributions.bernoulli import Bernoulli


class WeightedMean(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, weight_dims, normalize_dim, multilabel=False):
        super(WeightedMean, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(size=tuple(weight_dims)))
        self.normalize_dim = normalize_dim
        self.softmax = torch.nn.Softmax(dim=normalize_dim)
        self.multilabel = multilabel 
        
    def forward(self, x, context=None):
        x = x[:,:,:,0] if self.multilabel else x
        x = (x*self.softmax(self.weights)).sum(self.normalize_dim)
        return x

class NoisyOR(torch.nn.Module):
    """Implements a module that returns performs noisy or"""
    
    def __init__(self, normalize_dim, multilabel=False):
        super(NoisyOR, self).__init__()
        self.normalize_dim = normalize_dim
        self.multilabel = multilabel
        
    def forward(self, x, context=None):
        x = x[:,:,:,0] if self.multilabel else x
        y = 1 - torch.prod(1-x,dim=self.normalize_dim)
        y = y/y.sum(dim=-1,keepdim=True)
        return y


class RatSPN(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0, multilabel=False):
        super(RatSPN, self).__init__()
        self.model = make_rat(num_features=num_features, classes=classes, leaves=leaves, sums=sums, num_splits=num_splits, dropout=dropout)
        
    def forward(self, x, context=None):
        return self.model(x).exp()
    
class EinsumNet(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, num_channels, depth, num_sums, num_leaves,
                 num_repetitions, num_classes, leaf_type, leaf_kwargs, einet_mixture=False, layer_type='einsum', conditional_leaf=False, conditional_sum=False, dropout=0.0, multilabel=False):
        super(EinsumNet, self).__init__()
        leaf_type, leaf_kwargs = eval(leaf_type), leaf_kwargs
        self.config = EinetConfig(
            num_features=num_features,
            num_channels=num_channels,
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            num_classes=num_classes,
            leaf_kwargs=leaf_kwargs,
            leaf_type=leaf_type,
            dropout=dropout,
            layer_type=layer_type,
            conditional_leaf=conditional_leaf,
            conditional_sum=conditional_sum
        )
        self.multilabel = multilabel
        if einet_mixture:
            self.model = EinetMixture(n_components=num_classes, einet_config=self.config)
        else:
            self.model = Einet(self.config)
            
    def forward(self, x, context=None):
        if self.multilabel:
            x = x.view(x.shape[0],-1,x.shape[3]) if len(x.shape)==4 else x.view(x.shape[0],-1) 
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context).exp()
            probs = probs.view(probs.shape[0],-1,2)
            probs = probs/probs.sum(dim=-1, keepdim=True)
            return probs[:,:,0]
        else:    
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context).exp()
            return probs/probs.sum(dim=-1,keepdim=True)
       
class FlowCircuit(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0):
        super(FlowCircuit, self).__init__()
        raise NotImplementedError