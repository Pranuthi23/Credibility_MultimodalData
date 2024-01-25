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
        y = y if self.multilabel else y/y.sum(dim=-1,keepdim=True)
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
            
    def forward(self, x, context=None, marginalized_scopes=None):
        if self.multilabel:
            x = x.view(x.shape[0],-1,x.shape[3]) if len(x.shape)==4 else x.view(x.shape[0],-1) 
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=marginalized_scopes).exp()
            probs = probs.view(probs.shape[0],-1,2)
            probs = probs/probs.sum(dim=-1, keepdim=True)
            return probs[:,:,1]
            # return probs
        else:    
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=marginalized_scopes).exp()
            return probs/probs.sum(dim=-1,keepdim=True)
       
class FlowCircuit(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0):
        super(FlowCircuit, self).__init__()
        raise NotImplementedError
    


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = torch.nn.functional.one_hot(p, num_classes=c)
    
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))




class TMC(torch.nn.Module):
    def __init__(self, classes):
        super(TMC, self).__init__()
        # self.args = args
        # self.rgbenc = ImageEncoder(args)
        # self.depthenc = ImageEncoder(args)
        # depth_last_size = args.img_hidden_sz * args.num_image_embeds
        # rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        # self.clf_depth = nn.ModuleList()
        # self.clf_rgb = nn.ModuleList()
        # for hidden in args.hidden:
        #     self.clf_depth.append(nn.Linear(depth_last_size, hidden))
        #     self.clf_depth.append(nn.ReLU())
        #     self.clf_depth.append(nn.Dropout(args.dropout))
        #     depth_last_size = hidden
        # self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))

        # for hidden in args.hidden:
        #     self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
        #     self.clf_rgb.append(nn.ReLU())
        #     self.clf_rgb.append(nn.Dropout(args.dropout))
        #     rgb_last_size = hidden
        # self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))
        self.num_classes = classes

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.num_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.num_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return torch.nn.functional.softmax(alpha_a, dim = 1)

    def forward(self, x, context=None):
        # depth = self.depthenc(depth)
        # depth = torch.flatten(depth, start_dim=1)
        # rgb = self.rgbenc(rgb)
        # rgb = torch.flatten(rgb, start_dim=1)
        # depth_out = depth
        # for layer in self.clf_depth:
        #     depth_out = layer(depth_out)
        # rgb_out = rgb
        # for layer in self.clf_rgb:
        #     rgb_out = layer(rgb_out)

        
        rgb_out = x[:, :1, :]
        depth_out = x[:, 1:, :]
        rgb_out = rgb_out.squeeze(dim = 1)
        depth_out = depth_out.squeeze(dim = 1)

        # depth_evidence, rgb_evidence = torch.nn.functional.softplus(depth_out), torch.nn.functional.softplus(rgb_out)
        depth_alpha, rgb_alpha = depth_out+1, rgb_out+1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        # return depth_alpha, rgb_alpha, depth_rgb_alpha
        return depth_rgb_alpha