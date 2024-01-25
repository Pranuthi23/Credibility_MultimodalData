import torch

def make_classifier(in_dim, out_dim, n_layers, n_hidden, activation=torch.nn.Tanh(), final_activation='torch.nn.Softmax(dim=-1)'):
    layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_dim, out_features=n_hidden),
            activation,
        ]
    for _ in range(n_layers):
        layers += [
        torch.nn.Linear(in_features=n_hidden, out_features=n_hidden),
        activation,
    ]
    layers += [torch.nn.Linear(in_features=n_hidden, out_features=out_dim)]
    
    if(final_activation is not None):
        layers += [eval(final_activation)]
    return torch.nn.Sequential(*layers)
    
class Classifier(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, n_hidden, activation=torch.nn.Tanh(), final_activation='torch.nn.Softmax(dim=-1)'):
        super(Classifier, self).__init__()
        self.head = make_classifier(in_dim, out_dim, n_layers, n_hidden, activation, final_activation)
        
    def forward(self, x, context=None):
        x = torch.cat(x, dim=-1) if type(x) == list else x
        return self.head(x)
        
class MultiHeadClassifier(torch.nn.Module):
    def __init__(self, n_heads, in_dim, out_dim, n_layers, n_hidden, activation=torch.nn.Tanh(), final_activation='torch.nn.Softmax(dim=-1)'):
        assert len(in_dim) == n_heads
        super(MultiHeadClassifier, self).__init__()
        
        self.heads = torch.nn.ModuleList([
            make_classifier(in_dim[i], out_dim[i], n_layers[i], n_hidden[i], activation, final_activation) for i in range(n_heads)
        ])
    
    def forward(self, x, context=None):
        out = []
        for x_i, head in zip(x, self.heads):
            out += [head(x_i).unsqueeze(1)]
        return torch.cat(out,dim=1)
        

