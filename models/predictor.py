import torch

class Classifier(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, n_hidden, activation=torch.nn.Tanh()):
        super(Classifier, self).__init__()
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
        layers += [
            torch.nn.Linear(in_features=n_hidden, out_features=out_dim),
            torch.nn.Softmax(dim=-1),
        ]
        self.head = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)
        