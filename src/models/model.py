import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    """Neural Network Generator."""
    def __init__(
        self, input_size=24, hidden_sizes=[20,5], output_size=1, drop_p=0.4
    ):
        """Generate fully-connected neural network.

        parameters
        ----------
        input_size (int): size of the input
        hidden_sizes (list of int): size of the hidden layers
        output_layer (int): size of the output layer
        drop_p (float): dropout probability
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_size, hidden_sizes[0])
        ])
        layers = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layers])
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, X):
        for linear in self.hidden_layers:
            X = F.relu(linear(X))
            X = self.dropout(X)
        
        X = self.output(X)

        return F.relu(X)