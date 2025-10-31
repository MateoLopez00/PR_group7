import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
MLP with a single hidden layer.

Args:
    input_size (int): Number of input features
    hidden_size (int): Number of neurons in the hidden layer (10-256)
    output_size (int): Number of output neurons
"""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        ## assert one hidden layer between 10 and 256 neurons
        assert 10 <= hidden_size <= 256, "Hidden layer size must be between 10 and 256"
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):  
        return self.network(x)
    
    


