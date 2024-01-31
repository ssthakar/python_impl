"""
    * module responsible for the neural network
"""
from typing import Dict
import torch
from torch import nn

# define PINN class implementatio
class PINN(nn.Module):
    """
    * Generic Physics informed neural net class, models a feed forward neural net
    
    * Uses a dictionary as input for the init method. 
    
    * config dict example:
       'input_dim' : number of input features
       'output_dim' : number of output features
       'n_hidden_layers': number of hidden layers
       'hidden_layer_dim' : number of neurons in each hidden layer
       'abs_tol': target loss for the optimizer
    
    * device agnostic

    """
    def __init__(self,config : Dict,act_func) -> None:
        super().__init__()
        # neural net attributes
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.n_hidden_layers = config['n_hidden_layers']
        self.hidden_layer_dim = config['hidden_layer_dim']
        # define neural net
        self.net = nn.Sequential()
        self.net.add_module(
            'input_layer',
            nn.Linear(self.input_dim,self.hidden_layer_dim)
        )
        self.net.add_module(
            'input_layer_activation',
            act_func()
        )
        # add the hidden layers
        for n in range(self.n_hidden_layers):
            self.net.add_module(
                f'hidden_layer_{n}',
                nn.Linear(self.hidden_layer_dim,self.hidden_layer_dim)
            )
            # initialize the weights of the linear layer
            nn.init.xavier_normal_(self.net[-1].weight)
            self.net.add_module(
                f'hidden_layer_activation_{n}',
                act_func()
            )
        self.net.add_module(
            'output_layer',
            nn.Linear(self.hidden_layer_dim,self.output_dim)
        )
    # forward pass
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        """
            forward feed
        """
        return self.net(x)
