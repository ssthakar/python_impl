from torch.utils.data import Dataset
import torch
from typing import Dict
import numpy as np

class Grid2D(Dataset):
    """
        * Input features class, uses a dictionary as input for init method.
        
        * inherits from Dataset class 

        * passed to Dataloader class when using batches during training
            (PDE input size and neural net depth is very large and therefore will require 
            batching to fit on the GPU)
        
        * batch size and shuffle boolean are input to the data loader

        * config dictionary example:
            'x_min': lower bounds for x
            'x_max': upper bounds for x
            'nx': number of points in the x direction
            'y_min': lower bounds for y
            'y_max': upper bounds for y
            'ny': number of bounds in y direction
            't_min': lower bound for time
            't_max': upper bound for time
            'nt' : number of points along the time axis
            'n_samples' : number of random points to sample every epoch
        
        * device agnostic

    """
    def __init__(self,config : Dict,device) -> None:
        self.config = config
        self.device = device
        # samples won't change throughout the training process
        self.samples = [
            torch.empty(0,device=self.device),
            torch.empty(0,device=self.device),
            torch.empty(0,device=self.device)
        ]
        # inputs will be randomly sampled from samples every epoch
        self.inputs = [
            torch.empty(0,device=self.device),
            torch.empty(0,device=self.device),
            torch.empty(0,device=self.device)
        ]
        # construct samples
        self.create_grid()
    
    def create_grid(self) -> None:
        """
           * creates grid for the input feature

           * called by __init__ method
        """
        x_bounds = np.linspace(
            self.config['x_min'],
            self.config['x_max'],
            self.config['nx'])
        y_bounds = np.linspace(
            self.config['y_min'],
            self.config['y_max'],
            self.config['ny'])
        t_bounds = np.linspace(
            self.config['t_min'],
            self.config['t_max'],
            self.config['nt'])
        x_grid,y_grid,t_grid = np.meshgrid(x_bounds,y_bounds,t_bounds)
        # vectorize the meshgrid for random sampling using indices
        self.samples[0] = torch.tensor(
            x_grid.flatten(),
            dtype = torch.float32,
            device=self.device
        ) # x samples  
        self.samples[1] = torch.tensor(
            y_grid.flatten(),
            dtype = torch.float32,
            device=self.device
        ) # y samples
        self.samples[2] = torch.tensor(
            t_grid.flatten(),
            dtype = torch.float32,
            device=self.device
        ) # t samples

    def __len__(self):
        """
            * over-ride of parent method, returns total input sample size
        """
        return self.config['n_samples']

    def update_input(self):
        """
            * creates randomly sampled input for the neural net's forward pass

            * called every epoch
        """
        n_total = self.samples[0].numel()
        n_samples = self.config['n_samples']
        # generate random indices within bounds of vectorized samples,extract n_samples 
        indices = torch.randperm(n_total,device = self.device,dtype=torch.int32)[:n_samples]
        self.inputs[0] = self.samples[0].index_select(0,indices)        # x inputs 
        self.inputs[1] = self.samples[1].index_select(0,indices)        # y inputs
        self.inputs[2] = self.samples[2].index_select(0,indices)        # t inputs
    
    def __getitem__(self, index):
        """
            * returns the batch from input samples, over-ride of the parent class method

            * used with the data loader class
        """
        return self.inputs[0][index],self.inputs[1][index],self.inputs[2][index]
    def get_inputs(self):
        """
            * returns stacked inputs when batching is not needed

            * generally used for grids with smaller number of n_samples
        """
        return torch.stack(self.inputs,1).requires_grad_(True)


