"""
    * module dealing with boundary losses for the physics informed neural net
"""
from typing import Dict
import torch
from .utilities import gradients
from .pinn import PINN

class BC():
    """
        * class to construct supervised loss based on boundary conditions
    """
    def __init__(self, config : Dict,thermo : Dict, model : PINN, device):
        self.config = config        # class constructor config
        self.thermo = thermo        # thermo properties config
        self.device = device
        self.model  = model
        self.input = torch.empty(1,device = self.device)
        self.output = torch.empty(1,device = self.device)

    def update_features(self,input):
        """
            * updates input and output features for boundary loss object, using
            neural net forward pass
        """
        self.input = input                  #input features
        self.output = self.model.net(input)  #output features

    def slip_wall(self,normal_dir : int,tangential_dir : int):
        """
            * computes the slip velocity loss, tangential dir and normal dir are wrt
            the wall.

            * for example: tangential dir = 0 normal dir = 1 would mean:

                du/dy = 0
        """
        normal_vel = self.output[:,normal_dir:normal_dir+1]
        tangential_vel = self.output[:,tangential_dir:tangential_dir+1]
        grad = gradients(tangential_vel,self.input)[0]
        vel_grad = grad[:,normal_dir:normal_dir+1]
        target_loss = torch.zeros_like(vel_grad)
        return torch.nn.functional.mse_loss(vel_grad,target_loss)+ \
            torch.nn.functional.mse_loss(normal_vel,target_loss)
    
    def no_slip_wall(self):
        """
            * computes no slip wall loss for given boundary
        """
        u = self.output[:,0:1]
        v = self.output[:,1:2]
        target_loss = torch.zeros_like(u)
        return torch.nn.functional.mse_loss(u,target_loss) + \
            torch.nn.functional.mse_loss(v,target_loss)


