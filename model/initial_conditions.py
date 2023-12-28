from typing import Dict
import torch
from .pinn import PINN
import math

class IC():
    """
        * class to construct supervised loss based on the initial condition
    """
    def __init__(self,config : Dict,thermo : Dict, model : PINN,device):
        self.config = config
        self.thermo = thermo
        self.device = device
        self.model = model
        self.model_prev = model
        self.x = torch.empty(1,device = self.device)
        self.y = torch.empty(1,device = self.device)
    def update_features(self,x):
        """
            upate input and output features
        """
        self.x = x
        # forward feed to update output features from updated input features
        self.y = self.model.net(x)
    def c_at_initial_time(self):
        """
            * get phase field at intial time 
        """
        if self.config['t_min'] == 0:
            r = self.config['r']
            x_coords = self.x[:,0:1]
            y_coords = self.x[:,1:2]
            xc = self.config['x_c']
            yc = self.config['y_c']
            z = (torch.sqrt((x_coords - xc)**2+(y_coords-yc)**2) - r)/(math.sqrt(2)*self.thermo['epsilon'])
            c_initial = torch.tanh(z)
        else:
            with torch.no_grad():
                c_initial = self.model_prev.net(self.x)[:,3:4]
        return c_initial
            
    def u_at_initial_time(self):
        """
            * get x velocity at initial time
        """
        if self.config['t_min'] == 0:
            u_initial = torch.zeros_like(self.y[:,0:1])
        else:
            with torch.no_grad():
                u_initial = self.model_prev.net(self.x)[:,0:1]
        return u_initial
    def v_at_initial_time(self):
        """
            * get y velocity at initial time
        """
        if self.config['t_min'] == 0:
            v_initial = torch.zeros_like(self.y[:,1:2])
        else:
            with torch.no_grad():
                v_initial = self.model_prev.net(self.x)[:,1:2]
        return v_initial
    def update_initial_condition(self):
        """
            * update initial conditions
        """
        self.config['t_min'] = self.config['t_min'] + self.config['t_step']
    def update_model_prev(self):
        """
            * transfer learned parameters from current model to model_prev
        """
        self.model_prev.load_state_dict(self.model.state_dict())
    def loss(self):
        target_loss = torch.zeros_like(self.x[:,0:1])
        u = self.y[:,0:1]
        v = self.y[:,1:2]
        c = self.y[:,3:4]
        c_initial = self.c_at_initial_time()
        u_initial = self.u_at_initial_time()
        v_initial = self.v_at_initial_time()
        c_loss = torch.nn.functional.mse_loss(c - c_initial,target_loss)
        u_loss = torch.nn.functional.mse_loss(u-u_initial,target_loss)
        v_loss = torch.nn.functional.mse_loss(v-v_initial,target_loss)
        return u_loss + v_loss + c_loss

