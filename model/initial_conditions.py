"""
    * module to generate loss based on the initial condition
"""
import math
from typing import Dict
from matplotlib import tri
from matplotlib import pyplot as plt
import torch
from .pinn import PINN

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
        self.input = torch.empty(1,device = self.device)
        self.output = torch.empty(1,device = self.device)

    def update_features(self,input):
        """
            upate input and output features using neural network forward pass
        """
        self.input = input
        self.output = self.model.net(input)

    def c_at_initial_time(self):
        """
            * get intial value of phase field
        """
        # check if t0 = 0, then use the bubble suspended in water condition
        if self.config['t_min'] == 0:
            r = self.config['r']
            x_coords = self.input[:,0:1]
            y_coords = self.input[:,1:2]
            xc = self.config['x_c']
            yc = self.config['y_c']
            z = (torch.sqrt((x_coords - xc)**2+(y_coords-yc)**2) - r)/ \
                (math.sqrt(2)*self.thermo['epsilon'])
            c_initial = torch.tanh(z)
        # else use previously converged neural net's parameters to get initial value
        else:
            # grad flag so neural net clone does not update when optimizer steps
            with torch.no_grad():
                c_initial = self.model_prev.net(self.input)[:,3:4]
        return c_initial

    def u_at_initial_time(self):
        """
            * get x velocity at initial time
        """
        if self.config['t_min'] == 0:
            u_initial = torch.zeros_like(self.output[:,0:1])
        else:
            with torch.no_grad():
                u_initial = self.model_prev.net(self.input)[:,0:1]
        return u_initial

    def v_at_initial_time(self):
        """
            * get y velocity at initial time
        """
        if self.config['t_min'] == 0:
            v_initial = torch.zeros_like(self.output[:,1:2])
        else:
            with torch.no_grad():
                v_initial = self.model_prev.net(self.input)[:,1:2]
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
        """
            * generate total loss based on initial condition
        """
        target_loss = torch.zeros_like(self.input[:,0:1])
        u = self.output[:,0:1]
        v = self.output[:,1:2]
        c = self.output[:,3:4]
        c_initial = self.c_at_initial_time()
        u_initial = self.u_at_initial_time()
        v_initial = self.v_at_initial_time()
        c_loss = torch.nn.functional.mse_loss(c - c_initial,target_loss)
        u_loss = torch.nn.functional.mse_loss(u - u_initial,target_loss)
        v_loss = torch.nn.functional.mse_loss(v - v_initial,target_loss)
        return u_loss + v_loss + c_loss

    def visualize_initial_condition(self,spatial_grid :  torch.Tensor,field_idx : int):
        """
            * function to plot contours of solution fields at initial condition
        """

        field_name_dict = {
            '0': 'x velocity',
            '1': 'y velocity',
            '2': 'phase field marker'
        }
        with torch.no_grad():

        # update just the input features from this function
            self.input = spatial_grid
            c = self.c_at_initial_time()[:,0]
            u = self.u_at_initial_time()[:,0]
            v = self.v_at_initial_time()[:,0]
            outputs = torch.stack([u,v,c],1)
            # Assuming the first 4 columns represent u, v, p, c
            fields = outputs[:, :3]
            # Selecting the specific field to visualize
            output_np = fields[:, field_idx].cpu().detach().numpy()
            input_np = spatial_grid.cpu().detach().numpy()
            x = input_np[:,0]
            y = input_np[:,1]
            triang = tri.Triangulation(x, y)
            plt.figure(figsize=(8, 6))
            contour = plt.tricontourf(triang, output_np, cmap='coolwarm',extend = 'both')
            plt.colorbar(contour)
            field_name = field_name_dict[f'{field_idx}']
            plt.title(f'Contour Plot of {field_name} on Spatial Grid')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.gca().set_aspect('equal')
            plt.show()
