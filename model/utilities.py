import torch
from matplotlib import pyplot as plt
import numpy as np
import json
from typing import Dict
from .pinn import PINN

def visualize_input(input_tensor : torch.Tensor,device) -> None:
    """
        * method to visualize input tensor to neural net, not to be called during
        training, more for visualization and sanity check purposes
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x_samples, y_samples, and t_samples from input_tensor
    if device == 'cuda':
        x_samples = input_tensor[:, 0].detach().cpu().numpy()
        y_samples = input_tensor[:, 1].detach().cpu().numpy()
        t_samples = input_tensor[:, 2].detach().cpu().numpy()
        # Scatter plot in 3D
        ax.scatter(x_samples, y_samples, t_samples,s=1)
    else:
        x_samples = input_tensor[:, 0].detach().numpy()
        y_samples = input_tensor[:, 1].detach().numpy()
        t_samples = input_tensor[:, 2].detach().numpy()
        # Scatter plot in 3D
        ax.scatter(x_samples, y_samples, t_samples,s=1)

    # Set labels for axes
    ax.set_xlabel('X Samples')
    ax.set_ylabel('Y Samples')
    ax.set_zlabel('T Samples')
    ax.set_box_aspect([np.ptp(x_samples), np.ptp(y_samples), np.ptp(t_samples)])
    plt.title('3D Scatter Plot of Input')
    plt.show()

def gradients(outputs : torch.Tensor, inputs : torch.Tensor):
    """
        returns gradients of a single output wrt to a input tensor
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )

def read_configs(file_path) -> Dict:
    """
        function to read in .json file storing config data for all classes
    """
    with open(file_path,'r',encoding='utf-8') as file:
        config_dict = json.load(file)
    # returns list of dictionaries
    return config_dict


