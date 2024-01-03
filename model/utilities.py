import torch
from matplotlib import pyplot as plt
import numpy as np
import json
from typing import Dict
from .pinn import PINN
import matplotlib.tri as tri

def visualize_model_input(input_tensor : torch.Tensor,device) -> None:
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
    ax.set_aspect('auto')
    # ax.set_box_aspect([np.ptp(x_samples), np.ptp(y_samples), np.ptp(t_samples)])
    plt.title('3D Scatter Plot of Input')
    plt.show()

def visualize_model_inputs(input_tensors, device):
    """
    Visualizes multiple input tensors in a 3D scatter plot, differentiating them by color.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm']  # List of colors for differentiating tensors

    for i, input_tensor in enumerate(input_tensors):
        if device == 'cuda':
            data = input_tensor.cpu().detach().numpy()
        else:
            data = input_tensor.detach().numpy()

        # Extract x, y, z samples from the tensor data
        x_samples = data[:, 0]
        y_samples = data[:, 1]
        z_samples = data[:, 2]

        # Scatter plot in 3D with color differentiation
        ax.scatter(x_samples, y_samples, z_samples, s=10, c=colors[i], label=f'Tensor {i+1}')

    # Set labels for axes and legend
    ax.set_xlabel('X Samples')
    ax.set_ylabel('Y Samples')
    ax.set_zlabel('Z Samples')
    ax.legend()
    plt.title('3D Scatter Plot of Multiple Input Tensors')
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
        function to read in .json file, returns dict
    """
    with open(file_path,'r',encoding='utf-8') as file:
        config_dict = json.load(file)
    # returns list of dictionaries
    return config_dict

def model_inference(config :Dict, t_final : float, model : PINN):
    """
        * function to load saved model and predict output based on input
    """
    device = model.net.input_layer.bias.device
    inputs = get_spatial_grid(config,t_final,device)
    # inference
    outputs = model.net(inputs)
    # print(outputs.size())
    return inputs,outputs

def visualize_model_output(input_tensor, output_tensor, field_idx: int):
    """
    * Post-process function to visualize the various solution fields
    * Used for model inference
    * field_idx tells us what solution field is to be plotted
    """
    field_name_dict = {
        '0': 'x velocity',
        '1': 'y velocity',
        '2': 'pressure',
        '3':'phase field marker'
    }
    # Get solution fields from output tensor
    fields = output_tensor[:, :4]  # Assuming the first 4 columns represent u, v, p, c
    output_np = fields[:, field_idx].cpu().detach().numpy()  # Selecting the specific field to visualize
    input_np = input_tensor.cpu().detach().numpy()
    x = input_np[:,0]
    y = input_np[:,1]
    triang = tri.Triangulation(x, y)
    #
    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(triang, output_np, cmap='coolwarm')  # Adjust colormap as needed
    plt.colorbar(contour)  # Add color bar indicating temperature scale
    field_name = field_name_dict[f'{field_idx}']
    plt.title(f'Contour Plot of {field_name} on Spatial Grid')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal')
    plt.show()

def get_spatial_grid(config : Dict,t_final : float,device):
    x_bounds = np.linspace(config['x_min'],config['x_max'],config['nx'])
    y_bounds = np.linspace(config['y_min'],config['y_max'],config['ny'])
    x_grid,y_grid = np.meshgrid(x_bounds,y_bounds)
    t_grid = np.full_like(y_grid,t_final)


    # generate the input features for model inference
    x_inputs = torch.tensor(
        x_grid.flatten(),
        dtype=torch.float32,
        device=device
    )
    y_inputs = torch.tensor(
        y_grid.flatten(),
        dtype=torch.float32,
        device=device
    )
    t_inputs = torch.tensor(
        t_grid.flatten(),
        dtype=torch.float32,
        device=device
    )
    inputs = torch.stack([x_inputs,y_inputs,t_inputs],1)
    return inputs

def tensor_to_file(input_tensor : torch.Tensor,file_name :str):
    tensor_np = input_tensor.detach().cpu().numpy()
    np.savetxt(file_name,tensor_np)
    


