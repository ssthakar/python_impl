"""
    * training sub-routine for Bubble rise pinns implementation
"""
from model.pinn import PINN
from model.boundary_conditions import BC
from model.initial_conditions import IC
from model.pde import PDE
from data.datasets import Grid2D
from torch.utils.data import DataLoader
import torch
import model.utilities as util
import logging as log

def train(config_file, pde_batch_size: int, n_epoch : int,debug_mode  :bool):
    """
        sub-routine to train the neural network
        
        * batch size for pde input pde_batch_size

        * n_epochs is the total number of epochs the net is trained for
    """
    # set device for the training process
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Training on: {device}")

    # main config, holds all other configs, json load
    config = util.read_configs(config_file)

    # construct neural network 
    model = PINN(config['pinn'])

    # transfer model to device
    model.to(device)
    
    # start data construction

    """
        * construct grids, boundary, initial and pde loss objects
    """

    # construct internal grid, requires batching
    internal_grid = Grid2D(config['internal_grid'],device)
    
    # get number of iterations needed in the batch feed loop
    n_iterations = internal_grid.config['n_samples']/pde_batch_size
    
    # make data loader for internal grid inputs
    train_loader = DataLoader(dataset = internal_grid,batch_size=pde_batch_size,shuffle=False,num_workers=0) 
    
    # construct pde loss object
    pde = PDE(config['pde'],config['thermo'],model,device)
    
    # construct initial grid, does not require batching
    initial_grid = Grid2D(config['initial_grid'],device)
    
    # construct intial condition loss object
    ic = IC(config['ic'],config['thermo'],model,device)
    
    # construct optimizer class instance, pass model paramters to optimizer, set learning rate
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)  # Initialize Adam optimizer0I

    # list of grids that don't require batching, use this to populate all dicts
    boundary_grid_names = ['left_grid','right_grid','top_grid','bottom_grid']
    
    # init an empty dict to store actual boundary grid objects
    boundary_grids = {}
    
    # populte the dict by constructing grid objects
    for name in boundary_grid_names:
        boundary_grids[name] = Grid2D(config[name],device)
    
    # init empty dict to store boundary loss objects
    bc = {}
    
    # populate the dict with boundary loss objects associated with each grid
    for name in boundary_grid_names:
        bc[name] = BC(config['bc'],config['thermo'],model,device)
    
    # init empty dict to store input values for boundary loss objects
    bc_inputs = {}
    
    

    """
        * sanity check for seeing in model inputs are within bounds and correct
    """
    if debug_mode:
        for name  in boundary_grid_names:
            bc_inputs[name] = torch.empty(1,device=device)
        
        # generate random samples from total samples
        for grid in boundary_grids.values():
                grid.update_input()

        # loop through all boundary_grids and get inputs as tensors
        for name in boundary_grid_names:
            bc_inputs[name] = boundary_grids[name].get_inputs()

        # visualize boundary inputs 
        temp_inputs = list(bc_inputs.values())
        # util.visualize_model_input(bc_inputs['right_grid'],device)
        
        util.visualize_model_inputs(temp_inputs,device)
        
        util.tensor_to_file(bc_inputs['top_grid'],'top_grid.txt')
        
    





    """
        * begin training
    """
    
    # set model mode to training
    model.train()
    
    

    # begin epoch loop
    for epoch in range(n_epoch):
        """
            * update inputs for internal, initial, and boundary grids

            * generates n(user_ defined) random samples
        """

        # update ineternal grid and intial grid inputs
        # generate N_PDE, N_BC,N_IC samples
        internal_grid.update_input()
        initial_grid.update_input()
        
        # loop through all boundary grids and update input
        for grid in boundary_grids.values():
            grid.update_input()
        
        # get inputs for loss objects that don't need batching 
        
        # initial inputs
        initial_inputs = initial_grid.get_inputs()
        
        # loop through all boundary inputs and get them
        for name in boundary_grid_names:
            bc_inputs[name] = boundary_grids[name].get_inputs()
        
        # init place holder float value for loss
        total_loss = 0.0

        # batching loop
        optim.zero_grad()

        # batch loop for pde inputs which are very large in number
        for _,(x_inputs,y_inputs,t_inputs) in enumerate(train_loader):
            # stack inputs obtained from data-loader to generate pde_input 
            pde_inputs = torch.stack([x_inputs,y_inputs,t_inputs],1).requires_grad_(True)
            
            # forward pass for pde inputs
            pde.update_features(pde_inputs)
            
            # forward pass for intial inputs
            ic.update_features(initial_inputs)
            
            # forward pass for boundary inputs, loops through all boundaries using named list
            for name in boundary_grid_names:
                bc[name].update_features(bc_inputs[name])
            
            # construct the loss
            batch_loss = pde.loss() + ic.loss() + bc['left_grid'].slip_wall(0,1) + \
                bc['right_grid'].slip_wall(0,1) + bc['top_grid'].no_slip_wall() +\
                bc['bottom_grid'].no_slip_wall()
            
            # backward pass
            batch_loss.backward()

            # get loss value
            total_loss += batch_loss.item() 
            # print(total_loss)
        
        # update network parameters based on the accumulated gradients
        optim.step()
        
        # print out loss info for monitoring purposes
        if epoch%10 == 0 and epoch != 0:
            log.info(total_loss/n_iterations)
        
        # model check point 
        if epoch%5000 == 0 and epoch!=0:
            torch.save(model,'model_checkpoint.pth')
        
        # break loop if convergence is achieved
        if total_loss/n_iterations < config['pinn']['abs_tol']:
            # save converged model for later use
            torch.save(model,'model.pth')
            # break out of epoch loop 
            break


