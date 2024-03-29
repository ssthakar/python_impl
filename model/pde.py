"""
    * module to train a pinn to simulate the Hyesing benchmark
"""
import math
from typing import Dict
import torch
from .pinn import PINN
from .utilities import gradients
class PDE():
    """
        * Class to construct the unsupervised loss from the governing Equations:
        the Cahn-Hilliard Equations in this case.
        * Notation:
            * _g represents tensor storing gradients of a single variable wrt to all 
            inputs x,y and t
            * grad_ represents the spatial gradient
            * _var represents the partial derivative of any field wrt to that particu-
            lar input variable
            * thermo physical parameters are stored in thermo config dict
            passed to the init method
    """
    def __init__(self,config : Dict, thermo : Dict,model : PINN,device):
        self.config = config
        # dict that stores thermophysicla parameters
        self.thermo = thermo
        self.model = model
        self.device = device
        self.input = torch.empty(1,device = self.device)
        self.output = torch.empty(1,device = self.device)

    def update_features(self,input):
        """
            * update input and output features
            * x (x,y,t) X nsamples is the randomly shuffled input from the  
            internal grid
            * y is the output from the forward pass of neural net
            * c - phasefield marker
            * u - x velocity
            * v - y velocity 
            * p - pressure
        
        """
        # inputs to the neural network N_samples X 3
        self.input = input
        # outputs of the neural network N_samples X 4
        self.output = self.model.net(self.input)

    def phi(self):
        """
            returns the double well potential term seen in the Cahn-Hilliard equations
        """
        # ensure indexing maintains 2D shape of all tensors
        c = self.output[:,3:4]
        c_g = gradients(c,self.input)[0]
        c_x = c_g[:,0:1]
        c_y = c_g[:,1:2]
        c_x_g = gradients(c_x,self.input)[0]
        c_y_g = gradients(c_y,self.input)[0]
        c_xx = c_x_g[:,0:1]
        c_yy = c_y_g[:,1:2]
        """
            * c(c^2-1) - e^2(nabla^2(c))
        """
        return c**3 - c - (c_xx + c_yy)*self.thermo['epsilon']*self.thermo['epsilon']

    def surface_tension(self):
        """
            returns the surface tension term needed in the momemtum equations
        """
        sigma = self.thermo['sigma']
        epsilon = self.thermo['epsilon']
        # this the constant value of 3*sqrt(2)/4, I have stored in the dict
        c  = self.output[:,3:4]
        c_g = gradients(c,self.input)[0]
        # return the spatial gradient of phasefield marker
        grad_c = c_g[:,0:2]
        # return the two dimensional surface tension force vector
        return  (self.phi()*grad_c*sigma*3*math.sqrt(2))/(epsilon*4)

    def c_bar(self):
        """
            * computes the auxiliary variable c_bar to constrain thermophysical 
            properties for bubble rise
        """
        # actual phasefield variable
        c = self.output[:,3:4]
        abs_c = torch.abs(c)
        """
            * torch.where is element wise operation:

                out_i = { input_i       if condition
                        { other_i       otherwise
        """
        return torch.where(abs_c <= 1,c,torch.sign(c))

    def thermo_prop(self,prop_l,prop_g):
        """
            * computes the thermophysical properties of the mixture
        """
        c_bar = self.c_bar()
        return 0.5*(1+c_bar)*prop_l + 0.5*(1-c_bar)*prop_g

    def cahn_hilliard_loss(self):
        """
            * loss residual derived from the phase field transport equation
        """
        c = self.output[:,3:4]
        u = self.output[:,0:1]
        v = self.output[:,1:2]
        phi = self.phi()
        c_g = gradients(c,self.input)[0]
        phi_g = gradients(phi,self.input)[0]
        c_x = c_g[:,0:1]
        c_y = c_g[:,1:2]
        c_t = c_g[:,2:3]
        phi_x = phi_g[:,0:1]
        phi_y = phi_g[:,1:2]
        phi_x_g = gradients(phi_x,self.input)[0]
        phi_y_g = gradients(phi_y,self.input)[0]
        phi_xx = phi_x_g[:,0:1]
        phi_yy = phi_y_g[:,1:2]
        # construct residual
        loss = c_t + u*c_x + v*c_y - \
            (self.thermo['Mo'])*(phi_xx + phi_yy)
        # residual is loss
        target_loss = torch.zeros_like(loss)
        return torch.nn.functional.mse_loss(loss,target_loss)

    def continuity_loss(self):
        """
            * loss residual derived from continuity equation
        """
        u = self.output[:,0:1]
        v = self.output[:,1:2]
        u_g = gradients(u,self.input)[0]
        v_g = gradients(v,self.input)[0]
        u_x = u_g[:,0:1]
        v_y = v_g[:,1:2]
        loss = u_x + v_y
        target_loss = torch.zeros_like(loss)
        return torch.nn.functional.mse_loss(loss,target_loss)

    def x_momentum_loss(self):
        """
            * loss residual derived from x momentum equation
        """
        # get solution fields from update ouput
        u = self.output[:,0:1]
        v = self.output[:,1:2]
        p = self.output[:,2:3]
        c = self.output[:,3:4]
        # surface tension force component
        fx = self.surface_tension()[:,0:1]
        # compute gradients of fields wrt updated input
        u_g = gradients(u,self.input)[0]
        v_g = gradients(v,self.input)[0]
        c_g = gradients(c,self.input)[0]
        p_g = gradients(p,self.input)[0]
        # get partial derivatives needed in the residual
        u_t = u_g[:,2:3]
        u_x = u_g[:,0:1]
        u_y = u_g[:,1:2]
        v_x = v_g[:,0:1]
        u_x_g = gradients(u_x,self.input)[0]
        u_xx = u_x_g[:,0:1]
        u_y_g = gradients(u_y,self.input)[0]
        u_yy = u_y_g[:,1:2]
        c_x = c_g[:,0:1]
        c_y = c_g[:,1:2]
        p_x = p_g[:,0:1]
        # get density and viscosity of mixture
        rho_mix = self.thermo_prop(self.thermo['rhoL'],self.thermo['rhoG'])
        mu_mix = self.thermo_prop(self.thermo['muL'],self.thermo['muG'])
        # thermophysical properties of liquid and gas phases
        mu_l = self.thermo['muL']
        mu_g = self.thermo['muG']
        rho_l = self.thermo['rhoG']
        # construct the loss
        loss = rho_mix*(u_t + u*u_x + v*u_y) + p_x \
            -0.5*(mu_l -mu_g)*c_y*(u_y + v_x) -\
            (mu_l - mu_g)*c_x*u_x -fx - mu_mix*(u_xx + u_yy)
        target_loss = torch.zeros_like(loss)
        # divide by rho_l to normalize the loss
        return torch.nn.functional.mse_loss(loss/rho_l,target_loss)

    def y_momentum_loss(self):
        """
            * loss residual derived from the y momentum equation
        """
        u = self.output[:,0:1]
        v = self.output[:,1:2]
        p = self.output[:,2:3]
        c = self.output[:,3:4]
        u_g = gradients(u,self.input)[0]
        v_g = gradients(v,self.input)[0]
        c_g = gradients(c,self.input)[0]
        p_g = gradients(p,self.input)[0]
        fy = self.surface_tension()[:,1:2]
        v_t = v_g[:,2:3]
        v_x = v_g[:,0:1]
        v_y = v_g[:,1:2]
        p_y = p_g[:,1:2]
        c_x = c_g[:,0:1]
        c_y = c_g[:,1:2]
        u_y = u_g[:,1:2]
        v_x_g = gradients(v_x,self.input)[0]
        v_xx = v_x_g[:,0:1]
        v_y_g = gradients(v_y,self.input)[0]
        v_yy = v_y_g[:,1:2]
        rho_mix = self.thermo_prop(self.thermo['rhoL'],self.thermo['rhoG'])
        mu_mix = self.thermo_prop(self.thermo['muL'],self.thermo['muG'])
        # thermophysical properties of liquid and gas phases
        mu_l = self.thermo['muL']
        mu_g = self.thermo['muG']
        rho_l = self.thermo['rhoG']
        # gravity term
        gy = torch.full_like(u,self.thermo['g'])
        loss = rho_mix*(v_t + u*v_x + v*v_y) + p_y \
            -0.5*(mu_l -mu_g)*c_x*(v_x + u_y) -\
            (mu_l - mu_g)*c_y*v_y -fy -rho_mix*gy - mu_mix*(v_xx + v_yy)
        target_loss = torch.zeros_like(loss)
        # divide by rho_l to normalize the loss
        return torch.nn.functional.mse_loss(loss/rho_l,target_loss)


    def loss(self):
        """
            * constructs the total PDE loss residual

            * called every batch iteration
        """
        # print(self.x_momentum_loss())
        return self.cahn_hilliard_loss() + \
            self.continuity_loss() + self.x_momentum_loss() + \
            self.y_momentum_loss()
