from time import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


# Based on "Analyzing Inverse Problems with Invertible Neural Networks" by L. Ardizzone et al.
# This class initializes an INN and contains functions for training it
class INN():

    def __init__(self, ndim_tot, ndim_y, ndim_x,  ndim_z, feature, num_blocks, batch_size, lr, lambd_predict = 1.,lambd_predict_back=1., lambd_latent = 300., lambd_rev = 450., loss_back='mse', debug_grad=False, scale=1, sub_len=1, device = "cpu"):
        self.device = device
        # Depth of subnetworks
        self.sub_len = sub_len
        # Scale for MMD loss (h)
        self.scale = scale
        # Toggle printing of gradients of the losses, used in manual weighting
        self.debug_grad = debug_grad
        # Set the loss for the energy distribution (y) loss
        self.loss_back=loss_back
        # Set the total dimension of both inputs
        self.ndim_tot = ndim_tot
        # Set the dimension of y (energy distribuiton)
        self.ndim_y =   ndim_y 
        # Set the dimension of x (parameter)
        self.ndim_x =   ndim_x 
        # Set the dimension of z (latent)
        self.ndim_z =   ndim_z 
        # Set the width of the subnetworks
        self.feature =  feature
        # Set he number of blocks
        self.num_blocks = num_blocks
    
        # Direction depending mmd kernels, for testing
        #self.mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
        #self.mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
        torch.manual_seed(1234)
        
        # Set up subnetwork
        def subnet_fc(c_in, c_out):
            layers = []
            layers.append(nn.Linear(c_in, self.feature))
            layers.append(nn.LeakyReLU())
            for i in range(self.sub_len-1):
                layers.append(nn.Linear(self.feature, self.feature))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(self.feature,  c_out))
            return nn.Sequential(*layers)
        
        # Set up nodes/coupling blocks
        nodes = [InputNode(self.ndim_tot, name='input')]

        for k in range(self.num_blocks):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':subnet_fc, 'clamp':2.0}, #2.0
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))
        
        # Set model (from FrEIA)
        self.model = ReversibleGraphNet(nodes, verbose=False)
        
        # Training parameters
        self.batch_size = batch_size
        # Set learning rate
        self.lr = lr
        # Set l2 regression parameter
        l2_reg = 2e-5
        
        # Set noise scale for padding
        self.y_noise_scale = 1e-1
        self.zeros_noise_scale = 5e-2

        # relative weighting of losses:
        self.lambd_predict = lambd_predict
        self.lambd_predict_back=lambd_predict_back
        self.lambd_latent = lambd_latent
        self.lambd_rev = lambd_rev

        # Zero padding dimensions if both sides don't match [x,0] <-> [z,y,0]
        self.pad_x = torch.zeros(self.batch_size, ndim_tot - ndim_x)
        self.pad_yz = torch.zeros(self.batch_size, ndim_tot - ndim_y - ndim_z)

        # Set up the optimizer
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr, betas=(0.8, 0.9),
                                     eps=1e-6, weight_decay=l2_reg)
        # MMD losses
        self.loss_backward = self.MMD_multiscale
        self.loss_latent = self.MMD_multiscale
        
        # Supervised loss for the y,z -> x direction
        self.loss_fit_back = torch.nn.MSELoss()
        
        # Initialize weights
        for param in trainable_parameters:
            # To avoid NaNs if the number of blocks is high, a smaller initalization for the weights is recommended
            if num_blocks < 20:
                param.data = 0.05*torch.randn_like(param)
            else:
                param.data = 0.0005*torch.randn_like(param)
        self.model.to(self.device);
        return

    
     # A function which to make the supervised loss x <- y,z choosable
    def loss_fit(self, output, gt):        
        if self.loss_back == "l1":
            return torch.nn.L1Loss()(output, gt)
        if self.loss_back == "mape":
            # MAPE with the addition of a small constant, due to the presence of zeros in the data
            return torch.mean(torch.abs(output-gt)/torch.abs(gt+1e-8))
        else:
            return torch.nn.MSELoss()(output, gt)


    # A function to get the gradients for a given loss term
    def loss_grad(self,loss):
        grads = []
        for m in self.model.modules():
            if not isinstance(m, nn.Linear):
                continue
            w = grad(loss, m.weight, create_graph=True)[0]
            b = grad(loss, m.bias, create_graph=True)[0]
            grads.append([w,b])
        return grads
    
    
    # This function trains the INN, validation and logging included
    def train(self, n_epochs, train_loader, val_loader, log_writer=None):
        # Set number of epochs
        self.n_epochs = n_epochs
        
        # Training loop
        for i_epoch in range(n_epochs):
            # Training for one epoch
            loss, l_y, l_z, l_x_1, l_x_2 = self.train_epoch(train_loader, i_epoch)
            print('{} Loss: {} l_x_2: {} l_y: {} l_z: {} l_x_1: {}'.format(i_epoch, loss, l_x_2, l_y, l_z, l_x_1))
            
            # Validation and logging
            
            # Compute validation output every 5 epochs 
            self.model.eval()
            if i_epoch % 5 == 0:
                
                # Get validation data
                x_samps = torch.cat([x for x,y in val_loader], dim=0)
                y_samps = torch.cat([y  for x,y in val_loader], dim=0)
                N_samp  = x_samps.shape[0]
                
                x_samps_val = x_samps
                y_samps_val = y_samps
                
                # Pad validation data
                x_samps_val_pad = torch.cat((x_samps_val, torch.zeros(N_samp, self.ndim_tot - self.ndim_x).to(self.device)), dim=1)
                y_samps_val_pad = torch.cat([torch.randn(N_samp, self.ndim_z).to(self.device),
                                     self.zeros_noise_scale * torch.zeros(N_samp, self.ndim_tot - self.ndim_y - self.ndim_z).to(self.device), y_samps_val], dim=1)
                y_samps_val_pad = y_samps_val_pad.to(self.device)
                x_samps_val_pad = x_samps_val_pad.to(self.device)
                
                # Compute output of both directions, with a single z
                rev_x, _ = self.model(y_samps_val_pad, rev=True)
                out_y, _ = self.model(x_samps_val_pad)
                predicted_y = out_y[:, -self.ndim_y:]
                predicted_x = rev_x[:,:x_samps_val.shape[1]]
                ground_truth_x = x_samps_val
                ground_truth_y = y_samps_val
                
                # Compute validation losses
                val_loss_x = torch.nn.MSELoss()(predicted_x, ground_truth_x)
                val_loss_y = torch.nn.MSELoss()(predicted_y, ground_truth_y)
                print('val loss x: {} val loss y {}'.format(val_loss_x, val_loss_y))
                    
                # Log validation loss
                if log_writer != None:
                    if log_writer[0] == 'tensorboard':
                        log_writer[1].add_scalar('val Loss x', val_loss_x.item(), i_epoch)
                        log_writer[1].add_scalar('val Loss y', val_loss_y.item(), i_epoch)
                        log_writer[1].add_scalar('val loss y ssim', val_loss_y_ssim.item(), i_epoch)
                        log_writer[1].add_scalar('val loss y sdtw', val_loss_y_sdtw.item(), i_epoch)
                    if log_writer[0] == 'wb':
                        log_writer[1].log({'val Loss x': val_loss_x.item(), 'epoch': i_epoch})
                        log_writer[1].log({'val Loss y': val_loss_y.item(), 'epoch': i_epoch})
                        log_writer[1].log({'val loss ssim y': val_loss_y_ssim.item(), 'epoch': i_epoch})
                        log_writer[1].log({'val loss sdtw y': val_loss_y_sdtw.item(), 'epoch': i_epoch})

            
            # Log all training loss terms
            if log_writer != None:
                if log_writer[0] == 'tensorboard':
                    log_writer[1].add_scalar('Loss', loss, i_epoch)
                    log_writer[1].add_scalar('L_y', l_y, i_epoch)
                    log_writer[1].add_scalar('L_z', l_z, i_epoch)
                    log_writer[1].add_scalar('L_x_1', l_x_1, i_epoch)
                    log_writer[1].add_scalar('L_x_2', l_x_2, i_epoch)
                if log_writer[0] == 'wb':
                    log_writer[1].log({'Loss': loss, 'epoch': i_epoch})
                    log_writer[1].log({'L_y': l_y, 'epoch': i_epoch})
                    log_writer[1].log({'L_z': l_z, 'epoch': i_epoch})
                    log_writer[1].log({'L_x_1': l_x_1, 'epoch': i_epoch})
                    log_writer[1].log({'L_x_2': l_x_2, 'epoch': i_epoch})
            
            # Create checkpoint every 300 epochs
            if i_epoch % 300 == 0 and i_epoch != 0:
                if log_writer[0] == 'wb':
                    model_name_path = Path('checkpoints/{}_{}_{}'.format(log_writer[1].run.id, log_writer[1].run.name, i_epoch))
                    torch.save(self.model.state_dict(), model_name_path)
        return
        
    # MMD Loss
    def MMD_multiscale(self, x, y, scale=1):
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2.*xx
        dyy = ry.t() + ry - 2.*yy
        dxy = rx.t() + ry - 2.*zz

        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device))

        for a in [0.05*scale, 0.2*scale, 0.9*scale]:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2.*XY)
    
    # MMD matrix multiscale loss, for testing purposes
    def MMD_matrix_multiscale(self, x, y, widths_exponents):
        xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
        dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
        dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device),
                      torch.zeros(xx.shape).to(self.device))

        for C,a in widths_exponents:
            XX += C**a * ((C + dxx) / a)**-a
            YY += C**a * ((C + dyy) / a)**-a
            XY += C**a * ((C + dxy) / a)**-a

        return XX + YY - 2.*XY
    
    # Train a single epoch
    def train_epoch(self, train_loader, i_epoch=0):
        self.model.train()

        # Init loss variables for logging only
        l_tot = 0
        l_y = 0
        l_z = 0
        l_x_1 = 0
        l_x_2 = 0
        batch_idx = 0

        # If MMD on x-space is present from the start, the self.model can get stuck.
        #Instead, ramp it up exponetially.  
        loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / self.n_epochs)))
        #loss_factor = 1
        
        # Iterate over batches
        for x, y in train_loader:
            size_batch, dim_x = x.shape
            size_batch, dim_y = y.shape

            batch_idx += 1
            self.optimizer.zero_grad()
            y_clean = y.clone()
            
            # Padding for x (parameters)
            self.pad_x = self.zeros_noise_scale * torch.randn(size_batch, self.ndim_tot -
                                                    self.ndim_x, device=self.device)
            # Padding for  y (energy distribuiton) wtih z
            self.pad_yz = self.zeros_noise_scale * torch.randn(size_batch, self.ndim_tot -
                                                     self.ndim_y - self.ndim_z, device=self.device)
            
            # Add noise to y (energy distribution) and z, for robustness
            y += self.y_noise_scale * torch.randn(size_batch, self.ndim_y, dtype=torch.float, device=self.device)           
            
            x, y = (torch.cat((x, self.pad_x),  dim=1),
                    torch.cat((torch.randn(size_batch, self.ndim_z, device=self.device), self.pad_yz, y),
                              dim=1))
            # Forward step:
            output, _ = self.model(x)

            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)
            
            # Compute loss supervised loss for direciton parameter (x) -> energy distribution (y)
            l = self.loss_fit(output[:, self.ndim_z:], y[:, self.ndim_z:])
            l_y += l.data.item()
            l_tot += l.data.item()
            l = self.lambd_predict * l
            
            # Print gradients, for weighting of loss terms
            if self.debug_grad == True:
                print('l_y: ', np.mean([(torch.mean(x[0]).cpu().detach().numpy(),torch.mean(x[1]).cpu().detach().numpy()) for x in self.loss_grad(l)]))
            
            # Compute MMD loss for z
            output_block_grad = torch.cat((output[:, :self.ndim_z],

                                           output[:, -self.ndim_y:].data), dim=1)
            
            l_latent = self.loss_latent(output_block_grad, y_short, scale=self.scale)
            l += self.lambd_latent * l_latent
            l_tot += l_latent.data.item()
            l_z += l_latent.data.item()
    
            # Print gradients, for weighting of loss terms
            if self.debug_grad == True:
                print('l_z: ', np.mean([(torch.mean(x[0]).cpu().detach().numpy(),torch.mean(x[1]).cpu().detach().numpy()) for x in self.loss_grad(l_latent)]))

            l.backward(retain_graph=True)
            
            # Backward step:
            self.pad_yz = self.zeros_noise_scale * torch.randn(size_batch, self.ndim_tot -
                                                     self.ndim_y - self.ndim_z, device=self.device)
            y = y_clean #+ self.y_noise_scale * torch.randn(self.batch_size, self.ndim_y, device=device)
            orig_z_perturbed = (output.data[:, :self.ndim_z] + self.y_noise_scale *
                                torch.randn(size_batch, self.ndim_z, device=self.device))
            
            # y (energy distribution) with z from forward step+noise
            y_rev = torch.cat((orig_z_perturbed, self.pad_yz,
                                y), dim=1)
            # y (energy distribution) with new z
            y_rev_rand = torch.cat((torch.randn(size_batch, self.ndim_z, device=self.device), self.pad_yz,
                                    y), dim=1)

            # Compute output (parameters) for both
            output_rev, _ = self.model(y_rev, rev=True)
            output_rev_rand, _ = self.model(y_rev_rand, rev=True)
            
            # MMD Loss for x (parameter) + padding
            l_rev = self.loss_backward(output_rev_rand[:, :self.ndim_x],
                                x[:, :self.ndim_x], scale=self.scale)

            l_x_1 += l_rev.data.item()
            l_tot += l_rev.data.item()
            l_rev = self.lambd_rev * loss_factor * l_rev
            
            # Print gradients, for weighting of loss terms
            if self.debug_grad == True:
                print('l_x_1: ', np.mean([(torch.mean(x[0]).cpu().detach().numpy(),torch.mean(x[1]).cpu().detach().numpy()) for x in self.loss_grad(l_rev)]))
            
            # Supervised loss for the y (energy distribution) + z -> x (parameter) direction
            l_rev_2 = self.loss_fit_back(output_rev, x)
            l_rev += self.lambd_predict_back * l_rev_2 
            l_tot += l_rev_2.data.item()
            l_x_2 += l_rev_2.data.item()
            
            # Print gradients, for weighting of loss terms
            if self.debug_grad == True:
                print('l_x_2: ', np.mean([(torch.mean(x[0]).cpu().detach().numpy(),torch.mean(x[1]).cpu().detach().numpy()) for x in self.loss_grad(l_rev_2)]))
                      
            l_rev.backward()
            
            # Clamp gradients
            for p in self.model.parameters():
                p.grad.data.clamp_(-15.00, 15.00)
            self.optimizer.step()
        
        # Return losses for logging
        return l_tot / batch_idx, l_y/ batch_idx, l_z/ batch_idx, l_x_1/ batch_idx, l_x_2/ batch_idx