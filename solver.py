"""solver.py"""

import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import StructurePreservingNet
from dataset import load_dataset, split_dataset
from utilities.plot import plot_results
from utilities.utils import print_mse_data, print_mse_degeneracy


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Study Case
        self.sys_name = args.sys_name

        # Dataset Parameters
        self.dataset = load_dataset(args)
        self.dim_z = self.dataset.dim_z
        self.dim_t = self.dataset.dim_t
        self.L = self.dataset.L
        self.M = self.dataset.M
        self.dt = self.dataset.dt
        self.t_vec = self.dataset.t_vec

        self.train_trajectories, self.test_trajectories = split_dataset(args.train_percent, self.dataset.total_trajectories)
        if (args.dset_norm):
            self.mean, self.std = self.dataset.get_statistics(self.train_trajectories)
        else:
            self.mean, self.std = 0, 1

        # Training Parameters
        self.max_epoch = args.max_epoch
        self.lambda_d = args.lambda_d

        # Net Parameters
        self.dim_in = self.dim_z
        self.dim_out = 2*(self.dim_z)**2
        self.SPNN = StructurePreservingNet(self.dim_in, self.dim_out, args.hidden_vec, args.activation).float() 
 
        if (args.train == False):
            # Load pretrained net
            load_name = 'net_' + self.sys_name + '.pt'
            load_path = os.path.join(args.dset_dir, load_name)
            self.SPNN.load_state_dict(torch.load(load_path)) 
        else:
            self.SPNN.weight_init(args.net_init)
        
        self.optim = optim.Adam(self.SPNN.parameters(), lr=args.lr, weight_decay=args.lambda_r) 
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.miles, gamma=args.gamma)
                       
        # Load/Save options
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots = args.save_plots


    def train_model(self):
        epoch = 1
        start = time.time()
        log_epoch, log_time, log_loss_data, log_loss_degeneracy = [], [], [], []

        print("\n[Training Started]\n")
        
        # Main training loop
        while (epoch <= self.max_epoch):
            loss_data_sum, loss_degeneracy_sum = 0, 0

            # Trajectory loop
            for trajectory in self.train_trajectories:
                z_gt, z1_gt = self.dataset[trajectory]
                z, z1 = self.normalize(z_gt), self.normalize(z1_gt)

                # Net forward pass
                z1_net, deg_E, deg_S = self.SPNN(z, self.L, self.M, self.dt)

                # Compute loss
                loss_data = ((z1 - z1_net)**2).sum()
                loss_degeneracy = (deg_E**2).sum() + (deg_S**2).sum()
                loss = self.lambda_d*loss_data + loss_degeneracy

                loss_data_sum += loss_data
                loss_degeneracy_sum += loss_degeneracy

                # Backpropagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # Learning rate scheduler
            self.scheduler.step()

            # Print epoch error
            loss_data_mean = loss_data_sum / (self.dim_t - 1) / len(self.train_trajectories)
            loss_degeneracy_mean = loss_degeneracy_sum / (self.dim_t - 1) / len(self.train_trajectories)
            print('Epoch [{}/{}], Data Loss: {:1.2e}, Degeneracy Loss: {:1.2e}'.format(
                epoch, int(self.max_epoch), loss_data_mean, loss_degeneracy_mean))

            # Log epoch results
            end = time.time()
            log_epoch.append(epoch)
            log_time.append(end - start) 
            log_loss_data.append(loss_data_mean.item())
            log_loss_degeneracy.append(loss_degeneracy_mean.item())

            epoch += 1

        print("\n[Training Finished]\n")
        print("[Train Set Evaluation]\n")

        # Compute train trajectories
        z_real, z_net, deg_E, deg_S = self.integrator_loop(self.train_trajectories)

        # Compute train error
        print_mse_data(z_real, z_net)
        print_mse_degeneracy(deg_E, deg_S)

        # Save net
        file_name = 'net_' + self.sys_name + '.pt'
        save_dir = os.path.join(self.output_dir, file_name)
        torch.save(self.SPNN.state_dict(), save_dir)

        # Save logs
        file_name = 'log_' + self.sys_name + '.txt'
        save_dir = os.path.join(self.output_dir, file_name)
        f = open(save_dir, "w")
        f.write('epoch time loss_data loss_degeneracy loss_total\n')
        for idx in range(len(log_epoch)):
            f.write(str(log_epoch[idx]) + " " + str(log_time[idx]) + " ")
            f.write(str(log_loss_data[idx]) + " " + str(log_loss_degeneracy[idx]) + " ")
            f.write(str(log_loss_data[idx] + log_loss_degeneracy[idx]) + "\n")
        f.close()


    def test_model(self):
        print("\n[Test Set Evaluation]\n")

        # Compute test trajectories
        z_real, z_net, deg_E, deg_S = self.integrator_loop(self.test_trajectories)

        # Compute test error
        print_mse_data(z_real, z_net)
        print_mse_degeneracy(deg_E, deg_S)

        # Save plots
        if (self.save_plots):
            for trajectory in range(0, len(self.test_trajectories)):
                plot_name = 'Test Trajectory {}'.format(trajectory + 1)
                plot_results(self.output_dir, plot_name, z_real[trajectory,:,:], z_net[trajectory,:,:], self.t_vec, self.sys_name)

        print("\n[Test Finished]\n")


    def integrator_loop(self, trajectories):
        # Database initialization
        z_real = torch.zeros((len(trajectories), self.dim_z, self.dim_t))
        z_net = torch.zeros((len(trajectories), self.dim_z, self.dim_t))
        deg_E = torch.zeros((len(trajectories), self.dim_z, self.dim_t))
        deg_S = torch.zeros((len(trajectories), self.dim_z, self.dim_t))

        # Trajectory loop
        idx = 0
        for trajectory in trajectories:
            z_gt, z1_gt = self.dataset[trajectory]
            z_real[idx,:,0] = z_gt[0,:].detach()
            z_net[idx,:,0] = z_gt[0,:].detach()
            
            # Snapshot loop
            z = self.normalize(z_gt[0,:].view(-1,self.dim_z))
            for snapshot in range(self.dim_t - 1):
                # Net Forward pass
                z1_net, deg1, deg2 = self.SPNN(z, self.L, self.M, self.dt)

                # State Vector
                z_net[idx,:,snapshot + 1] = self.denormalize(z1_net.squeeze(0).detach())
                z_real[idx,:,snapshot + 1] = z1_gt[snapshot,:]

                # Degeneracy Conditions
                deg_E[idx,:,snapshot] = deg1.squeeze(0).detach()
                deg_S[idx,:,snapshot] = deg2.squeeze(0).detach()

                # Snapshot update
                z = z1_net
            idx += 1

        return z_real, z_net, deg_E, deg_S


    def normalize(self, z):
        return (z - self.mean) / self.std 


    def denormalize(self, z):
        return z * self.std + self.mean 

if __name__ == '__main__':
    pass


