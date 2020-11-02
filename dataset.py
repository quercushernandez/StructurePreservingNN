"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset


class GroundTruthDataset(Dataset):
    def __init__(self, dset_dir):
        self.mat_data = scipy.io.loadmat(dset_dir)

        # Load Ground Truth simulations from Matlab    
        self.Z = torch.from_numpy(self.mat_data['Z']).float()
        self.L = torch.from_numpy(self.mat_data['L']).float()
        self.M = torch.from_numpy(self.mat_data['M']).float()
        self.t_vec = self.mat_data['t_vec']

        # Extract relevant dimensions and lengths of the problem
        self.dt = self.mat_data['dt'][0,0]
        self.dim_z = self.L.shape[1]
        self.dim_t = self.t_vec.shape[1]
        
        self.total_trajectories, _, _ = self.Z.shape
        self.len = self.total_trajectories
    
    def __getitem__(self, trajectory):
        # Space state vector
        z = self.Z[trajectory,:,:]

        # Batched state vectors: z(t) and z(t+1)
        return z[:,0:-1].T, z[:,1:].T

    def __len__(self):
        return self.len

    def get_statistics(self, trajectories):
        mean = torch.mean(self.Z[trajectories],[0,2])
        std = torch.std(self.Z[trajectories],[0,2])
        return mean, std


def load_dataset(args):
    # Dataset directory path
    sys_name = args.sys_name
    dset_dir = os.path.join(args.dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(dset_dir)

    return dataset


def split_dataset(p, total_trajectories):
    # Train and test trajectories
    train_trajectories = int(p*total_trajectories)

    # Random split
    indices = list(range(total_trajectories))
    np.random.shuffle(indices)

    train_indices = indices[:train_trajectories]
    test_indices = indices[train_trajectories:total_trajectories]

    return train_indices, test_indices


if __name__ == '__main__':
    pass

