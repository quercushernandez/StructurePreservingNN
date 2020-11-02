"""main.py"""

import argparse

import numpy as np
import torch

from solver import Solver
from utilities.utils import str2bool


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    SPNN_solver = Solver(args)

    if args.train:
        SPNN_solver.train_model()
    SPNN_solver.test_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Structure Preserving Neural Networks')

    # Study Case
    parser.add_argument('--sys_name', default='double_pendulum', type=str, help='physic system name')
    parser.add_argument('--train', default=False, type=str2bool, help='train or test')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--train_percent', default=0.8, type=float, help='dataset train snapshots split porcentage')
    parser.add_argument('--dset_norm', default=True, type=str2bool, help='dataset normalization')

    # Net Parameters
    parser.add_argument('--hidden_vec', default=5*[50], nargs='+', type=int, help='layer vector of hidden layers (excluding input and output layers)')
    parser.add_argument('--activation', default='relu', type=str, help='activation function')

    # Training Parameters
    parser.add_argument('--net_init', default='kaiming_uniform', type=str, help='net weight initialization')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lambda_r', default=1e-5, type=float, help='regularizer')
    parser.add_argument('--lambda_d', default=1e2, type=float, help='data loss weight')
    parser.add_argument('--max_epoch', default=6000, type=int, help='maximum training iterations')
    parser.add_argument('--miles', default=[2000, 4000], nargs='+', type=int, help='learning rate scheduler milestones')
    parser.add_argument('--gamma', default=1e-1, type=float, help='learning rate milestone decay')

    # Save and plot options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--save_plots', default=True, type=str2bool, help='save test plots')

    args = parser.parse_args()

    main(args)
