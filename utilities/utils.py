"""utils.py"""

import torch

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_mse_data(z_real, z_net):

    # Compute SE
    z_se = (z_real - z_net)**2

    # Compute MSE
    z_mse = torch.mean(z_se,(0,2))

    # Print MSE
    print('Data MSE')
    for variable in range(0, len(z_mse)):
        print('  State Variable {} MSE = {:1.2e}'.format(variable + 1, z_mse[variable]))


def print_mse_degeneracy(deg_E, deg_S):

    # Compute SE
    deg_E_se = (deg_E)**2
    deg_S_se = (deg_S)**2

    # Compute MSE
    deg_E_mse = torch.mean(deg_E_se,(0,2))
    deg_S_mse = torch.mean(deg_S_se,(0,2))

    # Print MSE
    print('Degeneracy MSE')
    for variable in range(len(deg_E_mse)):
        print('  State Variable {} MSE = {:1.2e}'.format(variable + 1, deg_E_mse[variable] + deg_S_mse[variable]))
