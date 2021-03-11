
<div align="center">  
  
# Structure-Preserving Neural Networks

[![Project page](https://img.shields.io/badge/-Project%20page-blue)](https://amb.unizar.es/people/quercus-hernandez/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2004.04653.pdf)
[![JCP](https://img.shields.io/badge/JCP-2020-green)](https://www.sciencedirect.com/science/article/pii/S0021999120307245)

</div>

## Abstract

In this work, we propose a neural network model which learns the intrinsic physical
nature of nonlinear dynamical problems through the GENERIC formalism. The
GENERIC structure of an arbitrary system divides the problem in a conservative
term, related to the reversible evolution of the system (Hamiltonian mechanics), and
a dissipative term, related to the entropy or irreversible part of the system. Furthermore,
the degeneracy conditions of this formulation ensures the energy conservation
and the entropy inequality, fulfilling the first and second laws of thermodynamics
respectively.

The introduction of the GENERIC approach inside the neural network framework
allows us to take advantage of current machine learning methods as a solver, while
imposing the physical restrictions of GENERIC. In other words, we are learning the
physics of a system from measured data, ensuring that future estimations of the state
of that system will remain consistent, as they are imposed by those restrictions. We
provide some examples of nonlinear dynamical systems to show how our physicalbased
machine learning system is able to estimate the correct values, although the
real equation is not known for it.

For more information, please refer to the following:

- Hernández, Quercus and Badías, Alberto and González, David and Chinesta, Francisco and Cueto, Elías. "[Structure-preserving neural networks](https://www.sciencedirect.com/science/article/pii/S0021999120307245)." Journal of Computational Physics (2020).

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/quercushernandez/StructurePreservingNN.git
cd StructurePreservingNN
```

Then, install the needed dependencies. The code is implemented in [Pytorch](https://pytorch.org). _Note that this has been tested using Python 3.7_.

```bash
# install dependencies
pip install numpy scipy matplotlib pytorch
 ```

## How to run the code  

### Test pretrained nets

The results of the paper (Double Pendulum and Viscolastic Fluid) can be reproduced with the following scripts, found in the `executables/` folder.

```bash
python main.py --sys_name double_pendulum --train False --hidden_vec 200 200 200 200 200
python main.py --sys_name viscolastic --train False --hidden_vec 50 50 50 50 50 --dset_norm False
```

The `data/` folder includes the database and the pretrained parameters of the networks. The resulting time evolution of the state variables is plotted and saved in .png format in a generated `outputs/` folder.

|             Double Pendulum                  |         Viscolastic Fluid             |
| ---------------------------------------------|---------------------------------------|
|<div align="center"> <img src="/data/double_pendulum.png" width="500"></div>|<div align="center"> <img src="/data/viscolastic.png" width="500"></div>|

### Train a custom net

You can also run your own experiments for the implemented datasets by setting custom parameters manually. Several training examples can be found in the `executables/` folder. The manually trained parameters and output plots are saved in the `outputs/` folder.

```bash
e.g.
python main.py --sys_name double_pendulum --train True --lr 1e-3 ...
```

General Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--sys_name`              | Study case                                        | `double_pendulum`, `viscolastic`                      |
| `--train`                 | Train mode                                        | `True`, `False`                                       |
| `--dset_dir`              | Dataset and pretrained nets directory             | Default: `data`                                       |
| `--output_dir`            | Output data directory                             | Default: `output`                                     |
| `--save_plots`            | Save plots of state variables                     | `True`, `False`                                       |

Training Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--train_percent`         | Train porcentage of the full database             | Default: `0.8`                                        |
| `--dset_norm`             | Dataset normalization                             | `True`, `False`                                       |
| `--hidden_vec`            | Hidden layers vector                              | Default: `50 50 50 50 50`                             |
| `--activation`            | Activation functions of the hidden layers         | `linear`, `sigmoid`, `relu`, `tanh`                   |
| `--net_init`              | Net initialization method                         | `kaiming_normal`, `xavier_normal`                     |
| `--lr`                    | Learning rate                                     | Default: `1e-4`                                       |
| `--lambda_r`              | Weight decay regularizer                          | Default: `1e-5`                                       |
| `--lambda_d`              | Data loss weight                                  | Default: `1e2`                                        |
| `--max_epoch`             | Maximum number of training epochs                 | Default: `6e3`                                        |
| `--miles`                 | Learning rate scheduler milestones                | Default: `2e3 4e3`                                    |
| `--gamma`                 | Learning rate scheduler decay                     | Default: `1e-1`                                       |

## Citation

If you found this code useful please cite our work as:

```
@article{hernandez2021structure,
  title={Structure-preserving neural networks},
  author={Hernandez, Quercus and Badias, Alberto and Gonz{\'a}lez, David and Chinesta, Francisco and Cueto, El{\'\i}as},
  journal={Journal of Computational Physics},
  volume={426},
  pages={109950},
  year={2021},
  publisher={Elsevier}
}
```
