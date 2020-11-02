"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class StructurePreservingNet(nn.Module):
    """Structure Preserving Neural Network"""
    def __init__(self, dim_in, dim_out, hidden_vec, activation):
        super(StructurePreservingNet, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_vec = hidden_vec
        self.activation = activation
        self.layer_vec = [self.dim_in] + self.hidden_vec + [self.dim_out]
        self.activation_vec = (len(self.layer_vec)-2)*[self.activation] + ['linear']

        # Linear layers append from the layer vector
        self.fc_hidden_layers = nn.ModuleList()
        for k in range(len(self.layer_vec)-1):
            self.fc_hidden_layers.append(nn.Linear(self.layer_vec[k], self.layer_vec[k+1]))

    def forward(self, input, L, M, dt):
        x = input
        z = input.unsqueeze(2)
        idx = 0
        for layer in self.fc_hidden_layers:
            if self.activation_vec[idx] == 'linear': x = layer(x)
            elif self.activation_vec[idx] == 'sigmoid': x = torch.sigmoid(layer(x))
            elif self.activation_vec[idx] == 'relu': x = F.relu(layer(x))
            elif self.activation_vec[idx] == 'rrelu': x = F.rrelu(layer(x))
            elif self.activation_vec[idx] == 'tanh': x = torch.tanh(layer(x))
            elif self.activation_vec[idx] == 'sin': x = torch.sin(layer(x))
            elif self.activation_vec[idx] == 'elu': x = F.elu(layer(x))
            else: raise NotImplementedError
            idx += 1
        A_out, B_out = x[:,0:self.dim_in*self.dim_in], x[:,self.dim_in*self.dim_in:]
        A_out, B_out = A_out.view(-1,self.dim_in,self.dim_in), B_out.view(-1,self.dim_in,self.dim_in)

        DE = torch.bmm(A_out,z)
        DS = torch.bmm(B_out,z)
        L_batch = L.expand(z.size(0),z.size(1),z.size(1))
        M_batch = M.expand(z.size(0),z.size(1),z.size(1))
        z1_out = z + dt*(torch.bmm(L_batch,DE) + torch.bmm(M_batch,DS))

        deg_E = torch.bmm(M_batch,DE)
        deg_S = torch.bmm(L_batch,DS)

        return z1_out.view(-1,self.dim_in), deg_E.view(-1,self.dim_in), deg_S.view(-1,self.dim_in)

    def weight_init(self, net_initialization):
        for layer in self.fc_hidden_layers:
            if net_initialization == 'zeros': 
                init.constant_(layer.bias, 0)
                init.constant_(layer.weight, 0)
            elif net_initialization == 'xavier_normal': 
                init.constant_(layer.bias, 0)
                init.xavier_normal_(layer.weight)
            elif net_initialization == 'xavier_uniform': 
                init.constant_(layer.bias, 0)
                init.xavier_uniform_(layer.weight)
            elif net_initialization == 'kaiming_uniform': 
                init.constant_(layer.bias, 0)
                init.kaiming_uniform_(layer.weight)
            elif net_initialization == 'sparse': 
                init.constant_(layer.bias, 0)
                init.sparse_(layer.weight, sparsity = 0.1)  
            else:
                raise NotImplementedError


if __name__ == '__main__':
    pass
