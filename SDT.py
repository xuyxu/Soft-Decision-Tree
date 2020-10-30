""" A fast implementation of soft decision tree. """

import torch
import torch.nn as nn


class SDT(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 depth=5, 
                 lamda=1e-3,
                 use_cuda=False):
        super(SDT, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth
        
        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-depth)) 
                             for depth in range(0, self.depth)] 
        
        # Initialize internal nodes and leaf nodes, the input dimension on 
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, 
                      self.internal_node_num_, bias=False),
            nn.Sigmoid())
        
        self.leaf_nodes = nn.Linear(self.leaf_node_num_, 
                                    self.output_dim, bias=False)
    
    def forward(self, X, is_training_data=False):
        
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)
        
        # When `X` is the training data, the model also returns the penalty
        # for computing the training loss.
        if is_training_data:
            return y_pred, _penalty 
        else:
            return y_pred
    
    """ 
      Implementation on the data forwarding process in the soft decision tree. 
    """
    def _forward(self, X):
        
        batch_size = X.size()[0]
        self._validate_parameters()
        
        X = self._data_augment(X)
        
        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)
        
        _mu = X.data.new(batch_size,1,1).fill_(1.)
        _penalty = torch.tensor(0.).to(self.device)
        
        # Iterate through nodes in each layer to compute the final path 
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1
        
        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            
            # Extract internal nodes in the current layer to compute the 
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            
            _mu = _mu * _path_prob  # update path probabilities
            
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx+1)
        
        mu = _mu.view(batch_size, self.leaf_node_num_)
        
        return mu, _penalty          
    
    """ 
      Compute the regularization term for internal nodes in different layers. 
    """
    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        
        penalty = torch.tensor(0.).to(self.device)   
        
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2**layer_idx)
        _path_prob = _path_prob.view(batch_size, 2**(layer_idx+1))
        
        for node in range(0, 2**(layer_idx+1)):
            alpha = (torch.sum(_path_prob[:, node] * _mu[:, node//2], dim=0) / 
                     torch.sum(_mu[:, node//2], dim=0))
            
            layer_penalty_coeff = self.penalty_list[layer_idx]
            
            penalty -= 0.5 * layer_penalty_coeff * (torch.log(alpha) + 
                                                    torch.log(1-alpha))
        
        return penalty
    
    """ Add a constant input `1` onto the front of each instance. """
    def _data_augment(self, X):
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)
        
        return X
    
    def _validate_parameters(self):
        pass
