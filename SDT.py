import torch
import torch.nn as nn


class SDT(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.

    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.

    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

    def __init__(
            self,
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
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim,
                                    bias=False)


    def forward(self, X, is_training_data=False):
        # Performs a forward pass through the model, deciding whether to include regularization penalty based on the context (training or evaluation)
        _mu, _penalty = self._forward(X)
        # Generates predictions based on the final path probabilities
        y_pred = self.leaf_nodes(_mu)

        if is_training_data:
            return y_pred, _penalty  # Returns predictions and penalty for training
        else:
            return y_pred  # Returns only predictions for evaluation


    def _forward(self, X):
        # Core implementation of the forward pass, calculating path probabilities through the tree and regularization penalty
        batch_size = X.size()[0]
        X = self._data_augment(X)  # Augments data with a bias term for each sample

        # Calculates initial routing probabilities using internal nodes
        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        # Splits probabilities for left and right routing
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        # Initializes path probabilities to 1 for all samples
        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        # Initializes regularization penalty
        _penalty = torch.tensor(0.0).to(self.device)

        begin_idx = 0
        end_idx = 1

        # Iterates through layers, updating path probabilities and penalties
        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Updates penalty based on current layer's paths
            _penalty += self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu *= _path_prob  # Updates path probabilities based on current decisions

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        # Finalizes path probabilities for leaf nodes
        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty  # Returns final path probabilities and total penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """

        penalty = torch.tensor(0.0).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))

    def compute_nfm(self, X):
        # Ensure model is in evaluation mode for consistent output
        self.eval()

        # We need to enable gradients for input for NFM computation
        X.requires_grad_(True)

        # Forward pass through the model
        mu, penalty = self._forward(X)
        y_pred = self.leaf_nodes(mu)

        # Initialize NFM as a zero tensor with the same size as the input
        nfm = torch.zeros_like(X)

        # Compute gradients for each output dimension
        for i in range(self.output_dim):
            self.zero_grad()  # Clear existing gradients
            # Backpropagate from each output dimension
            y_pred[:, i].sum().backward(retain_graph=True)

            # Sum gradients for each feature across all samples
            nfm += X.grad.data

        # Divide by the number of output dimensions to get the average influence
        nfm /= self.output_dim

        # Detach the NFM from the current graph to prevent further gradient computation
        nfm = nfm.detach()

        # Turn off gradients for input
        X.requires_grad_(False)

        return nfm.numpy()  # Return NFM as a NumPy array for analysis

    def compute_nfm_for_target(model, data_loader, target_class, device):
        """
        Compute the Neural Feature Map (NFM) for a specific target class.

        Args:
            model: The Soft Decision Tree model.
            data_loader: DataLoader providing the dataset.
            target_class: The target class for which to compute the NFM.
            device: The device (CPU or CUDA) on which to perform computations.

        Returns:
            A tensor representing the NFM for the specified target class.
        """
        model.eval()  # Set the model to evaluation mode
        feature_contributions = []  # List to store feature contributions

        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data if necessary

            # Forward pass through the model to get the paths and predictions
            # Assuming model.forward() has been modified to return paths or contributions
            output, paths = model.forward(data, return_paths=True)

            # Filter paths for the specific target_class
            for i in range(len(data)):
                if targets[i] == target_class:
                    # Assuming `paths` contains contribution info per input
                    # Modify as per your implementation
                    feature_contributions.append(paths[i])

        # Aggregate feature contributions across all filtered instances
        nfm = torch.mean(torch.stack(feature_contributions), dim=0)

        return nfm
