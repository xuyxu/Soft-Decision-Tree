import torch
import torch.nn as nn


class SDT(nn.Module):
    """
    Fast implementation of a soft decision tree in PyTorch.

    Attributes:
        input_dim (int): Number of input dimensions.
        output_dim (int): Number of output dimensions, e.g., number of classes in classification.
        depth (int): Depth of the tree, affecting its complexity.
        lamda (float): Regularization coefficient for the loss function.
        device (torch.device): Computation device (CPU or GPU).
        internal_node_num_ (int): Number of internal nodes in the tree.
        leaf_node_num_ (int): Number of leaf nodes in the tree.
        penalty_list (List[float]): Coefficients for regularization penalty of nodes at different depths.
        inner_nodes (nn.Sequential): Sequential model for internal nodes.
        leaf_nodes (nn.Linear): Linear layer representing leaf nodes.
    """

    def __init__(self, input_dim: int, output_dim: int, depth: int = 5, lamda: float = 1e-3, use_cuda: bool = False):
        """
        Initializes the Soft Decision Tree model.

        Parameters:
            input_dim (int): The number of features in the input data.
            output_dim (int): The number of target outputs or classes.
            depth (int): The depth of the tree, affecting the number of nodes.
            lamda (float): Regularization coefficient to control model complexity.
            use_cuda (bool): Flag to enable CUDA (GPU) computation.
        """
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        self.penalty_list = [self.lamda *
                             (2 ** (-depth)) for depth in range(self.depth)]

        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(
            self.leaf_node_num_, self.output_dim, bias=False)

    def forward(self, X: torch.Tensor, is_training_data: bool = False) -> torch.Tensor:
        """
        Performs a forward pass of the model.

        Parameters:
            X (torch.Tensor): Input data tensor.
            is_training_data (bool): Indicates if the pass is for training.

        Returns:
            torch.Tensor: The model's predictions. Includes penalty if is_training_data is True.
        """
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Core implementation of the model's forward pass.

        Parameters:
            X (torch.Tensor): Input data tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of path probabilities and total penalty.
        """
        batch_size = X.size(0)
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty += self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2) * _path_prob

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)
        return mu, _penalty

    def _cal_penalty(self, layer_idx: int, _mu: torch.Tensor, _path_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes regularization penalty for a given layer.

        Parameters:
            layer_idx (int): Index of the current tree layer.
            _mu (torch.Tensor): Path probabilities up to the current layer.
            _path_prob (torch.Tensor): Probabilities for routing at the current layer.

        Returns:
            torch.Tensor: Computed regularization penalty for the layer.
        """
        penalty = torch.tensor(0.0).to(self.device)
        batch_size = _mu.size(0)
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0) / torch.sum(_mu[:, node // 2], dim=0)
            coeff = self.penalty_list[layer_idx]
            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))
        return penalty

    def _data_augment(self, X: torch.Tensor) -> torch.Tensor:
        """
        Adds a constant bias term to the input data.

        Parameters:
            X (torch.Tensor): Original input data.

        Returns:
            torch.Tensor: Augmented input data.
        """
        batch_size = X.size(0)
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)
        return X

    def _validate_parameters(self):
        """
        Validates model parameters.
        """
        if not self.depth > 0:
            raise ValueError(
                f"The tree depth should be strictly positive, got {self.depth} instead.")
        if not self.lamda >= 0:
            raise ValueError(
                f"The coefficient of the regularization term should not be negative, got {self.lamda} instead.")

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

        return nfm.cpu().numpy()  # Return NFM as a NumPy array for analysis

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
