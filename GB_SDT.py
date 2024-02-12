import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SDT import SDT
import copy


class GB_SDT:
    """
    Gradient Boosted Soft Decision Trees (GB_SDT) implementation.

    This class implements an ensemble of soft decision trees using a gradient boosting approach.

    Attributes:
        input_dim (int): The number of input features.
        output_dim (int): The number of outputs (e.g., classes for classification).
        n_trees (int): The number of trees in the ensemble.
        lr (float): Learning rate for the ensemble's weight updates.
        internal_lr (float): Learning rate for training individual trees.
        depth (int): The depth of each tree in the ensemble.
        lamda (float): Regularization coefficient for tree training.
        weight_decay (float): Weight decay (L2 penalty) coefficient.
        epochs (int): Number of epochs to train each tree.
        log_interval (int): How often to log training progress.
        use_cuda (bool): Whether to use CUDA (GPU acceleration) if available.
        device (torch.device): Computation device (CPU or CUDA).
        trees (list): A list to store the trained trees.
    """

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 3, lr: float = 0.001, internal_lr: float = 0.01,
                 depth: int = 4, lamda: float = 1e-3, weight_decay: float = 5e-4, epochs: int = 50, log_interval: int = 10, use_cuda: bool = False):
        """
        Initializes the Gradient Boosted Soft Decision Trees ensemble.

        Parameters:
            input_dim (int): Number of features in the input data.
            output_dim (int): Number of target classes or output dimensions.
            n_trees (int): Number of trees to include in the ensemble.
            lr (float): Learning rate for ensemble optimization.
            internal_lr (float): Learning rate for training individual trees.
            depth (int): Depth of each decision tree.
            lamda (float): Regularization coefficient for trees' training.
            weight_decay (float): Coefficient for L2 regularization.
            epochs (int): Number of training epochs for each tree.
            log_interval (int): Interval for logging training progress.
            use_cuda (bool): Flag indicating whether to use CUDA for training.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_trees = n_trees
        self.lr = lr
        self.internal_lr = internal_lr
        self.depth = depth
        self.lamda = lamda
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.log_interval = log_interval
        self.use_cuda = use_cuda
        self.device = torch.device(
            'cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.trees = []

    def state_dict(self) -> dict:
        """
        Returns the state of the ensemble as a dictionary.

        This is useful for saving the model's state to a file.

        Returns:
            dict: A dictionary containing the state of each tree in the ensemble.
        """
        ensemble_state = {}
        for idx, tree in enumerate(self.trees):
            ensemble_state[f'tree_{idx}'] = tree.state_dict()
        return ensemble_state

    def load_state_dict(self, state_dict: dict):
        """
        Loads the model's state from a state_dict, allowing model weights to be loaded.

        Parameters:
            state_dict (dict): A dictionary containing the state of the ensemble.
        """
        for idx, tree_state in state_dict.items():
            # Assuming each tree is already instantiated and part of `self.trees`
            # You may need to adjust this if trees need to be instantiated here
            new_tree = SDT(self.input_dim, self.output_dim,
                           self.depth, self.lamda, self.use_cuda)
            new_tree.load_state_dict(tree_state)
            self.trees.append(new_tree)

    def train_tree(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, early_stopping_rounds: int = 5, log_interval: int = 100) -> SDT:
        """
        Trains a single decision tree with early stopping on a validation set.

        Parameters:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            early_stopping_rounds (int): Number of rounds to stop after if no improvement in validation loss.
            log_interval (int): Number of batches to wait before logging training status.

        Returns:
            SDT: The trained soft decision tree model.
        """
        tree = SDT(self.input_dim, self.output_dim,
                   self.depth, self.lamda, self.use_cuda)
        tree.to(self.device)

        optimizer = torch.optim.Adam(
            tree.parameters(), lr=self.internal_lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            tree.train()
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output, penalty = tree(data, is_training_data=True)
                loss = criterion(output, target) + penalty * self.lamda
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % log_interval == 0 and batch_idx > 0:
                    print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                          f'Loss: {running_loss / log_interval:.4f}')
                    running_loss = 0.0

            # Validation phase
            val_loss, val_acc = self.evaluate(tree, val_loader, criterion)
            print(f'Epoch: {epoch} Validation Loss: {val_loss:.4f}')
            print(f'Epoch: {epoch} Validation Accuracy: {val_acc:.2f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_wts = copy.deepcopy(tree.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stopping_rounds:
                    print(
                        f'Early stopping triggered after {epoch + 1} epochs.')
                    break

        tree.load_state_dict(best_model_wts)

        test_loss, test_accuracy = self.evaluate(tree, test_loader, criterion)
        print(
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        return tree

    def evaluate(self, model: SDT, data_loader: DataLoader, criterion: torch.nn.Module) -> (float, float):
        """
        Evaluates the model on a given dataset.

        Parameters:
            model (SDT): The model to evaluate.
            data_loader (DataLoader): DataLoader for the dataset to evaluate.
            criterion (torch.nn.Module): The loss function to use for evaluation.

        Returns:
            float: Average loss over the dataset.
            float: Accuracy percentage over the dataset.
        """
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data, is_training_data=False)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += data.size(0)

        average_loss = total_loss / total_samples
        accuracy = 100.0 * correct_predictions / total_samples
        return average_loss, accuracy

    def train(self, train_loader, val_loader, test_loader):
        for tree_num in range(self.n_trees):
            print(f"Training Tree {tree_num + 1}/{self.n_trees}")
            tree = self.train_tree(train_loader, val_loader, test_loader)
            self.trees.append(tree)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions with the ensemble by aggregating predictions from all trees.

        Parameters:
            X (torch.Tensor): Input features tensor.

        Returns:
            torch.Tensor: Predicted probabilities for each class, as an average over all trees.
        """
        X = X.to(self.device)
        ensemble_predictions = torch.zeros(
            X.size(0), self.output_dim).to(self.device)
        for tree in self.trees:
            tree.eval()
            output = tree(X)
            ensemble_predictions += output

        return F.softmax(ensemble_predictions, dim=1)

    def set_train(self):
        """
        Sets the ensemble to training mode.
        """
        for tree in self.trees:
            tree.train()
        return self

    def eval(self):
        """
        Sets the ensemble to evaluation mode.
        """
        for tree in self.trees:
            tree.train(False)

    def compute_gradients(self, predictions: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes gradients of the cross-entropy loss with respect to the predictions.

        Parameters:
            predictions (torch.Tensor): The model's predicted probabilities for each class.
            true_labels (torch.Tensor): The actual labels for the data.

        Returns:
            torch.Tensor: The gradients of the loss with respect to the predictions.
        """
        # Assuming predictions are logits (unnormalized scores), use log_softmax to convert to log probabilities
        probs = F.softmax(predictions, dim=1)

        # Compute the gradients as the difference between predicted probabilities and actual labels
        gradients = torch.exp(probs) - true_labels

        return gradients
