import torch
import torch.nn.functional as F
import numpy as np
from SDT import SDT
import copy


class GB_SDT:
    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 3, lr: float = 0.001, internal_lr: float = 0.01,
                 depth: int = 4, lamda: float = 1e-3, weight_decay: float = 5e-4, epochs: int = 50, log_interval: int = 10, use_cuda: bool = False):
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

    def state_dict(self):
        """Returns a dictionary containing a whole state of the ensemble."""
        ensemble_state = {}
        for idx, tree in enumerate(self.trees):
            ensemble_state[f'tree_{idx}'] = tree.state_dict()
        return ensemble_state

    def load_state_dict(self, state_dict):
        """Loads the model's state from a state_dict."""
        for idx, tree_state in state_dict.items():
            # Assuming each tree is already instantiated and part of `self.trees`
            # You may need to adjust this if trees need to be instantiated here
            new_tree = SDT(self.input_dim, self.output_dim,
                           self.depth, self.lamda, self.use_cuda)
            new_tree.load_state_dict(tree_state)
            self.trees.append(new_tree)

    def train_tree(self, train_loader, val_loader, test_loader, early_stopping_rounds=5, log_interval=100):
        """
        Train a single decision tree with early stopping on a validation set and logging at specified intervals.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of maximum epochs to train.
            early_stopping_rounds (int): Number of rounds to stop after if no improvement in validation loss.
            log_interval (int): Number of batches to wait before logging training status.
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

    def evaluate(self, model, data_loader, criterion):
        """
        Evaluate the model on a given dataset.

        Args:
            model: The model to evaluate.
            data_loader (DataLoader): DataLoader for the dataset to evaluate on.
            criterion: Loss function to use for evaluation.

        Returns:
            A tuple containing the average loss and accuracy over the dataset.
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

    def predict(self, X):
        # Aggregate predictions from all trees
        X = X.to(self.device)
        ensemble_predictions = torch.zeros(
            X.size(0), self.output_dim).to(self.device)
        for tree in self.trees:
            tree.eval()
            output = tree(X)
            ensemble_predictions += output

        return F.softmax(ensemble_predictions, dim=1)

    def set_train(self):
        """Sets the module in training mode."""
        for tree in self.trees:
            tree.train()
        return self

    def eval(self):
        for tree in self.trees:
            tree.train()

    def compute_gradients(self, predictions, true_labels):
        """
        Compute the gradient of the cross-entropy loss with respect to the predictions.

        Args:
        - predictions (torch.Tensor): The model's predicted probabilities for each class.
        - true_labels (torch.Tensor): The actual labels in one-hot encoded format.

        Returns:
        - gradients (torch.Tensor): The gradients of the loss with respect to the predictions.
        """
        # Assuming predictions are logits (unnormalized scores), use log_softmax to convert to log probabilities
        probs = F.softmax(predictions, dim=1)

        # Compute the gradients as the difference between predicted probabilities and actual labels
        gradients = torch.exp(probs) - true_labels

        return gradients
