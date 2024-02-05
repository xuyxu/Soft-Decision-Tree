import torch
import torch.nn.functional as F
import numpy as np
from SDT import SDT


class XG_SDT:

    def __init__(self, n_trees: int = 3, lr: float = 0.01, internal_lr: float = 0.01) -> None:
        self.n_trees = n_trees
        self.lr = lr
        self.internal_lr = internal_lr

    def train_tree(self, X_train, y_train):
        tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)

        optimizer = torch.optim.Adam(tree.parameters(),
                                     lr=self.internal_lr,
                                     weight_decay=weight_decaly)

    def train(self, X_train, y_train):
        # Assume `X_train` and `y_train` are your training data and labels, respectively
        # `n_trees` is the number of trees you want in your ensemble
        # `learning_rate` is the learning rate for the gradient boosting process

        # train initial tree on actual labels
        ensemble_predictions = self.train_tree(X_train, y_train)

        for tree_num in range(self.n_trees):
            # Compute pseudo-residuals as gradients (with cross-entropy it is residuals)
            gradients = self.compute_gradients(ensemble_predictions, y_train)

            # Train a new soft decision tree on these gradients
            new_tree = self.train_tree(X_train, gradients)

            # Update ensemble predictions
            # Get predictions for the new tree
            tree_predictions = new_tree.predict(X_train)
            ensemble_predictions += self.lr * tree_predictions

            # Optionally: Normalize ensemble_predictions if necessary

        # Convert final ensemble predictions to probabilities and make a classification decision
        final_probabilities = F.softmax(ensemble_predictions, dim=1)
        final_classes = np.argmax(final_probabilities, axis=1)

    def train_tree(self):
        return

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
