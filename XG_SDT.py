import torch
import torch.nn.functional as F
import numpy as np
from SDT import SDT


class XG_SDT:

    def __init__(self, input_dim: int, output_dim: int, n_trees: int = 3, lr: float = 0.001, internal_lr: float = 0.01,
                 depth: int = 4, lamda: float = 1e-3, weight_decay: float = 5e-4, use_cuda: bool = False) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_trees = n_trees
        self.lr = lr
        self.internal_lr = internal_lr
        self.depth = depth
        self.lamda = lamda
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else "cpu"

    def train_tree(self, train_loader, test_loader, epochs, log_interval):
        # Initialize SDT and move it to the appropriate device
        tree = SDT(self.input_dim, self.output_dim,
                   self.depth, self.lamda, self.use_cuda)
        tree.to(self.device)

        # Optimizer setup
        optimizer = torch.optim.Adam(tree.parameters(),
                                     lr=self.internal_lr,
                                     weight_decay=self.weight_decay)

        # Tracking metrics
        best_testing_acc = 0.0
        testing_acc_list = []
        training_loss_list = []

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Training phase
            tree.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size = data.size()[0]
                data, target = data.to(self.device), target.to(self.device)
                output, penalty = tree.forward(data, is_training_data=True)

                loss = criterion(output, target.view(-1))
                loss += penalty * self.lamda  # Applying regularization penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                if batch_idx % log_interval == 0:
                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()
                    print(
                        f"Epoch: {epoch:02d} | Batch: {batch_idx:03d} | Loss: {loss.item():.5f} | Correct: {correct}/{batch_size}")
                    training_loss_list.append(loss.item())

            # Evaluation phase
            tree.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = tree(data)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.view(-1).data).sum()

            accuracy = 100.0 * correct / len(test_loader.dataset)
            if accuracy > best_testing_acc:
                best_testing_acc = accuracy

            print(
                f"\nEpoch: {epoch:02d} | Testing Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.3f}%) | Historical Best: {best_testing_acc:.3f}%\n")
            testing_acc_list.append(accuracy)

        return tree

    def train(self, X_train, y_train):
        # Assume `X_train` and `y_train` are your training data and labels, respectively
        # `n_trees` is the number of trees you want in your ensemble
        # `learning_rate` is the learning rate for the gradient boosting process

        # train initial tree on actual labels
        ensemble_predictions = self.train_tree(X_train, y_train)

        for tree_num in range(self.n_trees):
            print("*"*20)
            print(f"Training Tree {tree_num}")
            # Compute pseudo-residuals as gradients (with cross-entropy it is residuals)
            gradients = self.compute_gradients(ensemble_predictions, y_train)

            # Train a new soft decision tree on these gradients
            new_tree = self.train_tree(X_train, gradients)

            # Update ensemble predictions
            # Get predictions for the new tree
            tree_predictions = new_tree.predict(X_train)
            ensemble_predictions += self.lr * tree_predictions

            # Optionally: Normalize ensemble_predictions if necessary
            print("*"*20)

        # Convert final ensemble predictions to probabilities and make a classification decision
        final_probabilities = F.softmax(ensemble_predictions, dim=1)
        final_classes = np.argmax(final_probabilities, axis=1)

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
