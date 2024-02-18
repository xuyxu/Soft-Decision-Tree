import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SDT import SDT
import os
from dataset import get_mnist, get_celeba, get_stl_star


def train_and_evaluate(args):
    if args.use_cuda and torch.cuda.is_available():
        print('Using CUDA')
    else:
        print('Using CPU')

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    # Load datasets
    if args.dataset == 'MNIST':
        args.input_dim = 28 * 28  # MNIST images are 28x28
        args.output_dim = 10
        train_loader, val_loader, test_loader = get_mnist(
            args.data_dir, args.batch_size, args.output_dim)
    elif args.dataset == 'CELEBA':
        args.input_dim = 96 * 96 * 3  # CELEBA images dimensions
        args.output_dim = 2
        train_loader, val_loader, test_loader = get_celeba(
            feature_idx=args.feature_idx, data_dir=args.data_dir,
            batch_size=args.batch_size, num_train=120_000, num_test=10_000)

    elif args.dataset == 'STL_STAR':
        args.input_dim = 96 * 96 * 3  # STL_STAR image size
        args.output_dim = 2
        train_loader, val_loader, test_loader = get_stl_star(
            data_dir=args.data_dir, batch_size=args.batch_size)

    # Initialize model
    tree = SDT(args.input_dim, args.output_dim, args.depth,
               args.lamda, args.use_cuda).to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(
        tree.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    best_testing_acc = 0.0
    for epoch in range(args.epochs):
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1,
                                     args.input_dim).to(device), target.to(device)

            output, penalty = tree(data, is_training_data=True)

            loss = criterion(output, target) + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # Validation set Evaluation
        tree.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.view(-1,
                                         args.input_dim).to(device), target.to(device)

                output = tree(data, is_training_data=False)
                pred = output.argmax(dim=1, keepdim=True)
                target_indices = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_indices).sum().item()

        accuracy = 100. * correct / len(val_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy

        print(f'\nVal set: Epoch: {epoch} Accuracy: {correct}/{len(val_loader.dataset)} '
              f'({accuracy:.0f}%) Best: {best_testing_acc:.0f}%\n')
    # Evaluation
    tree.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1,
                                     args.input_dim).to(device), target.to(device)
            output = tree(data, is_training_data=False)
            pred = output.argmax(dim=1, keepdim=True)
            target_indices = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_indices).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    if accuracy > best_testing_acc:
        best_testing_acc = accuracy

    print(f'\nTest set: Epoch: {epoch} Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%) Best: {best_testing_acc:.0f}%\n')

    print(f'Saving model to: {args.save_model_path}')
    torch.save(tree.state_dict(), args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training a Soft Decision Tree on MNIST or CELEBA")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets'),
                        help='Directory for storing input data')
    parser.add_argument('--dataset', type=str, choices=[
                        'MNIST', 'CELEBA', 'STL_STAR'], default='MNIST', help='Dataset to use.')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index for CelebA dataset')
    parser.add_argument('--input_dim', type=int, default=28*28,
                        help='Input dimension size. Will be overridden based on dataset.')
    parser.add_argument('--output_dim', type=int, default=10,
                        help='Output dimension size (number of classes). This will be overriden based on dataset')
    parser.add_argument('--depth', type=int, default=5,
                        help='Depth of the tree.')
    parser.add_argument('--lamda', type=float, default=1e-3,
                        help='Regularization coefficient.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to train.')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of times to wait before early stopping')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='How many batches to wait before logging training status.')
    parser.add_argument('--save_model_path', type=str,
                        default='stl_star_model.pth', help='Path to save the trained model.')
    parser.add_argument('--use_cuda', action='store_true',
                        default=False, help='Enable CUDA if available.')
    args = parser.parse_args()

    train_and_evaluate(args)
