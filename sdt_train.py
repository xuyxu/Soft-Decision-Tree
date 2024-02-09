import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from SDT import SDT  # Assuming SDT is a custom model you've defined


def onehot_coding(target, device, output_dim):
    """Convert the class labels into one-hot encoded vectors."""
    target_onehot = torch.FloatTensor(target.size(0), output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.0)
    return target_onehot


def train_and_evaluate(args):
    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    # Model
    tree = SDT(args.input_dim, args.output_dim, args.depth,
               args.lamda, args.use_cuda).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        tree.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Data loading
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True,
                       download=True, transform=transformer),
        batch_size=args.batch_size, shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=transformer),
        batch_size=args.batch_size, shuffle=True,
    )

    best_testing_acc = 0.0

    for epoch in range(args.epochs):
        # Training
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28*28).to(device), target.to(device)
            target_onehot = onehot_coding(target, device, args.output_dim)

            output, penalty = tree(data, is_training_data=True)

            loss = criterion(output, target) + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # Evaluation
        tree.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.view(-1, 28 *
                                         28).to(device), target.to(device)
                output = tree(data, is_training_data=False)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy

        print(f'\nTest set: Epoch: {epoch} Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.0f}%) Best: {best_testing_acc:.0f}%\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training a Soft Decision Tree on MNIST")
    parser.add_argument('--data_dir', type=str, default='../Dataset/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--input_dim', type=int, default=28 *
                        28, help='Input dimension size.')
    parser.add_argument('--output_dim', type=int, default=10,
                        help='Output dimension size (number of classes).')
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
    parser.add_argument('--log_interval', type=int, default=100,
                        help='How many batches to wait before logging training status.')
    parser.add_argument('--use_cuda', action='store_true',
                        default=True, help='Enable CUDA if available.')
    args = parser.parse_args()

    train_and_evaluate(args)
