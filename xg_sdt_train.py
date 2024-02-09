import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import argparse
from GB_SDT import GB_SDT


def evaluate(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.predict(data)  # Assuming model has a predict method
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f'\nTest set: Accuracy: {accuracy:.2f}%\n')


def main(args):
    # Parameters from args
    input_dim = 28 * 28  # For MNIST
    output_dim = 10  # Number of classes in MNIST
    n_trees = args.n_trees
    depth = args.depth
    lr = args.lr  # Learning rate for the ensemble update, not individual tree training
    internal_lr = args.internal_lr  # Learning rate for training individual trees
    lamda = args.lamda
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    epochs = args.epochs
    log_interval = args.log_interval
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loading and augmentation setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Dataset loading
    full_train_dataset = datasets.MNIST(
        '../data', train=True, download=True, transform=transform)
    test_loader = DataLoader(datasets.MNIST(
        '../data', train=False, transform=transform), batch_size=batch_size, shuffle=False)

    # Splitting the dataset into training and validation
    validation_split = 0.2
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if args.shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data loaders
    train_loader = DataLoader(
        full_train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validation_loader = DataLoader(
        full_train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))

    # Model initialization
    model = GB_SDT(input_dim=input_dim, output_dim=output_dim, n_trees=n_trees, lr=lr, internal_lr=internal_lr,
                   depth=depth, lamda=lamda, weight_decay=weight_decay, epochs=epochs, log_interval=log_interval, use_cuda=use_cuda)
    # Assuming this method exists within GB_SDT
    model.train(train_loader, validation_loader, test_loader)

    # Testing the model
    evaluate(model, test_loader, device=device)

    # Saving the model
    torch.save(model.state_dict(), args.save_model_path)
    print(f'Model parameters saved to {args.save_model_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GB_SDT model on MNIST.")
    parser.add_argument('--n_trees', type=int, default=4,
                        help='Number of trees in the ensemble.')
    parser.add_argument('--depth', type=int, default=5,
                        help='Depth of each tree.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for ensemble update.')
    parser.add_argument('--internal_lr', type=float, default=0.001,
                        help='Learning rate for individual trees.')
    parser.add_argument('--lamda', type=float, default=1e-3,
                        help='Lambda for regularization.')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='Weight decay for optimization.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging during training.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Whether to shuffle the dataset.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--save_model_path', type=str,
                        default='GB_SDT_model.pth', help='Path to save the trained model.')
    args = parser.parse_args()
    main(args)
