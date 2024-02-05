import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from SDT import SDT  # Ensure SDT.py is in the same directory or adjust this import


def onehot_coding(target, device, output_dim):
    """Convert class labels into one-hot encoded vectors."""
    target_onehot = torch.FloatTensor(target.size(0), output_dim).to(device)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    return target_onehot


def main():
    # Parameters
    input_dim = 28 * 28  # MNIST images are 28x28
    output_dim = 10  # MNIST has 10 classes
    depth = 5
    lamda = 1e-3
    lr = 1e-3
    weight_decay = 5e-4
    batch_size = 128
    epochs = 50
    log_interval = 100
    use_cuda = torch.cuda.is_available()

    # Model and Optimizer
    device = torch.device("cuda" if use_cuda else "cpu")
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda).to(device)
    optimizer = torch.optim.Adam(
        tree.parameters(), lr=lr, weight_decay=weight_decay)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True,
                       download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1,
                                     input_dim).to(device), target.to(device)
            optimizer.zero_grad()
            output, penalty = tree(data, is_training_data=True)
            loss = criterion(output, target) + lamda * penalty
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Evaluation loop
    tree.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1,
                                     input_dim).to(device), target.to(device)
            output = tree(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')


if __name__ == "__main__":
    main()
