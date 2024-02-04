import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import grad

from SDT import SDT


def compute_jacobian(model, inputs):
    """Computes the Jacobian of the model's output with respect to its inputs."""
    jacobians = []
    for input in inputs:
        input = input.unsqueeze(0)  # Add batch dimension
        input.requires_grad = True

        # Forward pass
        # Assuming the model returns (output, penalty)
        output = model(input)[0]
        output = F.softmax(output, dim=0)

        # Compute Jacobian for each class
        for class_idx in range(output.size(0)):
            model.zero_grad()
            class_output = output[0, class_idx]
            class_jacobian = grad(class_output, input, retain_graph=True)[0]
            jacobians.append(class_jacobian.detach())

    jacobians = torch.stack(jacobians)
    return jacobians


def calculate_egop(model, data_loader, batch_size=100):
    """Computes the EGOP for the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Determine the flattened input size
    sample_input, _ = next(iter(data_loader))
    input_size = sample_input.view(sample_input.size(0), -1).size(1)

    G = torch.zeros((input_size, input_size), device=device)
    total_batches = len(data_loader)

    for inputs, _ in data_loader:
        print(f"Current Batch: {inputs.size(0)}/{data_loader.size(0)}")
        # Flatten and send to device
        inputs = inputs.view(inputs.size(0), -1).to(device)
        J = compute_jacobian(model, inputs)
        G += torch.einsum("bid,bjd->ij", J, J)

    G /= total_batches
    return G.cpu()


if __name__ == "__main__":
    # Parameters
    input_dim = 28 * 28    # the number of input dimensions
    output_dim = 10        # the number of outputs (i.e., # classes on MNIST)
    depth = 5              # tree depth
    lamda = 1e-3           # coefficient of the regularization term
    use_cuda = False       # whether to use GPU

    # Load the trained model
    model = SDT(input_dim, output_dim, depth, lamda, use_cuda)
    model.load_state_dict(torch.load('tree_state_dict.pth'))
    model.eval()  # Set the model to evaluation mode

    # Load MNIST data for EGOP calculation
    data_dir = "../Dataset/mnist"
    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transformer),
        batch_size=128,
        shuffle=True
    )

    # Flatten the images for the SDT model
    train_data_flat = torch.cat(
        [batch[0].view(batch[0].size(0), -1) for batch in train_loader], dim=0)

    # Calculate EGOP
    egop_matrix = calculate_egop(model, train_data_flat, batch_size=128)
    print("EGOP Matrix:\n", egop_matrix)
