""" Training and evaluating a soft decision tree on the MNIST dataset. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from SDT import SDT


def onehot_coding(target, device, output_dim):
    """ 
      Convert the class labels into one-hot encoded vectors.
    """
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot


if __name__ == '__main__':
    
    # Parameters
    input_dim = 28 * 28
    output_dim = 10
    depth = 5
    lamda = 1e-3
    use_cuda = False
    
    lr = 1e-3
    weight_decaly = 5e-4
    
    batch_size = 128
    epochs = 50
    log_interval = 100
    
    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)
    
    optimizer = torch.optim.Adam(tree.parameters(), 
                                 lr=lr, 
                                 weight_decay=weight_decaly)
    
    # Load data
    data_dir = '../Dataset/mnist'
    
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),
                                                           (0.3081,))])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, 
                       transform=transformer), batch_size=batch_size, 
                       shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, 
                       transform=transformer), batch_size=batch_size,
                       shuffle=True)
    
    # Utils
    best_testing_acc = 0.
    testing_acc_list = []
    training_loss_list = []
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    for epoch in range(epochs):
        
        # Training stage
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)
            target_onehot = onehot_coding(target, device, output_dim)
            
            output, penalty = tree.forward(data, is_training_data=True)

            loss = criterion(output, target.view(-1))
            loss += penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print training status
            if batch_idx % log_interval == 0:
                pred = output.data.max(1)[1]
                correct = pred.eq(target.view(-1).data).sum()
                
                msg = ('Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |'
                       ' Correct: {:03d}/{:03d}')
                print(msg.format(epoch, batch_idx, loss, correct, batch_size))
                training_loss_list.append(loss.cpu().data.numpy())
        
        # Compute the testing accuracy after each training epoch
        tree.eval()
        correct = 0.
        
        for batch_idx, (data, target) in enumerate(test_loader):
            
            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)
            
            output = F.softmax(tree.forward(data), dim=1)
            
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
            
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        
        msg = ('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) |'
               ' Historical Best: {:.3f}%\n')
        print(msg.format(epoch, correct, len(test_loader.dataset),
                         accuracy, best_testing_acc))
        testing_acc_list.append(accuracy)
