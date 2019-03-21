# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from SDT import SDT
from torchvision import datasets, transforms


def onehot_coding(target, device, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot


if __name__ == '__main__':
    
    use_cuda = False
    
    learner_args = {'input_dim': 28*28,
                    'output_dim': 10,
                    'depth': 5,
                    'lamda': 1e-3,
                    'lr': 1e-3,
                    'weight_decay': 5e-4,
                    'batch_size': 128,
                    'epochs': 40,
                    'cuda': use_cuda,
                    'log_interval': 100}
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    tree = SDT(learner_args).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Load data
    data_dir = '../Dataset/mnist'
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=True, download=True,
                                                                      transform=transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))])),  # data normalization
                                                       batch_size=learner_args['batch_size'],
                                                       shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=False,
                                                                     transform=transforms.Compose([
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))])),
                                                      batch_size=learner_args['batch_size'],
                                                      shuffle=True)
    # Utility variables
    best_testing_acc = 0.
    testing_acc_list = []
    training_loss_list = []
    
    for epoch in range(1, learner_args['epochs']+1):
        
        # Training stage
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target_onehot = onehot_coding(target, device, learner_args['output_dim'])
            output = tree.forward(data)
            
            tree.optimizer.zero_grad()
            loss = criterion(output, target.view(-1))
            loss.backward()
            tree.optimizer.step()
            
            # Print intermediate training status
            if batch_idx % learner_args['log_interval'] == 0:
                with torch.no_grad():
                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()
                    loss = criterion(output, target.view(-1))
                    training_loss_list.append(loss.cpu().data.numpy())
                    print('Epoch: {:02d} | Batch: {:03d} | CrossEntropy-loss: {:.5f} | Correct: {}/{}'.format(
                            epoch, batch_idx, loss.data, correct, output.size()[0]))
        
        # Testing stage
        tree.eval()
        correct = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size()[0]
            output = tree.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        testing_acc_list.append(accuracy)
        print('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}%\n'.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))