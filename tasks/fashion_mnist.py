from __future__ import print_function

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from dataloader import *


class Net(nn.Module):
    """Inspired by many other designs, we design the following networks architecture:
    - two Conv layers of each block, first is to learn without changing dimension,
    second is to reduce by 2 times.
    - three CNN blocks stack together to learn feature representation, then
    final features are flatten and feed to a simplest classifier to 10 classes.
    """

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        hidden_channels = 24
        self.layer1a = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=hidden_channels,
                      kernel_size=3, padding=1),
            #nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        # H=(28+2-3)/1+1=28
        self.layer1b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3, padding=1),
            #nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # H=(28+2-3)/1+1=28
        # H=(28-2)/2+1=14
        # output shape=[batch_size, hidden_channels, 14, 14]

        self.layer2a = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels*2,
                      kernel_size=3, padding=1),
            #nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
        )
        # H=(14+2-3)/1+1=14
        self.layer2b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels*2,
                      out_channels=hidden_channels*2,
                      kernel_size=3, padding=0),
            #nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # H=(14+0-3)/1+1=12
        # H=(12-2)/2+1=6
        # output shape=[batch_size, hidden_channels*2, 6, 6]

        self.layer3a = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels*2,
                      out_channels=hidden_channels*4,
                      kernel_size=3, padding=1),
            #nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(),
        )
        # H=(6+2-3)/1+1=6
        self.layer3b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels*4,
                      out_channels=hidden_channels*4,
                      kernel_size=3, padding=0),
            #nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # H=(6+0-3)/1+1=4
        # H=(4-2)/2 + 1=2
        # output shape=[batch_size, hidden_channels*4, 2, 2]

        """No need dropout, ReLU, BatchNorm for this model after doing grid-search
        One layer is enough.
        """
        #fcs = [hidden_channels*4*2*2, 120, 84, 10]
        #fcs = [hidden_channels*4*2*2, 96, 10]
        fcs = [hidden_channels*4*2*2, 10]
        self.fc1 = nn.Linear(in_features=fcs[0],out_features=fcs[1])
        #self.fc2 = nn.Linear(in_features=fcs[1], out_features=fcs[2])
        #self.fc3 = nn.Linear(in_features=fcs[2], out_features=fcs[3])

    def forward(self, x):
        out = self.layer1a(x)
        out = self.layer1b(out)
        out = self.layer2a(out)
        out = self.layer2b(out)
        out = self.layer3a(out)
        out = self.layer3b(out)
        #out = out.view(out.size(0), -1)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        #out = self.fc2(out)
        #out = self.fc3(out)
        return out


def getDataset():
    dataset = datasets.FashionMNIST('./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))]))
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'],\
        'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients, loader_type)
            print('Save the dataloader {}'.format(path))
            with open(path, 'wb') as handle:
                pickle.dump(loader, handle)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients, loader_type)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data',
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))])),
            batch_size=test_batch_size,
            shuffle=True)
    return test_loader


if __name__ == '__main__':
    from torchinfo import summary

    print("#Initialize a network")
    net = Net()
    print(net)
    summary(net.cuda(), (1, 1, 28, 28))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'byLabel', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
