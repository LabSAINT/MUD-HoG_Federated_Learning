from __future__ import print_function

from copy import deepcopy
from collections import deque

import torch
import torch.nn.functional as F
import logging

from utils import utils

class Client():
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        self.cid = cid
        self.model = model
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.device = device
        self.log_interval = len(dataLoader) - 1
        self.init_stateChange()
        self.originalState = deepcopy(model.state_dict())
        self.isTrained = False
        self.inner_epochs = inner_epochs
        self.criterion = criterion
        # for moving average of gradient updates
        self.K_avg = 3 # window size of moving average of grad updates
        self.hog_avg = deque(maxlen=self.K_avg)
        #logging.info(f"Init client {cid} with size window avg grad={self.K_avg}")

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states
        self.avg_delta = deepcopy(states)
        self.sum_hog = deepcopy(states)

    def setModelParameter(self, states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()

    def data_transform(self, data, target):
        return data, target

    def get_data_size(self):
        return len(self.dataLoader)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = self.data_transform(data, target)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.isTrained = True
        self.model.cpu()  ## avoid occupying gpu when idle

    def test(self, testDataLoader):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testDataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testDataLoader.dataset)
        self.model.cpu()  ## avoid occupying gpu when idle
        # Uncomment to print the test scores of each client
        logging.info('client {} ## Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            self.cid, test_loss, correct, len(testDataLoader.dataset),
            100. * correct / len(testDataLoader.dataset)))

    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            self.sum_hog[p] += self.stateChange[p]
            K_ = len(self.hog_avg)
            if K_ == 0:
                self.avg_delta[p] = self.stateChange[p]
            elif K_ < self.K_avg:
                self.avg_delta[p] = (self.avg_delta[p]*K_ + self.stateChange[p])/(K_+1)
            else:
                self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p])/self.K_avg
        self.hog_avg.append(self.stateChange)
        self.isTrained = False

    #         self.test(self.dataLoader)
    def getDelta(self):
        return self.stateChange

    def get_avg_grad(self):
        return torch.cat([v.flatten() for v in self.avg_delta.values()])

    def get_sum_hog(self):
        #return utils.net2vec(self.sum_hog)
        return torch.cat([v.flatten() for v in self.sum_hog.values()])

    def get_L2_sum_hog(self):
        X = self.get_sum_hog()
        #X = torch.cat([v.flatten() for v in self.sum_hog.values()])
        #logging.info("L2_sum={}".format(X))
        return torch.linalg.norm(X)

    def get_L2_avg_grad(self):
        X = torch.cat([v.flatten() for v in self.avg_delta.values()])
        #X = utils.net2vec(self.avg_delta)
        return torch.linalg.norm(X)

    def get_L2_last_grad(self):
        X = torch.cat([v.flatten() for v in self.stateChange.values()])
        #X = utils.net2vec(self.stateChange)
        return torch.linalg.norm(X)
