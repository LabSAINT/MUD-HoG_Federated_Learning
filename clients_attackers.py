from __future__ import print_function

import torch
import torch.nn.functional as F
import logging

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
from clients import *
from utils.blur_images import GaussianSmoothing

class Unreliable_client(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', mean=0, max_std=0.5, fraction_noise=0.5,
            fraction_train=0.3, blur_method='gaussian_smooth', inner_epochs=1,
            channels=1, kernel_size=5):
        logging.info("init UNRELIABLE Client {}".format(cid))
        super(Unreliable_client, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.max_std = max_std
        self.mean = mean
        self.unreliable_fraction = fraction_noise
        self.fraction_train=fraction_train
        self.seed = 0
        self.channels = channels
        self.kernel_size = kernel_size
        self.blur_method = blur_method

    def data_transform(self, data, target):
        if torch.rand(1) < self.unreliable_fraction:
            if self.blur_method == 'add_noise':
                # APPROACH 1: simple add noise
                torch.manual_seed(self.seed)
                std = torch.rand(data.shape)*self.max_std
                gaussian = torch.normal(mean=self.mean, std=std)
                assert data.shape == gaussian.shape, "Inconsistent Gaussian noise shape"
                data_ = data + gaussian
            else: # gaussian_smooth
                # APPROACH 2: Gaussian smoothing
                smoothing = GaussianSmoothing(self.channels, self.kernel_size, self.max_std)
                data_ = F.pad(data, (2,2,2,2), mode='reflect')
                data_ = smoothing(data_)
        else:
            data_ = data
        self.seed += 1

        return data_, target

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                if torch.rand(1) > self.fraction_train: #just train 30% of local dataset
                    continue
                data, target = self.data_transform(data, target)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.isTrained = True
        self.model.cpu()  ## avoid occupying gpu when idle


#class Attacker_label_change_all_to_7(Client):
class Attacker_MultiLabelFlipping(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', inner_epochs=1, source_labels=[1,2,3], target_label=7):
        logging.info(f"init ATTACK MULTI-LABEL-FLIPPING, change label {source_labels} to {target_label} Client {cid}")
        super(Attacker_MultiLabelFlipping, self).__init__(cid, model,
            dataLoader, optimizer, criterion, device, inner_epochs)
        self.source_labels = source_labels
        self.target_label = target_label

    def data_transform(self, data, target):
        #target_ = torch.ones(target.shape, dtype=int)*7 # for all labels -->7
        target_ = torch.tensor(list(map(lambda x: self.target_label if x in self.source_labels else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_

class Attacker_LabelFlipping1to7(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', inner_epochs=1, source_label=1, target_label=7):
        super(Attacker_LabelFlipping1to7, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.source_label = source_label
        self.target_label = target_label
        logging.info(f"init ATTACK LABEL Change from {source_label} to {target_label} Client {cid}")

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: self.target_label if x == self.source_label else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_


class Attacker_LabelFlipping01swap(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        super(Attacker_LabelFlipping01swap, self).__init__(cid, model, dataLoader, optimizer, criterion, device,
                                                           inner_epochs)

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: 1 - x if x in [0, 1] else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_


class Attacker_Backdoor(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', inner_epochs=1):
        super(Attacker_Backdoor, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.utils = Backdoor_Utils()
        logging.info("init BACKDOOR ATTACK Client {}".format(cid))

    def data_transform(self, data, target):
        data, target = self.utils.get_poison_batch(data, target,
            backdoor_fraction=0.5, backdoor_label=self.utils.backdoor_label)
        return data, target

    #def testBackdoor(self):
    #    self.model.to(self.device)
    #    self.model.eval()
    #    test_loss = 0
    #    correct = 0
    #    utils = SemanticBackdoor_Utils()
    #    with torch.no_grad():
    #        for data, target in self.dataLoader:
    #            data, target = self.utils.get_poison_batch(data, target,
    #                backdoor_fraction=1.0,
    #                backdoor_label=self.utils.backdoor_label, evaluation=True)
    #            data, target = data.to(self.device), target.to(self.device)
    #            output = self.model(data)
    #            test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
    #            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #            correct += pred.eq(target.view_as(pred)).sum().item()

    #    test_loss /= len(self.dataLoader.dataset)
    #    accuracy = 100. * correct / len(self.dataLoader.dataset)

    #    self.model.cpu()  ## avoid occupying gpu when idle
    #    logging.info('(Testing at the attacker) Test set (Backdoored):'
    #        ' Average loss: {}, Success rate: {}/{} ({}%)'.format(
    #            test_loss, correct, len(self.dataLoader.dataset), accuracy))

    #def update(self):
    #    super().update()
    #    self.testBackdoor()


class Attacker_SemanticBackdoor(Client):
    '''
    suggested by 'How to backdoor Federated Learning'
    https://arxiv.org/pdf/1807.00459.pdf

    For each batch, 20 out of 64 samples (in the original paper) are replaced with semantic backdoor, this implementation replaces on average a 30% of the batch by the semantic backdoor

    '''

    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.cross_entropy, device='cpu', inner_epochs=1):
        super(Attacker_SemanticBackdoor, self).__init__(cid, model, dataLoader, optimizer, criterion, device,
                                                        inner_epochs)
        self.utils = SemanticBackdoor_Utils()

    def data_transform(self, data, target):
        data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=0.3,
                                                   backdoor_label=self.utils.backdoor_label)
        return data, target

    def testBackdoor(self):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=1.0,
                                                           backdoor_label=self.utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info('(Testing at the attacker) Test set (Semantic Backdoored):'
            ' Average loss: {}, Success rate: {}/{} ({}%)\n'.format(
                test_loss, correct, len(self.dataLoader.dataset), accuracy))

    def update(self):
        super().update()
        self.testBackdoor()


class Attacker_Omniscient(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', scale=1, inner_epochs=1):
        super(Attacker_Omniscient, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        logging.info("init ATTACK OMNISCIENT Client {}".format(cid))
        self.scale = scale

    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        trainable_parameter = utils.getTrainableParameters(self.model)
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            if p not in trainable_parameter:
                continue
            #             if not "FloatTensor" in self.originalState[p].type():
            #                 continue
            self.stateChange[p] *= (-self.scale)
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

class Attacker_AddNoise_Grad(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', mean=0, std=0.1 ,inner_epochs=1):
        super(Attacker_AddNoise_Grad, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.mean = mean
        self.std = std
        logging.info("init ATTACK ADD NOISE TO GRAD Client {}".format(cid))

    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        trainable_parameter = utils.getTrainableParameters(self.model)
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            if p not in trainable_parameter:
                continue
            std = torch.ones(self.stateChange[p].shape)*self.std
            gaussian = torch.normal(mean=self.mean, std=std)
            self.stateChange[p] += gaussian
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
