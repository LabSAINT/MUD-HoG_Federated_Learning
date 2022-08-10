from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F
import logging
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict, Counter

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
import time
import json

def find_separate_point(d):
    # d should be flatten and np or list
    d = sorted(d)
    sep_point = 0
    max_gap = 0
    for i in range(len(d)-1):
        if d[i+1] - d[i] > max_gap:
            max_gap = d[i+1] - d[i]
            sep_point = d[i] + max_gap/2
    return sep_point

def DBSCAN_cluster_minority(dict_data):
    ids = np.array(list(dict_data.keys()))
    values = np.array(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1,1)
    cluster_ = DBSCAN(n_jobs=-1).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id

def Kmean_cluster_minority(dict_data):
    ids = np.array(list(dict_data.keys()))
    values = np.array(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1,1)
    cluster_ = KMeans(n_clusters=2, random_state=0).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id

def find_minority_id(clf):
    count_1 = sum(clf.labels_ == 1)
    count_0 = sum(clf.labels_ == 0)
    mal_label = 0 if count_1 > count_0 else 1
    atk_id = np.where(clf.labels_ == mal_label)[0]
    atk_id = set(atk_id.reshape((-1)))
    return atk_id

def find_majority_id(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    major_id = np.where(clf.labels_ == major_label)[0]
    #major_id = set(major_id.reshape(-1))
    return major_id

def find_targeted_attack_complex(dict_lHoGs, cosine_dist=False):
    """Construct a set of suspecious of targeted and unreliable clients
    by using [normalized] long HoGs (dict_lHoGs dictionary).
    We use two ways of clustering to find all possible suspicious clients:
      - 1st cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
      - 2nd cluster: Using KMeans (K=2) based on angles between
      long_HoGs to median (that is calculated based on only
      normal clients output from the 1st cluster KMeans).
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array(list(dict_lHoGs.values()))
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    sus_tAtk_id1 = id_lHoGs[list(offset_tAtk_id1)]
    logging.info(f"sus_tAtk_id1: {sus_tAtk_id1}")

    offset_normal_ids = find_majority_id(cluster_lh1)
    normal_ids = id_lHoGs[list(offset_normal_ids)]
    normal_lHoGs = value_lHoGs[list(offset_normal_ids)]
    median_normal_lHoGs = np.median(normal_lHoGs, axis=0)
    d_med_lHoGs = {}
    for idx in id_lHoGs:
        if cosine_dist:
            # cosine similarity between median and all long HoGs points.
            d_med_lHoGs[idx] = np.dot(dict_lHoGs[idx], median_normal_lHoGs)
        else:
            # Euclidean distance
            d_med_lHoGs[idx] = np.linalg.norm(dict_lHoGs[idx]- median_normal_lHoGs)

    cluster_lh2 = KMeans(n_clusters=2, random_state=0).fit(np.array(list(d_med_lHoGs.values())).reshape(-1,1))
    offset_tAtk_id2 = find_minority_id(cluster_lh2)
    sus_tAtk_id2 = id_lHoGs[list(offset_tAtk_id2)]
    logging.debug(f"d_med_lHoGs={d_med_lHoGs}")
    logging.info(f"sus_tAtk_id2: {sus_tAtk_id2}")
    sus_tAtk_uRel_id = set(list(sus_tAtk_id1)).union(set(list(sus_tAtk_id2)))
    logging.info(f"sus_tAtk_uRel_id: {sus_tAtk_uRel_id}")
    return sus_tAtk_uRel_id


def find_targeted_attack(dict_lHoGs):
    """Construct a set of suspecious of targeted and unreliable clients
    by using long HoGs (dict_lHoGs dictionary).
      - cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array(list(dict_lHoGs.values()))
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    #cluster_lh = DBSCAN(eps=35, min_samples=7, metric='mahalanobis', n_jobs=-1).fit(value_lHoGs)
    #logging.info(f"DBSCAN labels={cluster_lh.labels_}")
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    sus_tAtk_id = id_lHoGs[list(offset_tAtk_id1)]
    logging.info(f"This round TARGETED ATTACK: {sus_tAtk_id}")
    return sus_tAtk_id

class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""
        self.sims = None
        self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        self.unreliable_ids = set()
        self.suspicious_id = set()
        self.log_sims = None
        self.log_norms = None
        # At least tao_0 + delay_decision rounds to get first decision.
        self.tao_0 = 3
        self.delay_decision = 2 # 2 consecutive rounds
        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)
        # DBSCAN hyper-parameters:
        self.dbscan_eps = 0.5
        self.dbscan_min_samples=5

    def set_log_path(self, log_path, exp_name, t_run):
        self.log_path = log_path
        self.log_sim_path = '{}/sims_{}_{}.npy'.format(log_path, exp_name, t_run)
        self.log_norm_path = '{}/norms_{}_{}.npy'.format(log_path, exp_name, t_run)
        self.log_results = f'{log_path}/acc_prec_rec_f1_{exp_name}_{t_run}.txt'
        self.output_file = open(self.log_results, 'w', encoding='utf-8')

    def close(self):
        if self.log_sims is None or self.log_norms is None:
            return
        with open(self.log_sim_path, 'wb') as f:
            np.save(f, self.log_sims, allow_pickle=False)
        with open(self.log_norm_path, 'wb') as f:
            np.save(f, self.log_norms, allow_pickle=False)
        self.output_file.close()

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)
        self.num_clients = len(self.clients)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        logging.info("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        nb_classes = 10 # for MNIST, Fashion-MNIST, CIFAR-10
        cf_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
                for t, p in zip(target.view(-1), pred.view(-1)):
                    cf_matrix[t.long(), p.long()] += 1
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
                test_loss, correct, count, accuracy))
        logging.info(f"[Sever] Confusion matrix:\n {cf_matrix.detach().cpu()}")
        cf_matrix = cf_matrix.detach().cpu().numpy()
        row_sum = np.sum(cf_matrix, axis=0) # predicted counts
        col_sum = np.sum(cf_matrix, axis=1) # targeted counts
        diag = np.diag(cf_matrix)
        precision = diag / row_sum # tp/(tp+fp), p is predicted positive.
        recall = diag / col_sum # tp/(tp+fn)
        f1 = 2*(precision*recall)/(precision+recall)
        m_acc = np.sum(diag)/np.sum(cf_matrix)
        results = {'accuracy':accuracy,'test_loss':test_loss,
                   'precision':precision.tolist(),'recall':recall.tolist(),
                   'f1':f1.tolist(),'confusion':cf_matrix.tolist(),
                   'epoch':self.iter}
        json.dump(results, self.output_file)
        self.output_file.write("\n")
        self.output_file.flush()
        logging.info(f"[Server] Precision={precision},\n Recall={recall},\n F1-score={f1},\n my_accuracy={m_acc*100.}[%]")

        return test_loss, accuracy

    def test_backdoor(self):
        logging.info("[Server] Start testing backdoor\n")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = Backdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info(
            '[Server] Test set (Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.
                format(test_loss, correct, len(self.dataLoader.dataset), accuracy))
        return test_loss, accuracy

    def test_semanticBackdoor(self):
        logging.info("[Server] Start testing semantic backdoor")

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info(
            '[Server] Test set (Semantic Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.
                format(test_loss, correct, len(self.dataLoader.dataset), accuracy))
        return test_loss, accuracy, data, pred

    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()

        if self.isSaveChanges:
            self.saveChanges(selectedClients)

        tic = time.perf_counter()
        Delta = self.AR(selectedClients)
        toc = time.perf_counter()
        logging.info(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")

        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        logging.info(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = False # True
        saveOriginal = True #False
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            logging.info(f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
#             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            logging.info(f'[Server] Update vectors have been saved to {savepath}')

    def set_AR_param(self, dbscan_eps=0.5, min_samples=5):
        logging.info(f"SET DBSCAN eps={dbscan_eps}, min_samples={min_samples}")
        self.dbscan_eps = dbscan_eps
        self.min_samples=min_samples

    ## Aggregation functions ##

    def set_AR(self, ar):
        if ar == 'fedavg':
            self.AR = self.FedAvg
        elif ar == 'median':
            self.AR = self.FedMedian
        elif ar == 'gm':
            self.AR = self.geometricMedian
        elif ar == 'krum':
            self.AR = self.krum
        elif ar == 'mkrum':
            self.AR = self.mkrum
        elif ar == 'foolsgold':
            self.AR = self.foolsGold
        elif ar == 'residualbase':
            self.AR = self.residualBase
        elif ar == 'attention':
            self.AR = self.net_attention
        elif ar == 'mlp':
            self.AR = self.net_mlp
        elif ar == 'mudhog':
            self.AR = self.mud_hog
        elif ar == 'fedavg_oracle':
            self.AR = self.fedavg_oracle
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def fedavg_oracle(self, clients):
        normal_clients = []
        for i in range(self.num_clients):
            if i >= 4:
                normal_clients.append(clients[i])
        out = self.FedFuncWholeNet(normal_clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

    def geometricMedian(self, clients):
        from rules.geometricMedian import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def krum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('krum').cpu()(arr.cpu()))
        return out

    def mkrum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('mkrum').cpu()(arr.cpu()))
        return out

    def foolsGold(self, clients):
        from rules.foolsGold import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def residualBase(self, clients):
        from rules.residualBase import Net
        out = self.FedFuncWholeStateDict(clients, Net().main)
        return out

    def net_attention(self, clients):
        from aaa.attention import Net

        net = Net()
        net.path_to_net = self.path_to_aggNet

        out = self.FedFuncWholeStateDict(clients, lambda arr: net.main(arr, self.model))
        return out

    def net_mlp(self, clients):
        from aaa.mlp import Net

        net = Net()
        net.path_to_net = self.path_to_aggNet

        out = self.FedFuncWholeStateDict(clients, lambda arr: net.main(arr, self.model))
        return out

        ## Helper functions, act as adaptor from aggregation function to the federated learning system##

    def add_mal_id(self, sus_flip_sign, sus_uAtk, sus_tAtk):
        all_suspicious = sus_flip_sign.union(sus_uAtk, sus_tAtk)
        for i in range(self.num_clients):
            if i not in all_suspicious:
                if self.pre_mal_id[i] == 0:
                    if i in self.mal_ids:
                        self.mal_ids.remove(i)
                    if i in self.flip_sign_ids:
                        self.flip_sign_ids.remove(i)
                    if i in self.uAtk_ids:
                        self.uAtk_ids.remove(i)
                    if i in self.tAtk_ids:
                        self.tAtk_ids.remove(i)
                else: #> 0
                    self.pre_mal_id[i] = 0
                    # Unreliable clients:
                    if i in self.uAtk_ids:
                        self.count_unreliable[i] += 1
                        if self.count_unreliable[i] >= self.delay_decision:
                            self.uAtk_ids.remove(i)
                            self.mal_ids.remove(i)
                            self.unreliable_ids.add(i)
            else:
                self.pre_mal_id[i] += 1
                if self.pre_mal_id[i] >= self.delay_decision:
                    if i in sus_flip_sign:
                        self.flip_sign_ids.add(i)
                        self.mal_ids.add(i)
                    if i in sus_uAtk:
                        self.uAtk_ids.add(i)
                        self.mal_ids.add(i)
                if self.pre_mal_id[i] >= 2*self.delay_decision and i in sus_tAtk:
                    self.tAtk_ids.add(i)
                    self.mal_ids.add(i)

        logging.debug("mal_ids={}, pre_mal_id={}".format(self.mal_ids, self.pre_mal_id))
        #logging.debug("Count_unreliable={}".format(self.count_unreliable))
        logging.info("FLIP-SIGN ATTACK={}".format(self.flip_sign_ids))
        logging.info("UNTARGETED ATTACK={}".format(self.uAtk_ids))
        logging.info("TARGETED ATTACK={}".format(self.tAtk_ids))

    def mud_hog(self, clients):
        # long_HoGs for clustering targeted and untargeted attackers
        # and for calculating angle > 90 for flip-sign attack
        long_HoGs = {}

        # normalized_sHoGs for calculating angle > 90 for flip-sign attack
        normalized_sHoGs = {}
        full_norm_short_HoGs = [] # for scan flip-sign each round

        # L2 norm short HoGs are for detecting additive noise,
        # or Gaussian/random noise untargeted attack
        short_HoGs = {}

        # STAGE 1: Collect long and short HoGs.
        for i in range(self.num_clients):
            # longHoGs
            sum_hog_i = clients[i].get_sum_hog().detach().cpu().numpy()
            L2_sum_hog_i = clients[i].get_L2_sum_hog().detach().cpu().numpy()
            long_HoGs[i] = sum_hog_i

            # shortHoGs
            sHoG = clients[i].get_avg_grad().detach().cpu().numpy()
            #logging.debug(f"sHoG={sHoG.shape}") # model's total parameters, cifar=sHoG=(11191262,)
            L2_sHoG = np.linalg.norm(sHoG)
            full_norm_short_HoGs.append(sHoG/L2_sHoG)
            short_HoGs[i] = sHoG

            # Exclude the firmed malicious clients
            if i not in self.mal_ids:
                normalized_sHoGs[i] = sHoG/L2_sHoG

        # STAGE 2: Clustering and find malicious clients
        if self.iter >= self.tao_0:
            # STEP 1: Detect FLIP_SIGN gradient attackers
            """By using angle between normalized short HoGs to the median
            of normalized short HoGs among good candidates.
            NOTE: we tested finding flip-sign attack with longHoG, but it failed after long running.
            """
            flip_sign_id = set()
            """
            median_norm_shortHoG = np.median(np.array([v for v in normalized_sHoGs.values()]), axis=0)
            for i, v in enumerate(full_norm_short_HoGs):
                dot_prod = np.dot(median_norm_shortHoG, v)
                if dot_prod < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")
            """
            non_mal_sHoGs = dict(short_HoGs) # deep copy dict
            for i in self.mal_ids:
                non_mal_sHoGs.pop(i)
            median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                #logging.info(f"median_sHoG={median_sHoG}, v={v}")
                v = np.array(list(v))
                d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
                if d_cos < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")


            # STEP 2: Detect UNTARGETED ATTACK
            """ Exclude sign-flipping first, the remaining nodes include
            {NORMAL, ADDITIVE-NOISE, TARGETED and UNRELIABLE}
            we use DBSCAN to cluster them on raw gradients (raw short HoGs),
            the largest cluster is normal clients cluster (C_norm). For the remaining raw gradients,
            compute their Euclidean distance to the centroid (mean or median) of C_norm.
            Then find the bi-partition of these distances, the group of smaller distances correspond to
            unreliable, the other group correspond to additive-noise (Assumption: Additive-noise is fairly
            large (since it is attack) while unreliable's noise is fairly small).
            """

            # Step 2.1: excluding sign-flipping nodes from raw short HoGs:
            logging.info("===========using shortHoGs for detecting UNTARGETED ATTACK====")
            for i in range(self.num_clients):
                if i in flip_sign_id or i in self.flip_sign_ids:
                    short_HoGs.pop(i)
            id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))
            # Find eps for MNIST and CIFAR:
            """
            dist_1 = {}
            for k,v in short_HoGs.items():
                if k != 1:
                    dist_1[k] = np.linalg.norm(v - short_HoGs[1])
                    logging.info(f"Euclidean distance between 1 and {k} is {dist_1[k]}")

            logging.info(f"Average Euclidean distances between 1 and others {np.mean(list(dist_1.values()))}")
            logging.info(f"Median Euclidean distances between 1 and others {np.median(list(dist_1.values()))}")
            """

            # DBSCAN is mandatory success for this step, KMeans failed.
            # MNIST uses default eps=0.5, min_sample=5
            # CIFAR uses eps=50, min_sample=5 (based on heuristic evaluation Euclidean distance of grad of RestNet18.
            start_t = time.time()
            cluster_sh = DBSCAN(eps=self.dbscan_eps, n_jobs=-1,
                min_samples=self.dbscan_min_samples).fit(value_sHoGs)
            t_dbscan = time.time() - start_t
            #logging.info(f"CLUSTER DBSCAN shortHoGs took {t_dbscan}[s]")
            # TODO: comment out this line
            logging.info("labels cluster_sh= {}".format(cluster_sh.labels_))
            offset_normal_ids = find_majority_id(cluster_sh)
            normal_ids = id_sHoGs[list(offset_normal_ids)]
            normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
            normal_cent = np.median(normal_sHoGs, axis=0)
            logging.debug(f"offset_normal_ids={offset_normal_ids}, normal_ids={normal_ids}")

            # suspicious ids of untargeted attacks and unreliable or targeted attacks.
            offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
            sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]
            logging.info(f"SUSPECTED UNTARGETED {sus_uAtk_ids}")

            # suspicious_ids consists both additive-noise, targeted and unreliable clients:
            suspicious_ids = [i for i in id_sHoGs if i not in normal_ids] # this includes sus_uAtk_ids
            logging.debug(f"suspicious_ids={suspicious_ids}")
            d_normal_sus = {} # distance from centroid of normal to suspicious clients.
            for sid in suspicious_ids:
                d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid]-normal_cent)

            # could not find separate points only based on suspected untargeted attacks.
            #d_sus_uAtk_values = [d_normal_sus[i] for i in sus_uAtk_ids]
            #d_separate = find_separate_point(d_sus_uAtk_values)
            d_separate = find_separate_point(list(d_normal_sus.values()))
            logging.debug(f"d_normal_sus={d_normal_sus}, d_separate={d_separate}")
            sus_tAtk_uRel_id0, uAtk_id = set(), set()
            for k, v in d_normal_sus.items():
                if v > d_separate and k in sus_uAtk_ids:
                    uAtk_id.add(k)
                else:
                    sus_tAtk_uRel_id0.add(k)
            logging.info(f"This round UNTARGETED={uAtk_id}, sus_tAtk_uRel_id0={sus_tAtk_uRel_id0}")


            # STEP 3: Detect TARGETED ATTACK
            """
              - First excluding flip_sign and untargeted attack from.
              - Using KMeans (K=2) based on Euclidean distance of
                long_HoGs==> find minority ids.
            """
            for i in range(self.num_clients):
                if i in self.flip_sign_ids or i in flip_sign_id:
                    if i in long_HoGs:
                        long_HoGs.pop(i)
                if i in uAtk_id or i in self.uAtk_ids:
                    if i in long_HoGs:
                        long_HoGs.pop(i)

            # Using Euclidean distance is as good as cosine distance (which used in MNIST).
            logging.info("=======Using LONG HOGs for detecting TARGETED ATTACK========")
            tAtk_id = find_targeted_attack(long_HoGs)

            # Aggregate, count and record ATTACKERs:
            self.add_mal_id(flip_sign_id, uAtk_id, tAtk_id)
            logging.info("OVERTIME MALICIOUS client ids ={}".format(self.mal_ids))

            # STEP 4: UNRELIABLE CLIENTS
            """using normalized short HoGs (normalized_sHoGs) to detect unreliable clients
            1st: remove all malicious clients (manipulate directly).
            2nd: find angles between normalized_sHoGs to the median point
            which mostly normal point and represent for aggreation (e.g., Median method).
            3rd: find confident mid-point. Unreliable clients have larger angles
            or smaller cosine similarities.
            """
            """
            for i in self.mal_ids:
                if i in normalized_sHoGs:
                    normalized_sHoGs.pop(i)

            angle_normalized_sHoGs = {}
            # update this value again after excluding malicious clients
            median_norm_shortHoG = np.median(np.array(list(normalized_sHoGs.values())), axis=0)
            for i, v in normalized_sHoGs.items():
                angle_normalized_sHoGs[i] = np.dot(median_norm_shortHoG, v)

            angle_sep_nsH = find_separate_point(list(angle_normalized_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_normalized_sHoGs.items():
                if v < angle_sep_nsH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)
            """
            for i in self.mal_ids:
                if i in short_HoGs:
                    short_HoGs.pop(i)

            angle_sHoGs = {}
            # update this value again after excluding malicious clients
            median_sHoG = np.median(np.array(list(short_HoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                angle_sHoGs[i] = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))

            angle_sep_sH = find_separate_point(list(angle_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_sHoGs.items():
                if v < angle_sep_sH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)
            logging.info(f"This round UNRELIABLE={uRel_id}, normal_id={normal_id}")
            #logging.debug(f"anlge_normalized_sHoGs={angle_normalized_sHoGs}, angle_sep_nsH={angle_sep_nsH}")
            logging.debug(f"anlge_sHoGs={angle_sHoGs}, angle_sep_nsH={angle_sep_sH}")

            for k in range(self.num_clients):
                if k in uRel_id:
                    self.count_unreliable[k] += 1
                    if self.count_unreliable[k] > self.delay_decision:
                        self.unreliable_ids.add(k)
                # do this before decreasing count
                if self.count_unreliable[k] == 0 and k in self.unreliable_ids:
                    self.unreliable_ids.remove(k)
                if k not in uRel_id and self.count_unreliable[k] > 0:
                    self.count_unreliable[k] -= 1
            logging.info("UNRELIABLE clients ={}".format(self.unreliable_ids))

            normal_clients = []
            for i, client in enumerate(clients):
                if i not in self.mal_ids and i not in tAtk_id and i not in uAtk_id:
                    normal_clients.append(client)
            self.normal_clients = normal_clients
        else:
            normal_clients = clients
        out = self.FedFuncWholeNet(normal_clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedFuncWholeNet(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # size is relative to number of samples, actually it is number of batches
        sizes = [c.get_data_size() for c in clients]
        total_s = sum(sizes)
        logging.info(f"clients' sizes={sizes}, total={total_s}")
        weights = [s/total_s for s in sizes]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        weighted_vecs = [w*v for w,v in zip(weights, vecs)]
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeStateDict(self, clients, func):
        '''
        The aggregation rule views the update vectors as a set of state dict.
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]

        resultDelta = func(deltas)

        Delta.update(resultDelta)
        return Delta
