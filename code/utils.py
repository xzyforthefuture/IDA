'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
import torch.nn.functional as F
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, cross_neg, cross_pos):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg, cross_neg, cross_pos)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class CrossEntropyLoss:
    def __init__(self,
                    recmodel: PairWiseModel,
                    config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        pos_scores = self.model(users, pos)
        neg_scores = self.model(users, neg)

        scores = torch.stack([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(len(users), dtype=torch.long).to(world.device)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(scores, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


class KLDivergenceLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        # 获取模型对正样本和负样本的预测得分
        pos_scores = self.model(users, pos)
        neg_scores = self.model(users, neg)

        # 将得分通过softmax函数转换为概率
        scores = torch.stack([pos_scores, neg_scores], dim=1)
        probs = F.softmax(scores, dim=1)
        target_probs = torch.tensor([[1.0, 0.0] for _ in range(len(users))]).to(probs.device)

        # 计算KL散度损失
        loss = F.kl_div(probs.log(), target_probs, reduction='batchmean')

        # 进行反向传播和优化步骤
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S



def UniformSample_original_python(dataset):
    total_start = time()
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    allPosUsers = dataset.allPosUsers
    i_i_graph = dataset.ItemNetItem
    u_u_graph = dataset.UserNetUser
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    max_len = max(map(len, allPos))
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        neg_candidates = np.random.randint(0, dataset.m_items, size=256 * 2)
        mask = ~np.isin(neg_candidates, posForUser)
        neg_items = neg_candidates[mask][:256]
        if len(neg_items) < 256:
            count = len(neg_items)
            while count < 256:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem not in posForUser:
                    neg_items = np.append(neg_items, negitem)
                    count += 1
        list = [user, positem] + neg_items.tolist()
        co_index = np.random.randint(0, len(posForUser), size=1)
        co_item = posForUser[co_index].tolist()
        list += co_item
        zero_indices = torch.nonzero(torch.all(i_i_graph[posForUser] == 0, dim=0)).squeeze()
        zero_indices = np.array(zero_indices, dtype=np.int32)
        cross_neg_item = np.random.randint(0, len(zero_indices), size=256)
        list += zero_indices[cross_neg_item].tolist()
        padded_posForUser = np.pad(posForUser, (0, max_len - len(posForUser)), mode='constant',
                                   constant_values=dataset.m_items).squeeze().tolist()
        list += padded_posForUser

        S.append(list)
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# def UniformSample_original_python(dataset):
#     total_start = time()
#     user_num = dataset.trainDataSize
#     users = np.random.randint(0, dataset.n_users, user_num)
#     allPos = dataset.allPos
#     S = []
#     sample_time1 = 0.
#     sample_time2 = 0.
#     for i, user in enumerate(users):
#         start = time()
#         posForUser = allPos[user]
#         if len(posForUser) == 0:
#             continue
#         sample_time2 += time() - start
#         posindex = np.random.randint(0, len(posForUser))
#         positem = posForUser[posindex]
#         # 一次性生成足够多的负例
#         neg_candidates = np.random.randint(0, dataset.m_items, size=512)
#         # 过滤掉正例
#         neg_candidates = np.setdiff1d(neg_candidates, posForUser)
#         # 选取 256 个不同的负例
#         if len(neg_candidates) >= 256:
#             neg_items = np.random.choice(neg_candidates, 256, replace=False)
#         else:
#             # 如果候选不足 256 个，继续生成新的负例
#             neg_items = neg_candidates.tolist()
#             count = len(neg_items)
#             while count < 256:
#                 negitem = np.random.randint(0, dataset.m_items)
#                 if negitem not in posForUser and negitem not in neg_items:
#                     neg_items.append(negitem)
#                     count += 1
#         list_data = [user, positem] + neg_items.tolist()
#         S.append(list_data)
#         end = time()
#         sample_time1 += end - start
#     total = time() - total_start
#     return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'ma':
        file = f"ma-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
