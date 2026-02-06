"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import math
import pandas as pd
import random
import argparse
import torch
import torch.nn as nn
from train import utils, losses
import os
def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

infonce_criterion = nn.CrossEntropyLoss()
def contrastive_loss(user, pos, neg, temp=0.2):
    user_emb = torch.nn.functional.normalize(user, dim=-1)
    pos_emb = torch.nn.functional.normalize(pos, dim=-1)
    neg_emb = torch.nn.functional.normalize(neg, dim=-1)
    pos_ratings = torch.sum(user_emb * pos_emb, dim=-1)
    neg_ratings = torch.matmul(torch.unsqueeze(user_emb, 1), neg_emb.transpose(1, 2)).squeeze(dim=1)
    numerator = torch.exp(pos_ratings / temp)
    denominator = numerator + torch.sum(torch.exp(neg_ratings / temp), dim=-1)
    ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))
    return ssm_loss


def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)

def item_ssm_loss(pos1, pos2, neg, temp=0.2):
    pos1_emb_norm = torch.nn.functional.normalize(pos1, dim=-1)
    pos2_emb_norm = torch.nn.functional.normalize(pos2, dim=-1)
    neg_emb_norm = torch.nn.functional.normalize(neg, dim=-1)
    pos_ratings = torch.sum(pos1_emb_norm * pos2_emb_norm, dim=-1)
    neg_ratings = torch.matmul(torch.unsqueeze(pos1_emb_norm, 1), neg_emb_norm.transpose(1, 2)).squeeze(dim=1)
    numerator = torch.exp(pos_ratings / temp)
    denominator = numerator + torch.sum(torch.exp(neg_ratings / temp), dim=-1)
    ttc_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))
    return ttc_loss


class HierachicalEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(HierachicalEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_v)

    def selfAttention(self, features):
        q = self.w_q(features)
        k = self.w_k(features)
        v = features
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        y = features.mean(dim=-2)  # [bs, d]

        return y


    def forward(self, features):
        final_feature = self.selfAttention(torch.nn.functional.normalize(features, dim=-1))
        return final_feature


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
class SGL(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(SGL, self).__init__()
        self.graph_1 = None
        self.graph_2 = None
        self.model = LightGCN(config, dataset)

    def prepare_each_epoch(self):
        self.graph_1 = utils.create_adj_mat(self.ui_dataset.trainUser, self.ui_dataset.trainItem,
                                            self.num_users, self.num_items, is_subgraph=True)
        self.graph_2 = utils.create_adj_mat(self.ui_dataset.trainUser, self.ui_dataset.trainItem,
                                            self.num_users, self.num_items, is_subgraph=True)

    def calculate_embedding(self):
        return self.model(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        all_users, all_items = self.calculate_embedding()
        loss[losses.Loss.BPR.value] = losses.loss_BPR(all_users, all_items, users, pos, neg)
        loss[losses.Loss.Regulation.value] = losses.loss_regulation(self.embedding_user, self.embedding_item, users,
                                                                        pos, neg)
        users_1, items_1, users_2, items_2 = self.forward(self.embedding_user.weight, self.embedding_item.weight,
                                                              self.graph_1, self.graph_2)
        loss_ssl_item = losses.loss_info_nce(items_1, items_2, pos)
        loss_ssl_user = losses.loss_info_nce(users_1, users_2, users)
        loss[losses.Loss.SSL.value] = loss_ssl_user + loss_ssl_item
        return loss

    def forward(self, all_users, all_items, graph_1, graph_2):
        users_1, items_1 = self.model(all_users, all_items, graph_1)
        users_2, items_2 = self.model(all_users, all_items, graph_2)
        return users_1, items_1, users_2, items_2


class LightGCN(BasicModel):
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.text_dim = self.config['text_dim']
        self.image_dim = self.config['image_dim']
        self.dataset_name = self.config['dataset_name']
        self.user_embedding_path = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "user_feat_llama.npy"
        self.item_embedding_path = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "item_feat_llama.npy"
        self.item_embedding_path_image = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "item_image_features.npy"
        self.user_embedding_path_image = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "user_image_features.npy"
        self.item_np = np.load(self.item_embedding_path, allow_pickle=True)
        self.user_np = np.load(self.user_embedding_path, allow_pickle=True)
        self.item_np_image = np.load(self.item_embedding_path_image, allow_pickle=True)
        self.user_np_image = np.load(self.user_embedding_path_image, allow_pickle=True)
        self.reduce_dim = nn.Sequential(nn.Linear(self.text_dim, int(0.3 * self.text_dim)),
                                        nn.LeakyReLU(),
                                        nn.Linear(int(0.3 * self.text_dim), self.latent_dim))
        self.reduce_dim_image = nn.Sequential(nn.Linear(self.image_dim, 2 * self.latent_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(2 * self.latent_dim, self.latent_dim))
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.embedding_item_text = torch.tensor(self.item_np).to(world.device)
        self.embedding_user_text = torch.tensor(self.user_np).to(world.device)
        self.embedding_item_image = torch.tensor(self.item_np_image).to(world.device)
        self.embedding_user_image = torch.tensor(self.user_np_image).to(world.device)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.fuser = HierachicalEncoder(self.latent_dim)

    def computer(self):
        users_emb = self.reduce_dim(self.embedding_user_text)
        items_emb = self.reduce_dim(self.embedding_item_text)
        user_id = self.embedding_user.weight
        item_id = self.embedding_item.weight
        users_image_emb = self.reduce_dim_image(self.embedding_user_image)
        items_image_emb = self.reduce_dim_image(self.embedding_item_image)
        users_emb_mm = torch.nn.functional.normalize(users_emb, dim=-1) + torch.nn.functional.normalize(users_image_emb, dim=-1)
        items_emb_mm = torch.nn.functional.normalize(items_emb, dim=-1) + torch.nn.functional.normalize(items_image_emb, dim=-1)

        all_emb_mm = torch.cat([users_emb_mm, items_emb_mm])
        all_id = torch.cat([user_id, item_id])
        embs_mm = [all_emb_mm]
        embs_id = [all_id]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb_mm = torch.sparse.mm(g_droped, all_emb_mm)
            all_id = torch.sparse.mm(g_droped, all_id)
            embs_mm.append(all_emb_mm)
            embs_id.append(all_id)
        embs_mm = torch.stack(embs_mm, dim=1)
        embs_id = torch.stack(embs_id, dim=1)
        light_mm = torch.mean(embs_mm, dim=1)
        light_id = torch.mean(embs_id, dim=1)
        all_users_mm, all_items_mm = torch.split(light_mm, [self.num_users, self.num_items])
        all_users_id, all_items_id = torch.split(light_id, [self.num_users, self.num_items])
        all_users = torch.cat([all_users_mm, all_users_id], dim=1)
        all_items = torch.cat([all_items_mm, all_items_id], dim=1)
        return all_users, all_items, user_id, item_id, all_users_mm, all_users_id, all_items_mm, all_items_id

    def getUsersRating(self, users):
        all_users, all_items, users_emb_id, items_embed_id, all_users_mm, all_users_id, all_items_mm, all_items_id = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating, all_users, all_items

    def getEmbedding(self, users, pos_items, neg_items, cross_neg_items, cross_pos_items):
        mask = cross_pos_items != self.num_items
        all_users, all_items, users_embed, items_embed, all_users_mm, all_users_id, all_items_mm, all_items_id = self.computer()
        valid = cross_pos_items * mask
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = users_embed[users]
        pos_emb_ego = items_embed[pos_items]
        neg_emb_ego = items_embed[neg_items]
        user_cl_mm = all_users_mm[users]
        user_cl_id = all_users_id[users]
        item_cl_mm = all_items_mm[pos_items]
        item_cl_id = all_items_id[pos_items]
        cross_cl_item = all_items[cross_neg_items]
        valid_vectors_sum = (all_items[valid] * mask.unsqueeze(-1)).sum(dim=1)
        valid_count = mask.sum(dim=1, keepdim=True)
        cross_pos_cl_item = valid_vectors_sum / valid_count.float()
        return (users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, user_cl_mm, user_cl_id, item_cl_mm, item_cl_id,
                cross_cl_item, cross_pos_cl_item)

    def bpr_loss(self, users, pos, neg, cross_neg, cross_pos):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0, user_cl_mm,
         user_cl_id, item_cl_mm, item_cl_id, cross_cl_item, cross_pos_cl_item) = self.getEmbedding(users.long(), pos.long(), neg.long(), cross_neg.long(), cross_pos.long())

        # uu对齐
        align_user1 = cl_loss_function(user_cl_id, user_cl_mm, 0.2)
        align_user2 = cl_loss_function(user_cl_mm, user_cl_id, 0.2)
        align_user = (align_user1 + align_user2) / 2

        # ii对齐
        align_item1 = cl_loss_function(item_cl_id, item_cl_mm, 0.2)
        align_item2 = cl_loss_function(item_cl_mm, item_cl_id, 0.2)
        align_item = (align_item1 + align_item2) / 2

        # item-level ssm
        cross_item_loss1 = item_ssm_loss(cross_cl_item[:, 0], cross_pos_cl_item, cross_cl_item[:, 1:], temp=0.15)
        cross_item_loss2 = item_ssm_loss(cross_pos_cl_item, cross_cl_item[:, 0], cross_cl_item[:, 1:], temp=0.15)
        item_level_ssm = (cross_item_loss1 + cross_item_loss2) / 2

        # #cross_user
        # align_mm_id_user = TTC_loss(user_cl_mm, cross_cl_user_id[:, 0], cross_cl_user_id[:, 1:], 0.5)
        # align_mm_id_reverse_user = TTC_loss(cross_cl_user_id[:, 0], user_cl_mm, cross_cl_user_mm[:, 1:], 0.5)
        # align_id_mm_user = TTC_loss(user_cl_id, cross_cl_user_mm[:, 0], cross_cl_user_mm[:, 1:], 0.5)
        # align_id_mm_reverses_user = TTC_loss(cross_cl_user_mm[:, 0], user_cl_id, cross_cl_user_id[:, 1:], 0.5)
        # align_cross_user = (align_mm_id_user + align_mm_id_reverse_user) / 2 + (align_id_mm_user + align_id_mm_reverses_user) / 2
        #
        # # cross_item
        # align_mm_id_item = TTC_loss(item_cl_mm, cross_cl_id[:, 0], cross_cl_id[:, 1:], 0.5)
        # align_mm_id_reverse_item = TTC_loss(cross_cl_id[:, 0], item_cl_mm, cross_cl_mm[:, 1:], 0.5)
        # align_id_mm_item = TTC_loss(item_cl_id, cross_cl_mm[:, 0], cross_cl_mm[:, 1:], 0.5)
        # align_id_mm_reverses_item = TTC_loss(cross_cl_mm[:, 0], item_cl_id, cross_cl_id[:, 1:], 0.5)
        # align_cross_item = (align_mm_id_item + align_mm_id_reverse_item) / 2 + (align_id_mm_item + align_id_mm_reverses_item) / 2

        # ssm loss
        users_emb_norm = torch.nn.functional.normalize(users_emb, dim=-1)
        pos_emb_norm = torch.nn.functional.normalize(pos_emb, dim=-1)
        neg_emb_norm = torch.nn.functional.normalize(neg_emb, dim=-1)
        pos_ratings = torch.sum(users_emb_norm * pos_emb_norm, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb_norm, 1), neg_emb_norm.transpose(1, 2)).squeeze(dim=1)
        numerator = torch.exp(pos_ratings / 0.15)
        denominator = numerator + torch.sum(torch.exp(neg_ratings / 0.15), dim=-1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        # BPR loss
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb[:, 0])
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = ssm_loss + bpr_loss + 0.2 * align_item + 0.2 * align_user + 0.2 * item_level_ssm
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items, users_embed, items_embed, all_users_mm, all_users_id, all_items_mm, all_items_id = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class Modal_Alignment(BasicModel):
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset):
        super(Modal_Alignment, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.text_dim = self.config['text_dim']
        self.image_dim = self.config['image_dim']
        self.dataset_name = self.config['dataset_name']
        self.user_embedding_path = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "user_feat_llama.npy"
        self.item_embedding_path = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "item_feat_llama.npy"
        self.item_embedding_path_image = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "item_image_features.npy"
        self.user_embedding_path_image = '../data/' + f'{self.dataset_name}/' + f"{self.dataset_name}/" + "user_image_features.npy"
        self.item_np = np.load(self.item_embedding_path, allow_pickle=True)
        self.user_np = np.load(self.user_embedding_path, allow_pickle=True)
        self.item_np_image = np.load(self.item_embedding_path_image, allow_pickle=True)
        self.user_np_image = np.load(self.user_embedding_path_image, allow_pickle=True)
        self.reduce_dim = nn.Sequential(nn.Linear(self.text_dim, int(0.3 * self.text_dim)),
                                        nn.LeakyReLU(),
                                        nn.Linear(int(0.3 * self.text_dim), self.latent_dim))
        self.reduce_dim_image = nn.Sequential(nn.Linear(self.image_dim, 2 * self.latent_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(2 * self.latent_dim, self.latent_dim))
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.embedding_item_text = torch.tensor(self.item_np).to(world.device)
        self.embedding_user_text = torch.tensor(self.user_np).to(world.device)
        self.embedding_item_image = torch.tensor(self.item_np_image).to(world.device)
        self.embedding_user_image = torch.tensor(self.user_np_image).to(world.device)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.fuser = HierachicalEncoder(self.latent_dim)

    def computer(self):
        users_emb = self.reduce_dim(self.embedding_user_text)
        items_emb = self.reduce_dim(self.embedding_item_text)
        user_id = self.embedding_user.weight
        item_id = self.embedding_item.weight
        users_image_emb = self.reduce_dim_image(self.embedding_user_image)
        items_image_emb = self.reduce_dim_image(self.embedding_item_image)
        users_emb_mm = torch.nn.functional.normalize(users_emb, dim=-1) + torch.nn.functional.normalize(users_image_emb,
                                                                                                        dim=-1)
        items_emb_mm = torch.nn.functional.normalize(items_emb, dim=-1) + torch.nn.functional.normalize(items_image_emb,
                                                                                                        dim=-1)
        # users_emb_mm = users_image_emb
        # items_emb_mm = items_image_emb


        all_emb_mm = torch.cat([users_emb_mm, items_emb_mm])
        all_id = torch.cat([user_id, item_id])
        embs_mm = [all_emb_mm]
        embs_id = [all_id]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb_mm = torch.sparse.mm(g_droped, all_emb_mm)
            all_id = torch.sparse.mm(g_droped, all_id)
            embs_mm.append(all_emb_mm)
            embs_id.append(all_id)
        embs_mm = torch.stack(embs_mm, dim=1)
        embs_id = torch.stack(embs_id, dim=1)
        light_mm = torch.mean(embs_mm, dim=1)
        light_id = torch.mean(embs_id, dim=1)
        all_users_mm, all_items_mm = torch.split(embs_mm, [self.num_users, self.num_items])
        all_users_id, all_items_id = torch.split(embs_id, [self.num_users, self.num_items])
        users_mm, items_mm = torch.split(light_mm, [self.num_users, self.num_items])
        users_id, items_id = torch.split(light_id, [self.num_users, self.num_items])
        all_users = torch.cat([users_mm, users_id], dim=1)
        all_items = torch.cat([items_mm, items_id], dim=1)
        return all_users, all_items, user_id, item_id, all_users_mm, all_users_id, all_items_mm, all_items_id, users_emb_mm, items_emb_mm

    def getUsersRating(self, users):
        all_users, all_items, users_emb_id, items_embed_id, all_users_mm, all_users_id, all_items_mm, all_items_id, users_emb_mmm, items_embed_mm = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating, all_users, all_items

    def getEmbedding(self, users, pos_items, neg_items, cross_neg_items, cross_pos_items):
        mask = cross_pos_items != self.num_items
        all_users, all_items, users_embed, items_embed, all_users_mm, all_users_id, all_items_mm, all_items_id, users_embed_mm, items_embed_mm = self.computer()
        valid = cross_pos_items * mask
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = users_embed[users]
        pos_emb_ego = items_embed[pos_items]
        neg_emb_ego = items_embed[neg_items]
        users_emb_ego_mm = users_embed_mm[users]
        pos_emb_ego_mm = items_embed_mm[pos_items]
        neg_emb_ego_mm = items_embed_mm[neg_items]
        user_cl_mm = all_users_mm[users]
        user_cl_id = all_users_id[users]
        item_cl_mm = all_items_mm[pos_items]
        item_cl_id = all_items_id[pos_items]
        cross_cl_item = all_items[cross_neg_items]
        valid_vectors_sum = (all_items[valid] * mask.unsqueeze(-1)).sum(dim=1)
        valid_count = mask.sum(dim=1, keepdim=True)
        cross_pos_cl_item = valid_vectors_sum / valid_count.float()
        return (
        users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, user_cl_mm, user_cl_id, item_cl_mm,
        item_cl_id,
        cross_cl_item, cross_pos_cl_item, users_emb_ego_mm, pos_emb_ego_mm, neg_emb_ego_mm)

    def bpr_loss(self, users, pos, neg, cross_neg, cross_pos):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0, user_cl_mm,
         user_cl_id, item_cl_mm, item_cl_id, cross_cl_item, cross_pos_cl_item, userEmb0_mm, posEmb0_mm, negEmb0_mm) = self.getEmbedding(users.long(),
                                                                                                   pos.long(),
                                                                                                   neg.long(),
                                                                                                   cross_neg.long(),
                                                                                                   cross_pos.long())

        # uu对齐
        # ii对齐
        align_user = 0
        align_item = 0
        for i in range(self.n_layers + 1):
            align_user1 = cl_loss_function(user_cl_id[:, i, :], user_cl_mm[:, i, :], 0.2)
            align_user2 = cl_loss_function(user_cl_mm[:, i, :], user_cl_id[:, i, :], 0.2)
            align_user += (align_user1 + align_user2) / 2


            align_item1 = cl_loss_function(item_cl_id[:, i, :], item_cl_mm[:, i, :], 0.2)
            align_item2 = cl_loss_function(item_cl_mm[:, i, :], item_cl_id[:, i, :], 0.2)
            align_item += (align_item1 + align_item2) / 2

        # item-level ssm
        cross_item_loss1 = item_ssm_loss(cross_cl_item[:, 0], cross_pos_cl_item, cross_cl_item[:, 1:], temp=0.15)
        cross_item_loss2 = item_ssm_loss(cross_pos_cl_item, cross_cl_item[:, 0], cross_cl_item[:, 1:], temp=0.15)
        item_level_ssm = (cross_item_loss1 + cross_item_loss2) / 2
        # ssm loss
        users_emb_norm = torch.nn.functional.normalize(users_emb, dim=-1)
        pos_emb_norm = torch.nn.functional.normalize(pos_emb, dim=-1)
        neg_emb_norm = torch.nn.functional.normalize(neg_emb, dim=-1)
        pos_ratings = torch.sum(users_emb_norm * pos_emb_norm, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb_norm, 1), neg_emb_norm.transpose(1, 2)).squeeze(dim=1)
        numerator = torch.exp(pos_ratings / 0.1)
        denominator = numerator + torch.sum(torch.exp(neg_ratings / 0.1), dim=-1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        # BPR loss
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb[:, 0])
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = ssm_loss + bpr_loss + 0.2 * (align_item + align_user) + 0.2 * item_level_ssm
        # loss = ssm_loss + bpr_loss + 0.2 * item_level_ssm
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items, users_embed, items_embed, all_users_mm, all_users_id, all_items_mm, all_items_id = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

