
from random import uniform, sample, random
# from LPAL import LPAL
from copy import deepcopy
import json
import numpy as np
from torch import nn
import torch
import time
from tqdm import tqdm

import torch.nn.functional as F

class SetE(nn.Module):
    # 概念、实体、关系三元组初始化
    def __init__(self,concept_num=1,instance_num=1,relation_num=1,beta = 0.005, learingRate = 0.01, dim = 50, B_t = 1, B_r = 2):
        super(SetE,self).__init__()
        self.learingRate = learingRate
        self.dim = dim

        self.B_t = B_t
        self.B_r = B_r
        self.beta = beta

        self.concept_num = concept_num
        self.instance_num = instance_num
        self.relation_num = relation_num

        self.concept_embeddings = nn.Embedding(concept_num, self.dim)
        self.instance_embeddings = nn.Embedding(instance_num, self.dim) # 小于1
        self.rel_embeddings = nn.Embedding(relation_num, 2*self.dim)

        self.init_weights()

    def init_weights(self):

        concept_init = np.random.uniform(0, 6 / np.sqrt(self.dim), (self.concept_num, self.dim)) * np.random.uniform(0, 1, (self.concept_num, self.dim))
        self.concept_embeddings.weight.data = torch.from_numpy(concept_init)

        instance_init = np.random.uniform(0, 6 / np.sqrt(self.dim), (self.instance_num, self.dim)) * np.random.uniform(0, 1, (self.instance_num, self.dim))
        self.instance_embeddings.weight.data = torch.from_numpy(instance_init)

        rel_init = np.random.uniform(0, 6 / np.sqrt(2*self.dim), (self.relation_num, 2*self.dim)) * np.random.uniform(0, 1, (self.relation_num, 2*self.dim))
        self.rel_embeddings.weight.data = torch.from_numpy(rel_init)



    def forward_bin(self,batch_pos,batch_neg):

        def get_loss_max(b_t, batchs, type):
            batch_instance = batchs[:, 0]
            batch_concept = batchs[:, 1]

            instance = self.instance_embeddings(batch_instance)  # 获取instance
            concept = self.concept_embeddings(batch_concept)
            f_tensor = torch.sum(instance * concept, 1)  # 每个sample相加
            # b_t = 1.1 * b_t
            if type == 0:
                ans_sub = f_tensor - b_t
            else:
                ans_sub = b_t - f_tensor

            return F.relu(ans_sub)



        loss_pos = get_loss_max(self.B_t*1, batch_pos, 1)

        loss_neg = get_loss_max(self.B_t*1, batch_neg, 0)

        losses = loss_pos + loss_neg

        return losses.sum()

    def forward_tri(self,batch_pos,batch_neg):

        def get_loss_max(b_r, batchs, type):
            batch_head = batchs[:, 0]
            batch_tail = batchs[:, 1]
            batch_rel = batchs[:, 2]

            heads = self.instance_embeddings(batch_head)  # 获取instance
            tails = self.instance_embeddings(batch_tail)
            rels = self.rel_embeddings(batch_rel)

            heads_tails = torch.cat((heads, tails), 1)
            f_tensor = torch.sum(heads_tails * rels, 1)  # 每个sample相加

            # b_t = 1.1 * b_t
            if type == 0:
                ans_sub = f_tensor - b_r
            else:
                ans_sub = b_r - f_tensor

            return F.relu(ans_sub)

        loss_pos = get_loss_max(self.B_r*1.1, batch_pos, 1)

        loss_neg = get_loss_max(self.B_r*0.9, batch_neg, 0)

        losses = loss_pos + loss_neg

        return losses.sum()


    def saveVector(self):
        # print("进行第%d次循环" % cycleIndex)
        self.write_vector2file("data\\YAGO39K\\Vector\\relationVector.txt", self.rel_embeddings)
        self.write_vector2file("data\\YAGO39K\\Vector\\conceptVector.txt", self.concept_embeddings)
        self.write_vector2file("data\\YAGO39K\\Vector\\instanceVector.txt", self.instance_embeddings)

    def write_vector2file(self,dir, embeddings):
        embeddings = embeddings.weight.data.numpy().tolist() # embedding装numpy
        with open(dir, 'w', encoding='utf-8') as file:
            for index in range(len(embeddings)):
                file.write(str(index) + "\t")
                file.write(str(embeddings[index]) + "\n")

    def normalize(self):

        self.instance_embeddings.weight.data = self.normalize_radius(self.instance_embeddings.weight.data)

        self.concept_embeddings.weight.data = self.normalize_radius(self.concept_embeddings.weight.data)

        # self.rel_embeddings.weight.data = self.normalize_radius(self.rel_embeddings.weight.data)

        # self.instance_embeddings.weight.data = self.normalize_emb(self.instance_embeddings.weight.data)
        #
        # self.concept_embeddings.weight.data = self.normalize_emb(self.concept_embeddings.weight.data)


        # self.rel_embeddings.weight.data = self.normalize_emb(self.rel_embeddings.weight.data)


        # self.instance_embeddings.weight.data = self.normalize_Bt(self.instance_embeddings.weight.data,self.B_t)
        # self.rel_embeddings.weight.data = self.normalize_Bt(self.rel_embeddings.weight.data, self.B_r)



        # self.instance_embeddings.weight.data = self.normalize_emb(self.instance_embeddings.weight.data)

        # self.instance_embeddings.weight.data = self.normalize_Bt(self.instance_embeddings.weight.data, self.B_t)
        # self.concept_embeddings.weight.data = self.normalize_Bt(self.concept_embeddings.weight.data, self.B_t)
        # self.rel_embeddings.weight.data = self.normalize_emb(self.rel_embeddings.weight.data)
        # self.concept_embeddings.weight.data = self.normalize_emb(self.concept_embeddings.weight.data)


        # self.concept_vec.weight.data[:, -1] = self.normalize_radius(self.concept_vec.weight.data[:, -1])


    def normalize_emb(self,x):
        # return  x/float(length)
        veclen = torch.norm(x, 2, -1, keepdim=True)
        ret = x / veclen
        return ret.detach()

    def normalize_radius(self,x):
        return torch.clamp(x, min=0, max=1.0)

    def normalize_Bt(self,x,B_t):
        # print(x[0])
        num = torch.sum(x,1)
        num_min = torch.min(num)
        # print(num_min)
        ans_index = (num < B_t).nonzero()

        ans_index_ = torch.squeeze(ans_index)

        rate = 0.9
        if ans_index.size()[0] > 0:
            if ans_index.size()[0] > 1:
                ans = torch.unsqueeze(num[ans_index_], 1)# 扩展维度，否则无法进行除法运算
                ans_ = x[ans_index_]
                ans_2 = torch.div(ans_, ans*rate)
                x[ans_index_] = ans_2

            else:
                ans = torch.unsqueeze(num[ans_index_], 0)
                ans_ = x[ans_index_]
                ans_2 = torch.div(ans_, ans*rate)
                x[ans_index_] = ans_2

        return x

    def normalize_Bt2(self,x,B_t):
        # print(x[0])
        num = torch.sum(x,1)
        ans_index = (num < B_t).nonzero()

        ans_index_ = torch.squeeze(ans_index)
        if ans_index.size()[0] > 1:
            ans = torch.unsqueeze(num[ans_index_], 1)# 扩展维度，否则无法进行除法运算

            ans_ = x[ans_index_]
            ans_2 = ans_ + (B_t-num)/50
            x[ans_index] = torch.unsqueeze(ans_2, 1)


        return x
