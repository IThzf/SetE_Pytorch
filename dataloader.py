

import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

# 生成N个负样本

class TripleDataset(Dataset):
    def __init__(self, entityList_path,conceptList_path,relationList_path,posInstanceOf_path,posTriples_path,train=True,triple=True,rate=1,sp=" "):
        super(Dataset, self).__init__()

        self.entityList = []
        self.conceptList = []
        self.relationList = []
        self.posTriples = []
        self.posTriples_final = []
        self.posTriples_set = set()
        self.negTriples = []
        self.posInstanceOf = []
        self.posInstanceOf_final = []
        self.posInstanceOf_set = set()
        self.negInstanceOf = []
        self.posSubClassOf = []
        self.negSubClassOf = []
        self.sp = sp
        self.loadData(entityList_path,conceptList_path,relationList_path,posInstanceOf_path,posTriples_path)
        self.train_flag = train
        self.triple_flag = triple
        self.rate = rate # 控制正负样本比例，本代码不同样本比例效果基本相同

    def loadData(self,entityList_path,conceptList_path, relationList_path,posInstanceOf_path,posTriples_path):
        _,self.entityList = openDetailsAndId(entityList_path)
        _,self.relationList = openDetailsAndId(relationList_path)
        _,self.conceptList = openDetailsAndId(conceptList_path)
        _,self.posTriples = openTrain(posTriples_path, type=3,sp=self.sp)
        _,self.posInstanceOf = openTrain(posInstanceOf_path, type=2,sp=self.sp)
        # _,self.posSubClassOf = openTrain(posSubClassOf_path, type=2)


        for sample in self.posInstanceOf:# 转换为set，加速生成负样本时的查询速度
            self.posInstanceOf_set.add(tuple(sample))

        for sample in self.posTriples:
            self.posTriples_set.add(tuple(sample))

    def loadDataFromPath(self,pos_path,neg_path,type=2):

        _, posTriples = openTrain(pos_path, type,sp=self.sp)
        _, negTriples = openTrain(neg_path, type,sp=self.sp)
        # 返回tensor

        return torch.from_numpy(np.array(posTriples)),torch.from_numpy(np.array(negTriples))

    def loadDataFromPath_list(self,path,type=2):

        _, triples = openTrain(path, type,sp=self.sp)
        return triples


    def generateNegSamples(self,repProba=0.5):
        repSeesd = 1
        np.random.seed(repSeesd)
        # repProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.posTriples),))
        # heads_tmp = np.random.randint(0,len(self.entityList),size=[len(self.entityList)])
        # tails_tmp = np.random.randint(0, len(self.entityList), size=[len(self.entityList)])
        # r_n = random.random()
        # if r_n > 0.5: # 替换头实体
        #     head_tmp = random.randint(0,len(self.entityList))
        self.negTriples = []
        self.negInstanceOf = []
        self.posTriples_final = []
        self.posInstanceOf_final = []

        if self.triple_flag:
            for sample in self.posTriples:
                for i in range(self.rate):
                    self.posTriples_final.append(sample)
                    negSample = self.generateNegSample(sample,0,repProba)
                    self.negTriples.append(negSample)
            # self.posTriples = self.posTriples_final
        else:
            for sample in self.posInstanceOf:
                for i in range(self.rate):
                    self.posInstanceOf_final.append(sample)
                    negSample = self.generateNegSample(sample,1,repProba)
                    self.negInstanceOf.append(negSample)
            # self.posInstanceOf = self.posInstanceOf_final


    def generateNegSample(self,sample,type=0,repProba=0.5):
        if type == 0:
            head, tail, rel = sample[0], sample[1], sample[2]
            r_n = random.random()  # 决定替换尾实体还是头实体
            if r_n < repProba:
                head = random.randint(0, len(self.entityList)-1)
                while (head, tail, rel) in self.posTriples_set:
                    head = random.randint(0, len(self.entityList)-1)

            else:
                tail = random.randint(0, len(self.entityList)-1)
                while (head, tail, rel) in self.posTriples_set:
                    tail = random.randint(0, len(self.entityList)-1)

            return [head,tail,rel]
        else:
            entity,concept = sample[0], sample[1]
            r_n = random.random()  # 决定替换实体还是概念
            if r_n < repProba:
                entity = random.randint(0, len(self.entityList)-1)
                while (entity, concept) in self.posInstanceOf_set:
                    entity = random.randint(0, len(self.entityList)-1)

            else:
                concept = random.randint(0, len(self.conceptList)-1)
                while (entity, concept) in self.posInstanceOf_set:
                    concept = random.randint(0, len(self.conceptList)-1)

            return [entity, concept]


    def __len__(self):
        if self.triple_flag:
            return len(self.posTriples_final)
        else:
            return len(self.posInstanceOf_final)

    def __getitem__(self, item):
        if self.train_flag:
            if self.triple_flag:
                return np.array(self.posTriples_final[item]),np.array(self.negTriples[item])
            else:
                return np.array(self.posInstanceOf_final[item]), np.array(self.negInstanceOf[item])
        else:
            if self.triple_flag:
                return np.array(self.posTriples_final[item])
            else:
                return np.array(self.posInstanceOf_final[item])


# 读取二元关系组,type表示二元关系类型
def openTrain(dir,type=2, sp=" "):
    # type=2  二元组
    # type=3  三元组
    num = 0
    list = []
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)

            if len(triple) <= 1:
                continue
            if type==2:
                instance,concept = triple[0],triple[1]
                re = [int(instance),int(concept)]
            else:
                instance1, instance2, relation = triple[0], triple[1], triple[2]
                re = [int(instance1), int(instance2), int(relation)]
            list.append(re)
            num += 1
        return num, list

# 读取概念、实体、关系id
def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(int(DetailsAndId[1]))
            idNum += 1
    return idNum, list

# 读取概念、实体、关系id
def openDetailsAndId_dict(dir, sp="\t"):
    idNum = 0
    dict_ = []
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            dict_[int(DetailsAndId[1])] = DetailsAndId[0]

            idNum += 1
    return idNum, dict_





