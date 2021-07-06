

import random
import torch
import numpy as np
from tqdm import tqdm
import time


# 读取概念、实体、关系id
def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    dict_ = {}
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            dict_[DetailsAndId[0]] = DetailsAndId[1]
            # list.append(int(DetailsAndId[1]))
            idNum += 1
    return idNum, dict_

# 读取二元关系组,type表示二元关系类型
def openTrain(dir,entity_dict,relation_dict,type=2, sp="\t"):
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
                instance1, instance2, relation = entity_dict[instance1], entity_dict[instance2], relation_dict[relation]
                re = [instance1, instance2, relation]
                # re = [int(instance1), int(instance2), int(relation)]
            list.append(re)
            num += 1
        return num, list

def generateNegSample(triples,sample,entity_num,repProba=0.5):
    head, tail, rel = sample[0], sample[1], sample[2]
    r_n = random.random()  # 决定替换尾实体还是头实体
    if r_n < repProba:
        head = random.randint(0, entity_num-1)
        while (head, tail, rel) in triples:
            head = random.randint(0, entity_num-1)

    else:
        tail = random.randint(0, entity_num-1)
        while (head, tail, rel) in triples:
            tail = random.randint(0, entity_num-1)

    return [head,tail,rel]

def generateNegs(triples,num):

    triples_neg = []
    for triple in triples:
        triple_neg = generateNegSample(triples,triple,num)
        triples_neg.append(triple_neg)
    return triples_neg

def write2file(triples,dir):
    with open(dir,mode="w",encoding="utf8") as file:
        for triple in triples:
            file.write(str(triple[0])+ "\t" + str(triple[1]) + "\t" + str(triple[2]))
            file.write("\n")

data_file = "data/FB15k"
entity_path = data_file +  "/entity2id.txt"
relation_path = data_file +  "/relation2id.txt"
triple_path = data_file +  "/test.txt"
triple_path2 = data_file + "/test2id_positive.txt"
triple_path_neg = data_file + "/test2id_negative.txt"

entity_num,entity_dict = openDetailsAndId(entity_path)
_,relation_dict = openDetailsAndId(relation_path)
_,triples = openTrain(triple_path,entity_dict,relation_dict,type=3)
print(triples[0])
write2file(triples,triple_path2)
triples_neg = generateNegs(triples,entity_num)
write2file(triples_neg,triple_path_neg)




