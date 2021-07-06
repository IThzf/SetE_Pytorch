import argparse
from datetime import datetime
import argparse
import torch
from tqdm import tqdm
import random
import numpy as np
import logging
from gensim.models import KeyedVectors
import re
from collections import defaultdict
import os
from os.path import join, exists
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import json
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from dataloader import TripleDataset
from model.SetE import SetE
import time

import os
from model.evaluate import TripleClassifier
from model.LinkPrediction import LinkPredictor



def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', default=False, action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--device', default="cpu",  help='不使用GPU进行训练')
    parser.add_argument('--dataset_path', default='data/YAGO39K', type=str, required=False,help='数据集根目录')
    parser.add_argument('--entity_path', default='data/YAGO39K/Train/instance2id.txt', type=str, required=False, help='实体id文件')
    parser.add_argument('--concept_path', default='data/YAGO39K/Train/concept2id.txt', type=str, required=False, help='概念id文件')
    parser.add_argument('--relation_path', default='data/YAGO39K/Train/relation2id.txt', type=str, required=False, help='关系id文件')
    parser.add_argument('--instanceOf_path', default='instanceOf2id.txt', type=str, required=False, help='instanceOf id文件')
    parser.add_argument('--subClassOf_path', default='subClassOf2id.txt', type=str, required=False,help='subClassOf id文件')
    parser.add_argument('--triple_path', default='triple2id.txt', type=str, required=False,help='triple id文件')

    parser.add_argument('--instanceOf_path_neg_test', default='data/YAGO39K/Test/instanceOf2id_negative.txt', type=str, required=False,help='测试集instanceOf负样本')
    parser.add_argument('--instanceOf_path_pos_test', default='data/YAGO39K/Test/instanceOf2id_positive.txt', type=str,
                        required=False, help='测试集instanceOf正样本')
    parser.add_argument('--subClassOf_path_neg_test', default='data/YAGO39K/Test/subClassOf2id_negative.txt', type=str,
                        required=False, help='测试集subClassOf负样本')
    parser.add_argument('--subClassOf_path_pos_test', default='data/YAGO39K/Test/subClassOf2id_positive.txt', type=str,
                        required=False, help='测试集subClassOf正样本')
    parser.add_argument('--triple_path_neg_test', default='data/YAGO39K/Test/triple2id_negative.txt', type=str,
                        required=False, help='测试集triple负样本')
    parser.add_argument('--triple_path_pos_test', default='data/YAGO39K/Test/triple2id_positive.txt', type=str,
                        required=False, help='测试集triple正样本')

    parser.add_argument('--test_path', default='data/valid_self_original_no_cands.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--log_path', default='log/training_{}.log'.format(datetime.now().strftime('%Y-%m-%d')), type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=1000, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=10000, type=int, required=False, help='训练batch size')
    parser.add_argument('--hidden_size', default=300, type=int, required=False, help='隐藏层大小')
    parser.add_argument('--embedding_size', type=int, default=50, help="embedding的维度")
    parser.add_argument('--lr', default=0.1, type=float, required=False, help='学习率')
    parser.add_argument('--beta', default=0.001, type=float, required=False, help='正则化系数beta')
    parser.add_argument('--B_t', default=1, type=float, required=False, help='超参数B_t')
    parser.add_argument('--B_r', default=2, type=float, required=False, help='超参数B_r')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=25, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=16, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--model_output_path', default='model/model_{}/'.format(datetime.now().strftime('%Y-%m-%d-%H')), type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--GEN_writer_dir', default='gen_tensorboard_summary_{}/'.format(datetime.now().strftime('%Y-%m-%d-%H')), type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=42, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--embedding_path', type=str, default='embedding/glove.txt', help="加载glove向量")
    parser.add_argument('--num_words', type=int, default=50000, help="embedding的词个数")
    parser.add_argument('--dropout', type=int, default=0.2, help="dropout的大小")
    parser.add_argument('--word2idx', type=str, default='embedding/word2idx.json', help="word2idx文件保存位置")
    parser.add_argument('--max_length', type=int, default=256, help="生成序列的长度")
    parser.add_argument("--overwrite_cache", default=False, required=False,
                        help="是否重复数据处理")

    return parser.parse_args()


def main():
    args = setup_train_args()
    args.instanceOf_path = args.dataset_path + "/Train/instanceOf2id.txt"
    args.subClassOf_path = args.dataset_path + "/Train/subClassOf2id.txt"
    args.triple_path = args.dataset_path + "/Train/triple2id.txt"
    args.rate = 1

    global logger
    logger = create_logger(args)
    dataset_triple = TripleDataset(args.entity_path,args.concept_path,args.relation_path,args.instanceOf_path,args.triple_path,triple=True,rate=args.rate)
    time1 = time.time()
    instanceOf_list = dataset_triple.loadDataFromPath_list(args.instanceOf_path)
    subClassOf_list = dataset_triple.loadDataFromPath_list(args.subClassOf_path)

    instance_concept_dict = {}
    concept_instance_dict = {}
    for instanceOf in instanceOf_list:
        instance = instanceOf[0]
        concept = instanceOf[1]
        if instance_concept_dict.get(instance) == None:
            instance_concept_dict[instance] = []

        instance_concept_dict[instance].append(concept)

        if concept_instance_dict.get(concept) == None:
            concept_instance_dict[concept] = []
        concept_instance_dict[concept].append(instance)

    concepti_conceptj_dict = {}
    for subClassOf in subClassOf_list:
        concept_i = subClassOf[0]
        concept_j = subClassOf[1]

        if concepti_conceptj_dict.get(concept_i) == None:
            concepti_conceptj_dict[concept_i] = []
        concepti_conceptj_dict[concept_i].append(concept_j)


    _,instance_dict = openDetailsAndId_dict(args.entity_path)
    _,concept_dict = openDetailsAndId_dict(args.concept_path)
    time1 = time.time()

    subClassOf_examples = []
    num = 0
    for concepts1 in subClassOf_list:
        num += 1
        # if num>1000:
        #     break
        concept_i1,concept_j1 = concepts1[0],concepts1[1]

        concept_j2s = concepti_conceptj_dict.get(concept_j1)
        if concept_j2s != None:
            concept_j2 = concept_j2s[0]
            subClassOf_examples.append([concept_i1,concept_j1,concept_j2])

        # for concepts2 in subClassOf_list:
        #     concept_i2, concept_j2 = concepts2[0], concepts2[1]
        #     if concept_j1 == concept_i2:
        #         subClassOf_examples.append([concept_i1,concept_i2,concept_j2])

    print("subClassOf_examples OK: ",time.time()-time1)
    time1 = time.time()
    final_examples = []
    finale_isA_examples = []

    for concepts in subClassOf_examples:
        concepts1 = concepts[0]
        concepts3 = concepts[2]
        instances = concept_instance_dict.get(concepts1)

        # if instances != None:
        #     final_examples.append([instances[0], concepts[0], concepts[1], concepts[2]])

        instances3 = concept_instance_dict.get(concepts3)
        set1 = uniteSet(instances,instances3)
        if len(set1) > 0:
            final_examples.append([set1[0], concepts[0], concepts[1], concepts[2]])

        if len(set1) == 0 and instances!= None:
            finale_isA_examples.append([instances[0], concepts[0], concepts[1], concepts[2]])



    # for instanceOf in instanceOf_list:
    #     instance,concept = instanceOf[0],instanceOf[1]
    #     for concepts in subClassOf_examples:
    #         if concept == concepts[0]:
    #             final_examples.append([instance,concept,concepts[1],concepts[2]])

    print("final_examples OK")
    print(time.time() - time1)
    final_examples_str = []
    final_examples_isA_str = []
    for example in final_examples:
        instance_str = instance_dict[example[0]]
        concept1_str = concept_dict[example[1]]
        concept2_str = concept_dict[example[2]]
        concept3_str = concept_dict[example[3]]
        final_examples_str.append([instance_str,concept1_str,concept2_str,concept3_str])

    for example in finale_isA_examples:
        instance_str = instance_dict[example[0]]
        concept1_str = concept_dict[example[1]]
        concept2_str = concept_dict[example[2]]
        concept3_str = concept_dict[example[3]]
        final_examples_isA_str.append([instance_str,concept1_str,concept2_str,concept3_str])


    # print(subClassOf_examples)
    # print(final_examples)
    # print(final_examples_str)
    print(time.time()-time1)
    write2file("data/view/examples_isA_int.json", finale_isA_examples)
    write2file("data/view/examples_isA.json", final_examples_isA_str)
    write2file("data/view/examples_int.json", final_examples)
    write2file("data/view/examples.json",final_examples_str)

def uniteSet(list1,list2):
    # print()
    if list1 == None:
        return []
    if list2 == None:
        return []
    set1 = set(list1)
    set2 = set(list2)
    set3 = set1&set2
    return list(set3)

def write2file(dir,list_):
    list_json = json.dumps(list_,indent=4)
    with open(dir,"w",encoding="utf8") as file:
        file.write(list_json)
        # json.dump(list_json,file,indent=4)

# 读取概念、实体、关系id
def openDetailsAndId_dict(dir, sp="\t"):
    idNum = 0
    dict_ = {}
    with open(dir, encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            dict_[int(DetailsAndId[1])] = DetailsAndId[0]

            idNum += 1
    return idNum, dict_

def saveModel(path,model,optimizer,epoch):

    if not os.path.exists(path):
        os.makedirs(path)
    model_path = join(path, "model_epoch{}.pt".format(epoch + 1))
    state = {"model":model.state_dict(),"optimizer":optimizer.state_dict()}
    torch.save(state,model_path)

def loadModel(path,model,optimizer):
    state = torch.load(path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

if __name__ == "__main__":
    main()



