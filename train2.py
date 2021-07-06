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
from Triple_Classification import *
import os
from model.evaluate import TripleClassifier



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
    parser.add_argument('--instanceOf_path_neg_val', default='data/YAGO39K/Valid/instanceOf2id_negative.txt', type=str, required=False,help='测试集instanceOf负样本')
    parser.add_argument('--instanceOf_path_pos_val', default='data/YAGO39K/Valid/instanceOf2id_positive.txt', type=str,
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
    parser.add_argument('--batch_size', default=1000, type=int, required=False, help='训练batch size')
    parser.add_argument('--hidden_size', default=50, type=int, required=False, help='隐藏层大小')
    parser.add_argument('--embedding_size', type=int, default=50, help="embedding的维度")
    parser.add_argument('--lr', default=0.1, type=float, required=False, help='学习率')
    parser.add_argument('--beta', default=0.005, type=float, required=False, help='正则化系数beta')
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
    logger.info(
        "B_t:{} B_r:{} learning_rate:{} beta:{} emdedding_dim:{} batch_size:{} ".format(args.B_t, args.B_r, args.lr, args.beta,args.embedding_size,args.batch_size))
    # logger.info("B_t: ",args.B_t," B_r: ",args.B_r," learning_rate: ",args.lr," beta: ",args.beta," emdedding_dim: ",args.embedding_size," batch_size: ",args.batch_size)

    dataset_instanceOf = TripleDataset(args.entity_path, args.concept_path, args.relation_path, args.instanceOf_path, args.triple_path, triple=False,rate=args.rate)

    subClassOf_pos, subClassOf_neg = dataset_instanceOf.loadDataFromPath(args.subClassOf_path_pos_test,
                                                                     args.subClassOf_path_neg_test,type=2)

    instanceOf_pos_val, instanceOf_neg_val = dataset_instanceOf.loadDataFromPath(args.instanceOf_path_pos_val,
                                                                                   args.instanceOf_path_neg_val,
                                                                                   type=2)

    instanceOf_pos_test, instanceOf_neg_test = dataset_instanceOf.loadDataFromPath(args.instanceOf_path_pos_test,
                                                                     args.instanceOf_path_neg_test,type=2)
    triple_pos, triple_neg = dataset_instanceOf.loadDataFromPath(args.triple_path_pos_test,
                                                                     args.triple_path_neg_test,type=3)

    model = SetE(concept_num=len(dataset_instanceOf.conceptList),instance_num=len(dataset_instanceOf.entityList),
                 relation_num=len(dataset_instanceOf.relationList),
                 beta = args.beta, learingRate =args.lr, dim = args.embedding_size, B_t = args.B_t, B_r = args.B_r)

    tripleClassifier = TripleClassifier()
    # test_classfier = init_classifier(args)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.beta)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(args.device)
    # test_classifier = init_classifier(args)
    # with tqdm(range(args.epochs)) as t:
    for epoch in range(args.epochs):
        model.train()
        time1 = time.time()

        dataset_instanceOf.generateNegSamples()
        # print(time.time()-time1)
        # dataloader_triple = DataLoader(dataset_triple, batch_size=args.batch_size,shuffle=True)
        dataloader_instanceOf = DataLoader(dataset_instanceOf, batch_size=args.batch_size, shuffle=True)

        loss = []
        loss_triple_total = 0
        loss_instanceOf_total = 0



        with tqdm(iterable=dataloader_instanceOf) as t2:
            for posSamples,negSamples in dataloader_instanceOf:
                model.normalize()
                posSamples.to(args.device)
                negSamples.to(args.device)
                loss_instanceOf = model.forward_bin(posSamples,negSamples)
                loss_instanceOf_total += loss_instanceOf
                optimizer.zero_grad()
                loss_instanceOf.backward()

                optimizer.step()

                t2.set_postfix(epoch="{}".format(epoch),loss="{:.2f}".format(loss_instanceOf.item()))
                t2.update()

        loss = loss_triple_total + loss_instanceOf_total
        # t.set_postfix(loss="{:.2f}".format(loss.item()))
        # t.update()


        model.eval()

        # test_classfier.instancelist = model.instance_embeddings.weight.data
        # test_classfier.conceptlist = model.concept_embeddings.weight.data
        # test_classfier.relationlist = model.rel_embeddings.weight.data
        # evaluate(test_classfier)

        print("验证集：")
        acc, p, r, f1 = tripleClassifier.evaluate_instanceOf(model, instanceOf_pos_val, instanceOf_neg_val)
        print("测试集： ")
        acc, p, r, f1 = tripleClassifier.evaluate_instanceOf(model, instanceOf_pos_test, instanceOf_neg_test)
        logger.info("triple_classification_instanceOf_epoch:{} acc:{:.4f} p:{:.4f} r:{:.4f} f1:{:.4f}".format(epoch + 1, acc, p, r, f1))
        # tripleClassifier.evaluate_triple(model, triple_pos, triple_neg)
        # logger.info("triple_classification_instanceOf_epoch:{} acc:{:.4f} p:{:.4f} r:{:.4f} f1:{:.4f}".format(epoch+1,acc,p,r,f1))
        # tripleClassifier.evaluate_triple(model, triple_pos, triple_neg)

        # model.saveVector()
        if (epoch+1) % 5== 0:
            model.saveVector()
            saveModel(args.model_output_path,model,optimizer,epoch)



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



