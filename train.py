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
from dataloader import TripleDataset
from model.SetE import SetE
import time


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

    parser.add_argument('--test_path', default='data/valid_self_original_no_cands.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--log_path', default='log/training_{}.log'.format(datetime.now().strftime('%Y-%m-%d')), type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=1000, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=10000, type=int, required=False, help='训练batch size')
    parser.add_argument('--hidden_size', default=300, type=int, required=False, help='隐藏层大小')
    parser.add_argument('--embedding_size', type=int, default=50, help="embedding的维度")
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='学习率')
    parser.add_argument('--beta', default=0.005, type=float, required=False, help='超参数beta')
    parser.add_argument('--B_t', default=1, type=float, required=False, help='超参数B_t')
    parser.add_argument('--B_r', default=2, type=float, required=False, help='超参数B_r')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=25, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=16, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--model_output_path', default='model_{}/'.format(datetime.now().strftime('%Y-%m-%d-%H')), type=str, required=False,
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
    dataset_triple = TripleDataset(args.entity_path,args.concept_path,args.relation_path,args.instanceOf_path,args.triple_path,triple=True)
    dataset_instanceOf = TripleDataset(args.entity_path, args.concept_path, args.relation_path, args.instanceOf_path,args.triple_path, triple=False)

    model = SetE(concept_num=len(dataset_triple.conceptList),instance_num=len(dataset_triple.entityList),
                 relation_num=len(dataset_triple.relationList),
                 beta = args.beta, learingRate =args.lr, dim = args.embedding_size, B_t = args.B_t, B_r = args.B_r)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.beta)

    with tqdm(range(args.epochs)) as t:
        for epoch in range(args.epochs):

            time1 = time.time()
            dataset_triple.generateNegSamples()
            dataset_instanceOf.generateNegSamples()
            dataloader_triple = DataLoader(dataset_triple, batch_size=args.batch_size,shuffle=True)
            dataloader_instanceOf = DataLoader(dataset_instanceOf, batch_size=args.batch_size, shuffle=True)

            loss = []
            loss_triple_total = 0
            loss_instanceOf_total = 0

            # with tqdm(iterable=dataloader_triple) as t1:
            for posSamples,negSamples in dataloader_triple:
                posSamples.to(args.device)
                negSamples.to(args.device)
                loss_triple = model.forward_tri(posSamples,negSamples)

                loss_triple_total += loss_triple
                optimizer.zero_grad()
                loss_triple.backward()
                # print(loss_triple)
                optimizer.step()

                    # t1.set_postfix(loss="{:.2f}".format(loss_triple.item()))
                    # t1.update()

            # 仅使用instanceOf训练并进行subclassOf公理学习
            # with tqdm(iterable=dataloader_instanceOf) as t2:
            for posSamples,negSamples in dataloader_instanceOf:
                posSamples.to(args.device)
                negSamples.to(args.device)
                loss_instanceOf = model.forward_bin(posSamples,negSamples)
                loss_instanceOf_total += loss_instanceOf
                optimizer.zero_grad()
                loss_instanceOf.backward()

                optimizer.step()

                    # t2.set_postfix(loss="{:.2f}".format(loss_instanceOf.item()))
                    # t2.update()

            loss = loss_triple_total + loss_instanceOf_total
            t.set_postfix(loss="{:.2f}".format(loss.item()))
            t.update()

            model_path = join(args.model_output_path,"model_epoch{}.pt".format(epoch+1))
            if (epoch+1) % 10 == 0:

                # torch.save(model, model_path)
                model.saveVector()


if __name__ == "__main__":
    main()



