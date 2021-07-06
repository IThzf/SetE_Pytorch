
import torch

class LinkPredictor():
    def __init__(self):

        self.num = 1


    def evaluate_instanceOf_filter(self,model,batch_test,batch_train):

        concepts_index_all = [i for i in range(model.concept_num)]
        concepts_index_all = torch.LongTensor(concepts_index_all)
        concepts = model.concept_embeddings(concepts_index_all)

        def instanceOfRank(instance,instance_index,concept_index,batch_train):
            # concept_i 与所有的instance相乘，求出其子集

            # 训练集中instanceOf所在的索引
            rel_concepts_train_index = (batch_train[:, 0] == instance_index.item()).nonzero() # 训练集中满足instance的concept
            # 获取concept的索引
            rel_concepts_index = batch_train[rel_concepts_train_index][:,0,1]

            f_tensor = torch.sum(instance * concepts, 1)  # 每个sample相加
            sorted,indexs = torch.sort(f_tensor,descending=True)

            indexs = indexs.numpy().tolist()
            rel_concepts_index = rel_concepts_index.numpy().tolist()
            concept_index = concept_index.item()
            rank = self.getRank(indexs,rel_concepts_index,concept_index)


            return rank


        batch_instance = batch_test[:, 0]
        batch_concept = batch_test[:, 1]

        instances = model.instance_embeddings(batch_instance)  # 获取instance
        # concepts = model.concept_embeddings(batch_concept)
        concepts_index = batch_concept
        ranks = []
        num = 0
        for i in range(instances.size()[0]):
            num += 1
            if num > 1000:
                break
            instance = instances[i]
            instance_index = batch_instance[i]
            concept_index = concepts_index[i]
            rank = instanceOfRank(instance,instance_index,concept_index,batch_train)
            ranks.append(rank)

        mrr = self.MRR(ranks)
        hit1 = self.hitN(ranks,1)
        hit3 = self.hitN(ranks,3)
        hit10 = self.hitN(ranks,10)
        print(mrr,hit1,hit3,hit10)
        print(ranks)
        return ranks

    def getRank(self,rank_list,rel_indexs,index):
        rel_indexs = set(rel_indexs)
        rank = 1
        for i in rank_list:
            if i==index:
                break
            if i in rel_indexs:
                continue
            rank += 1


        return rank

    def evaluate_instanceOf_raw(self,model,batch_test):

        concepts_index_all = [i for i in range(model.concept_num)]
        concepts_index_all = torch.LongTensor(concepts_index_all)
        concepts = model.concept_embeddings(concepts_index_all)

        def instanceOfRank(instance,instance_index,concept_index):

            f_tensor = torch.sum(instance * concepts, 1)  # 每个sample相加
            sorted,indexs = torch.sort(f_tensor,descending=True)

            sets_index = (indexs == concept_index).nonzero()


            return sets_index[0][0].item()+1


        batch_instance = batch_test[:, 0]
        batch_concept = batch_test[:, 1]

        instances = model.instance_embeddings(batch_instance)  # 获取instance
        # concepts = model.concept_embeddings(batch_concept)
        concepts_index = batch_concept
        ranks = []
        num = 0
        for i in range(instances.size()[0]):
            num += 1
            if num > 1000:
                break
            instance = instances[i]
            instance_index = batch_instance[i]
            concept_index = concepts_index[i]
            rank = instanceOfRank(instance,instance_index,concept_index)
            ranks.append(rank)

        mrr = self.MRR(ranks)
        hit1 = self.hitN(ranks,1)
        hit3 = self.hitN(ranks,3)
        hit10 = self.hitN(ranks,10)
        print(mrr,hit1,hit3,hit10)
        print(ranks)
        return ranks

    def evaluate_triple_raw(self,model,batch_test):

        tails_index_all = [i for i in range(model.instance_num)]
        tails_index_all = torch.LongTensor(tails_index_all)
        tails_all = model.instance_embeddings(tails_index_all)

        def tripleRank(head,tail,rel,tail_index):

            # head_tail = torch.cat((head, tail), 0)
            f_tensor_head = torch.sum(head * rel[:50], 0)  # 每个sample相加


            f_tensor_tails = torch.sum(rel[50:] * tails_all,1)
            f_tensor = f_tensor_head.item() + f_tensor_tails

            sorted,indexs = torch.sort(f_tensor,descending=True)
            # print(sorted.detach().numpy().tolist()[:10000])
            sets_index = (indexs == tail_index.item()).nonzero()


            return sets_index[0][0].item()+1


        batch_head = batch_test[:, 0]
        batch_tail = batch_test[:, 1]
        batch_rel = batch_test[:, 2]

        heads = model.instance_embeddings(batch_head)  # 获取instance
        tails = model.instance_embeddings(batch_tail)
        rels = model.rel_embeddings(batch_rel)

        rels_index = batch_rel
        ranks = []
        num = 0
        for i in range(heads.size()[0]):
            num += 1
            if num>1000:
                break
            head = heads[i]
            tail = tails[i]
            rel = rels[i]
            tail_index = batch_tail[i]

            rank = tripleRank(head,tail,rel,tail_index)
            ranks.append(rank)

        mrr = self.MRR(ranks)
        hit1 = self.hitN(ranks,1)
        hit3 = self.hitN(ranks,3)
        hit10 = self.hitN(ranks,10)
        print(mrr,hit1,hit3,hit10)
        print(ranks)
        return ranks


    def evaluate_triple_filter(self,model,batch_test,batch_train):

        tails_index_all = [i for i in range(model.instance_num)]
        tails_index_all = torch.LongTensor(tails_index_all)
        tails_all = model.instance_embeddings(tails_index_all)

        def tripleRank(head,tail,rel,head_index,tail_index,rel_index,batch_train):
            # concept_i 与所有的instance相乘，求出其子集

            # 训练集中instanceOf所在的索引
            head_train_index = (batch_train[:, 0] == head_index.item()).nonzero() # 训练集中满足tirple的rel

            # tmp = batch_train[head_train_index]
            # tmp_tail  = tmp[0][:,0,1]
            # 获取tails的索引
            rel_tails_index = batch_train[head_train_index][:,0,1]

            f_tensor_head = torch.sum(head * rel[:50], 0)  # 每个sample相加

            f_tensor_tails = torch.sum(rel[50:] * tails_all, 1)
            f_tensor = f_tensor_head.item() + f_tensor_tails

            sorted,indexs = torch.sort(f_tensor,descending=True)

            indexs = indexs.numpy().tolist()
            rel_tails_index = rel_tails_index.numpy().tolist()
            tail_index = tail_index.item()
            rank = self.getRank(indexs,rel_tails_index,tail_index)


            return rank

        batch_head = batch_test[:, 0]
        batch_tail = batch_test[:, 1]
        batch_rel = batch_test[:, 2]

        heads = model.instance_embeddings(batch_head)  # 获取instance
        tails = model.instance_embeddings(batch_tail)
        rels = model.rel_embeddings(batch_rel)

        ranks = []
        num = 0
        for i in range(heads.size()[0]):
            num += 1
            if num > 1000:
                break
            head = heads[i]
            tail = tails[i]
            rel = rels[i]
            head_index = batch_head[i]
            tail_index = batch_tail[i]
            rel_index = batch_rel[i]
            rank = tripleRank(head,tail,rel,head_index,tail_index,rel_index,batch_train)
            ranks.append(rank)

        mrr = self.MRR(ranks)
        hit1 = self.hitN(ranks,1)
        hit3 = self.hitN(ranks,3)
        hit10 = self.hitN(ranks,10)
        print(mrr,hit1,hit3,hit10)
        print(ranks)
        return ranks

    def evaluate_triple_filter2(self,model,batch_test,batch_train):

        rels_index_all = [i for i in range(model.relation_num)]
        rels_index_all = torch.LongTensor(rels_index_all)
        rels = model.rel_embeddings(rels_index_all)

        def tripleRank(head,tail,rel_index):

            head_tail = torch.cat((head, tail), 0)
            f_tensor = torch.sum(head_tail * rels, 1)  # 每个sample相加

            sorted,indexs = torch.sort(f_tensor,descending=True)

            sets_index = (indexs == rel_index).nonzero()


            return sets_index[0][0].item()+1


        batch_head = batch_test[:, 0]
        batch_tail = batch_test[:, 1]
        batch_rel = batch_test[:, 2]

        heads = model.instance_embeddings(batch_head)  # 获取instance
        tails = model.instance_embeddings(batch_tail)
        rels = model.rel_embeddings(batch_rel)

        rels_index = batch_rel
        ranks = []
        num = 0
        for i in range(heads.size()[0]):
            num += 1
            if num>1000:
                break
            head = heads[i]
            tail = tails[i]

            rel_index = batch_rel[i]
            rank = tripleRank(head,tail,rel_index,batch_train)
            ranks.append(rank)

        mrr = self.MRR(ranks)
        hit1 = self.hitN(ranks,1)
        hit3 = self.hitN(ranks,3)
        hit10 = self.hitN(ranks,10)
        print(mrr,hit1,hit3,hit10)
        print(ranks)
        return ranks

    def getRelConcepts(self,model,batch_test,batch_train):
        concepts_index_all = [i for i in range(model.concept_num)]

    def getIntersection(selt,list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        set_ = set1 - set2
        return list(set_)

    def hitN(self,ranks,n):

        num = 0
        for rank in ranks:
            if rank <= n:
                num += 1
        return num/len(ranks)

    def MRR(self,ranks):
        num = 0
        for rank in ranks:
            num += 1/rank
        return num/len(ranks)

