

import torch

class TripleClassifier():
    def __init__(self):

        self.num = 1

    def evaluate_subClassOf(self,model,batch_pos,batch_neg):
        instance_index = [i for i in range(model.instance_num)]

        instance = model.instance_embeddings(torch.LongTensor(instance_index))


        def isSubClassOf(concept_i,concept_j):
            # concept_i 与所有的instance相乘，求出其子集
            # print(concept_i.detach().numpy().tolist())
            # print(concept_j.detach().numpy().tolist())
            f_tensor = torch.sum(instance * concept_i, 1)  # 每个sample相加
            f_tensor_ = torch.sum(instance * concept_j, 1)  # 每个sample相加

            sets_index = (f_tensor > model.B_t).nonzero() # concept_i 子集中instance对应的index
            sets_index_ = (f_tensor_ > model.B_t).nonzero()
            sets_index = torch.squeeze(sets_index,1)
            sets_index_ = torch.squeeze(sets_index_,1)
            list1 = sets_index.numpy().tolist()
            list2 = sets_index_.numpy().tolist()

            set_unite = self.uniteSet(list1,list2)
            # instance_sets_ = instance[sets_index]  # concept_i 子集中instance
            instance_sets = model.instance_embeddings(sets_index) # concept_i 子集中instance
            f_tensor_sets = f_tensor[sets_index]
            sets_num = sets_index.size()[0]
            #
            f_tensor2 = torch.sum(instance_sets * concept_j, 1) # 计算concept_j 与 concept_i的子集的乘积
            # tensor_max = torch.max(f_tensor2)
            # print(tensor_max.item())
            ans_index = (f_tensor2 > model.B_t).nonzero()
            ans_num = ans_index.size()[0] # 获取concept_j 自己的个数

            # if ans_num > 0:
            #     print(ans_index)

            if ans_num >= sets_num/10: # 如果子集元素个数相同，即可说明concept_i SubClassOf concept_j
               return  True

            return False

        nums_TP = 1
        nums_FN = 1
        nums_FP = 1
        nums_TN = 1
        pos_flag = 0

        subClassOf_list = []

        for batchs in (batch_pos, batch_neg):
            batch_concept_i = batchs[:, 0]
            batch_concept_j = batchs[:, 1]

            concepts_i = model.concept_embeddings(batch_concept_i)  # 获取instance
            concepts_j = model.concept_embeddings(batch_concept_j)

            for i in range(concepts_i.size()[0]):
                concept_i = concepts_i[i]
                concept_j = concepts_j[i]
                concept_i_index = batch_concept_i[i].item()
                concept_j_index = batch_concept_j[i].item()

                ans = isSubClassOf(concept_i,concept_j)
                if pos_flag == 0:
                    if ans:
                        subClassOf_list.append([concept_i_index, concept_j_index])
                        nums_TP += 1
                    else:
                        nums_FN += 1
                else:
                    if ans:
                        nums_FP += 1
                    else:
                        nums_TN += 1

            pos_flag += 1
            break
        print("TP: ", nums_TP, " FN: ", nums_FN, " TN: ", nums_TN, " FP: ", nums_FP)
        return self.score(nums_TP,nums_TN,nums_FP,nums_FN)

    def evaluate_instanceOf(self,model,batch_pos,batch_neg):

        def isInstanceOf(instance,concept):
            # concept_i 与所有的instance相乘，求出其子集

            f_tensor = torch.sum(instance * concept, 1)  # 每个sample相加
            sets_index = (f_tensor > model.B_t).nonzero() # f_tensor 中instanceOf对应的index
            pos_num = sets_index.size()[0] # instanceOf的数量

            total_num = instance.size()[0] # 测试样本的总数量
            neg_num = total_num - pos_num

            return pos_num,neg_num

        nums_TP = 0
        nums_FN = 0
        nums_FP = 0
        nums_TN = 0
        pos_flag = 0

        for batchs in (batch_pos, batch_neg):
            batch_instance = batchs[:, 0]
            batch_concept = batchs[:, 1]

            instance = model.instance_embeddings(batch_instance)  # 获取instance
            concept = model.concept_embeddings(batch_concept)

            pos_num,neg_num = isInstanceOf(instance,concept)

            if pos_flag == 0:
                nums_TP += pos_num
                nums_FN += neg_num
            else:
                nums_FP += pos_num
                nums_TN += neg_num

            pos_flag += 1
        print("TP: ", nums_TP, " FN: ", nums_FN, " TN: ", nums_TN, " FP: ", nums_FP)
        return self.score(nums_TP,nums_TN,nums_FP,nums_FN)

    def evaluate_triple(self,model,batch_pos,batch_neg):

        def isTriple(heads,tails,rels):
            # concept_i 与所有的instance相乘，求出其子集

            heads_tails = torch.cat((heads, tails), 1)
            f_tensor = torch.sum(heads_tails * rels, 1)  # 每个sample相加
            f_numpy = f_tensor.detach().numpy()
            self.write_vector2file("f_numy.txt",f_numpy)

            sets_index = (f_tensor > model.B_r).nonzero() # f_tensor 中instanceOf对应的index
            pos_num = sets_index.size()[0] # instanceOf的数量

            total_num = heads.size()[0] # 测试样本的总数量
            neg_num = total_num - pos_num

            return pos_num,neg_num

        nums_TP = 1
        nums_FN = 1
        nums_FP = 1
        nums_TN = 1
        pos_flag = 0

        for batchs in (batch_pos, batch_neg):

            batch_head = batchs[:, 0]
            batch_tail = batchs[:, 1]
            batch_rel = batchs[:, 2]

            heads = model.instance_embeddings(batch_head)  # 获取instance
            tails = model.instance_embeddings(batch_tail)
            rels = model.rel_embeddings(batch_rel)

            pos_num,neg_num = isTriple(heads,tails,rels)

            if pos_flag == 0:
                nums_TP += pos_num
                nums_FN += neg_num
            else:
                nums_FP += pos_num
                nums_TN += neg_num

            pos_flag += 1
            # break

        print("TP: ", nums_TP, " FN: ", nums_FN," TN: ",nums_TN," FP: ",nums_FP)
        return self.score(nums_TP,nums_TN,nums_FP,nums_FN)

    def score(self,nums_TP,nums_TN,nums_FP,nums_FN):
        acc = (nums_TP + nums_TN) / (nums_TP + nums_TN + nums_FP + nums_FN)
        p = nums_TP / (nums_TP + nums_FP)
        r = nums_TP / (nums_TP + nums_FN)
        F1 = 2 * r * p / (r + p+1)
        # print()
        print(acc, p, r, F1)
        return acc,p,r,F1

    def write_vector2file(self,dir, embeddings):

        with open(dir, 'w', encoding='utf-8') as file:
            for index in range(len(embeddings)):
                file.write(str(index) + "\t")
                file.write(str(embeddings[index]) + "\n")

    def uniteSet(self,list1,list2):
        # print()
        set1 = set(list1)
        set2 = set(list2)
        set3 = set1&set2
        return list(set3)