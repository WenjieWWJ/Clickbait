import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel
from torch_geometric.utils import scatter_


class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat#features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        
        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
          
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).cuda()

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)
        if self.num_layer > 2:
            h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)
        return x


class MMGCN(torch.nn.Module):
    def __init__(self, v_feat, a_feat, words_tensor, edge_index, batch_size, num_user, num_item, aggr_mode, concate, num_layer, has_id, user_item_dict, dim_x, alpha):
        super(MMGCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.words_tensor = torch.tensor(words_tensor, dtype=torch.long).cuda()
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        self.a_feat = torch.tensor(a_feat, dtype=torch.float).cuda()
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.user_item_dict = user_item_dict
        self.alpha = alpha
        
        self.word_embedding = nn.Embedding(torch.max(self.words_tensor[1])+1, 128)
        nn.init.xavier_normal_(self.word_embedding.weight) 
        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        # self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()
        self.pre_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()
        self.post_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()

        self.a_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.a_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=128)#256)
        self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=128)#256)
        self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, 128, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=128)

    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes):
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        a_rep = self.a_gcn(self.a_feat, self.id_embedding)
        self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        
        # pre_interaction_score
        pre_representation = t_rep
        self.pre_embedding = pre_representation
        pre_user_tensor = pre_representation[user_nodes]
        pre_pos_item_tensor = pre_representation[pos_item_nodes]
        pre_neg_item_tensor = pre_representation[neg_item_nodes]
        pre_pos_scores = torch.sum(pre_user_tensor*pre_pos_item_tensor, dim=1)
        pre_neg_scores = torch.sum(pre_user_tensor*pre_neg_item_tensor, dim=1)

        # post_interaction_score
        post_representation = (v_rep+a_rep+t_rep)/3
        self.post_embedding = post_representation
        post_user_tensor = post_representation[user_nodes]
        post_pos_item_tensor = post_representation[pos_item_nodes]
        post_neg_item_tensor = post_representation[neg_item_nodes]
        post_pos_scores = torch.sum(post_user_tensor*post_pos_item_tensor, dim=1)
        post_neg_scores = torch.sum(post_user_tensor*post_neg_item_tensor, dim=1)

        # fusion of pre_ and post_ interaction scores
        # # post*sigmoid(pre)
        pos_scores = post_pos_scores*torch.sigmoid(pre_pos_scores)
        neg_scores = post_neg_scores*torch.sigmoid(pre_neg_scores)

        return pos_scores, neg_scores, pre_pos_scores, pre_neg_scores

    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores, pre_pos_scores, pre_neg_scores = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda()) 
        # BPR loss
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        loss_value_pre = -torch.sum(torch.log2(torch.sigmoid(pre_pos_scores - pre_neg_scores)))

        # # CE loss
        # loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores))) - torch.sum(torch.log2(torch.sigmoid(1-neg_scores)))
        # loss_value_pre = -torch.sum(torch.log2(torch.sigmoid(pre_pos_scores))) - torch.sum(torch.log2(torch.sigmoid(1-pre_neg_scores)))
        
        return loss_value + self.alpha * loss_value_pre

    def full_ranking(self, val_data, topk=[10, 20, 50, 100]):
        pre_user_tensor = self.pre_embedding[:self.num_user]
        pre_item_tensor = self.pre_embedding[self.num_user:]
        post_user_tensor = self.post_embedding[:self.num_user]
        post_item_tensor = self.post_embedding[self.num_user:]

        start_index = 0
        end_index = 3000

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_pre_user_tensor = pre_user_tensor[start_index:end_index]
            temp_post_user_tensor = post_user_tensor[start_index:end_index]
            pre_score_matrix = torch.matmul(temp_pre_user_tensor, pre_item_tensor.t())
            post_score_matrix = torch.matmul(temp_post_user_tensor, post_item_tensor.t())
            
            score_matrix = (post_score_matrix - torch.mean(post_score_matrix, 1, True))*torch.sigmoid(pre_score_matrix)

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-self.num_user
                    score_matrix[row][col] = 1e-9

            _, index_of_rank_list = torch.topk(score_matrix, topk[-1])
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+3000 < self.num_user:
                end_index += 3000
            else:
                end_index = self.num_user

        length = len(val_data)        
        precision = np.array([0.0]*len(topk))
        recall = np.array([0.0]*len(topk))
        ndcg = np.array([0.0]*len(topk))

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()
            for ind, k in enumerate(topk):
                items = set(items_list[0:k])
                num_hit = len(pos_items.intersection(items))
                
                precision[ind] += float(num_hit / k)
                recall[ind] += float(num_hit / num_pos)

                ndcg_score = 0.0
                max_ndcg_score = 0.0

                for i in range(min(num_pos, k)):
                    max_ndcg_score += 1 / math.log2(i+2)
                if max_ndcg_score == 0:
                    continue
                    
                for i, temp_item in enumerate(items_list[0:k]):
                    if temp_item in pos_items:
                        ndcg_score += 1 / math.log2(i+2)
                        
                ndcg[ind] += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length

    def accuracy(self, dataset, topk=10, neg_num=1000):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        bar = tqdm(total=len(dataset))

        for data in dataset:
            bar.update(1)
            if len(data) < 1002:
                continue

            sum_item += 1
            user = data[0]
            neg_items = data[1:1001]
            pos_items = data[1001:]

            batch_user_tensor = torch.tensor(user).cuda() 
            batch_pos_tensor = torch.tensor(pos_items).cuda()
            batch_neg_tensor = torch.tensor(neg_items).cuda()

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

            _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
            num_hit = len(index_set.difference(all_set))
            sum_pre += float(num_hit/topk)
            sum_recall += float(num_hit/num_pos)
            ndcg_score = 0.0
            for i in range(num_pos):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
            sum_ndcg += ndcg_score/num_pos
        bar.close()

        return sum_pre/sum_item, sum_recall/sum_item, sum_ndcg/sum_item

