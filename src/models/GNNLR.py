# coding=utf-8

import torch
import torch.nn.functional as F
import logging
import pandas as pd
import torch.nn as nn
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
from models.NLR import NLR
from utils import utils
from utils.global_p import *
from torch_geometric.nn import Sequential, GCNConv, ChebConv
from torch_geometric.data import Data
import numpy as np

class GNNLR(NLR):
    include_id = True
    include_user_features = False
    include_item_features = False
    include_context_features = False
    data_loader = 'DataLoader'
    data_processor = 'ProLogicRecDP'

    @staticmethod
    def parse_model_args(parser, model_name='NLRRec'):
        parser.add_argument('--variable_num', type=int, default=-1,
                            help='Placeholder of variable_num')
        parser.add_argument('--seq_rec', type=int, default=1,
                            help='Whether keep the order of sequence.')
        parser.add_argument('--or_and', type=int, default=1,
                            help='Whether or-and or and-or.')
        return NLR.parse_model_args(parser, model_name)

    def __init__(self, or_and, seq_rec, item_num, dataset, variable_num=-1, *args, **kwargs):
        self.or_and = or_and
        self.seq_rec = seq_rec
        self.dataset = dataset
        NLR.__init__(self, variable_num=item_num, *args, **kwargs)



    def _init_weights(self):
        #self.load_graph = 1
        self.use_uid = 0 #是否在计算中利用用户信息
        self.total_num = self.user_num + self.variable_num
        self.v_vector_size = 64
        if self.use_uid:
            self.feature_embeddings = torch.nn.Embedding(self.total_num, self.v_vector_size)
            self.entitys = torch.arange(0, self.total_num, dtype=int)
        else:
            self.feature_embeddings = torch.nn.Embedding(self.variable_num, self.v_vector_size)
            self.entitys = torch.arange(0, self.variable_num, dtype=int)
        self.l2_embeddings = ['feature_embeddings']
        self.graph = Data()
        self.load_path = '../dataset/' + self.dataset + '/' + self.dataset + '.train.csv'
        #self.load_uid_iid()
        self.load_iid_iid()

        self.gcn = Sequential('x, edge_index, edge_weight', [
            (GCNConv(self.v_vector_size, self.v_vector_size), 'x, edge_index, edge_weight -> x'),
            nn.BatchNorm1d(self.v_vector_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            (GCNConv(self.v_vector_size, self.v_vector_size), 'x, edge_index, edge_weight -> x'),

        ])

        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, self.v_vector_size]).astype(np.float32)), requires_grad=False)

        self.not_layer = nn.Sequential(
            nn.Linear(self.v_vector_size, self.v_vector_size),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(self.v_vector_size, self.v_vector_size)
        )

        self.and_layer = nn.Sequential(
            nn.Linear(self.v_vector_size * 2, self.v_vector_size * 2),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(self.v_vector_size * 2, self.v_vector_size)
        )

        self.or_layer = nn.Sequential(
            nn.Linear(self.v_vector_size * 2, self.v_vector_size * 2),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(self.v_vector_size * 2, self.v_vector_size)
        )


        if True:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def uniform_size(self, vector1, vector2, train):
        if len(vector1.size()) < len(vector2.size()):
            vector1 = vector1.expand_as(vector2)
        else:
            vector2 = vector2.expand_as(vector1)
        if train:
            r12 = torch.Tensor(vector1.size()[:-1]).uniform_(0, 1).bernoulli()
            r12 = utils.tensor_to_gpu(r12).unsqueeze(-1)
            new_v1 = r12 * vector1 + (-r12 + 1) * vector2
            new_v2 = r12 * vector2 + (-r12 + 1) * vector1
            return new_v1, new_v2
        return vector1, vector2

    def logic_not(self, vector):
        vector = self.not_layer(vector)
        return vector

    def logic_and(self, vector1, vector2, train):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        vector = self.and_layer(vector)
        return vector

    def logic_or(self, vector1, vector2, train):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        vector = self.or_layer(vector)
        return vector

    def load_iid_iid(self):
        df = pd.read_csv(self.load_path, sep='\t')
        df.drop('time', axis=1, inplace=True)
        if self.use_uid == 1:
            df['iid'] += self.user_num
        uid_dict = {}
        i = 0
        for uid in df['uid']:
            if df.at[i, 'label'] == 1:
                if uid not in uid_dict:
                    uid_dict[uid] = []
                    uid_dict[uid].append(df.at[i, 'iid'])
                elif df.at[i, 'iid'] not in uid_dict[uid]:
                    uid_dict[uid].append(df.at[i, 'iid'])
            i = i + 1
        edge_list = []
        edge_dict = {}
        edge_weight = []
        for key in uid_dict:
            for i in range(len(uid_dict[key]) - 1):
                if (uid_dict[key][i], uid_dict[key][i + 1]) not in edge_list:
                    edge_list.append((uid_dict[key][i], uid_dict[key][i + 1]))
                    edge_list.append((uid_dict[key][i + 1], uid_dict[key][i]))
                    edge_dict[(uid_dict[key][i], uid_dict[key][i + 1])] = 1
                    edge_dict[(uid_dict[key][i + 1], uid_dict[key][i])] = 1
                else:
                    edge_dict[(uid_dict[key][i], uid_dict[key][i + 1])] += 1
                    edge_dict[(uid_dict[key][i + 1], uid_dict[key][i])] += 1
        for edge in edge_list:
            edge_weight.append(edge_dict[edge])

        edge_index = torch.tensor(edge_list, dtype=torch.int64)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        self.graph.edge_index = edge_index.t().cuda()
        self.graph.edge_weight = edge_weight.cuda()
        print('init iid_iid Graph: num of edge:' + str(self.graph.edge_index.shape[1]))

    def load_uid_iid(self):
        df = pd.read_csv(self.load_path, sep='\t')
        df.drop('time', axis=1, inplace=True)
        df['iid'] += self.user_num
        edge_list = []
        edge_weight = []
        edge_dict = {}
        i = 0
        for uid in df['uid']:
            if df.at[i, 'label'] == 1:
                if (uid, df.at[i, 'iid']) not in edge_list:
                    edge_list.append((uid, df.at[i, 'iid']))
                    edge_list.append((df.at[i, 'iid'], uid))
                    edge_dict[(uid, df.at[i, 'iid'])] = 1
                    edge_dict[(df.at[i, 'iid'], uid)] = 1
                else:
                    edge_dict[(uid, df.at[i, 'iid'])] += 1
                    edge_dict[(df.at[i, 'iid'], uid)] += 1
            i = i + 1

        for edge in edge_list:
            edge_weight.append(edge_dict[edge])

        edge_index = torch.tensor(edge_list, dtype=torch.int64)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        self.graph.edge_index = edge_index.t().cuda()
        self.graph.edge_weight = edge_weight.cuda()
        print('init uid_iid Graph: num of edge:' + str(self.graph.edge_index.shape[1]))


    def predict_and_or(self, feed_dict):

        return feed_dict

    def predict_or_and(self, feed_dict):
        check_list, embedding_l2 = [], []
        train = feed_dict[TRAIN]
        seq_rec = self.seq_rec == 0
        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]  # = B
        real_batch_size = feed_dict[REAL_BATCH_SIZE]  # = rB

        history = feed_dict[C_HISTORY]  # B * H
        #如果使用用户信息，则处理one-hot
        if self.use_uid == 1:
            is_history_zero = torch.where(history == 0, 0, 1)
            is_history_neg = torch.where(history < 0, -1, 1)
            history = history.abs() + self.user_num
            history = history * is_history_zero
            history = history * is_history_neg
        #
        history_length = feed_dict[C_HISTORY_LENGTH]  # B

        his_pos_neg = history.ge(0).float().unsqueeze(-1)  # B * H * 1
        his_valid = history.abs().gt(0).float()  # B * H
        #第一次训练集获得图
        # if self.load_graph == 1:
        #     self.edge_index = feed_dict[X]
        #     self.graph.edge_index = self.edge_index.t()
        #     print('load graph init')
        #     print('num of edge_index:' + str(self.graph.edge_index.shape[1]))
        #     self.load_graph = 0
        #利用图神经网络与实体特征与邻接矩阵获得聚合特征
        graph_x = self.feature_embeddings(self.entitys.cuda())
        self.graph.x = graph_x
        entity_features = self.gcn(self.graph.x, self.graph.edge_index, self.graph.edge_weight)
        #
        elements = history
        elements = elements.type(torch.float32).unsqueeze(2).expand(history.shape[0], history.shape[1], self.v_vector_size).clone()
        for i in range(history.shape[0]):
            for j in range(history.shape[1]):
                elements[i][j] = entity_features[history[i][j].abs()]
        #
        #elements = self.feature_embeddings(history.abs())  # B * H * V
        # if train:
        #     print('train')
        embedding_l2.append(elements)
        constraint = [elements.view([total_batch_size, -1, self.v_vector_size])]  # B * H * V
        constraint_valid = [his_valid.view([total_batch_size, -1])]  # B * H

        not_elements = self.logic_not(elements)  # B * H * V
        constraint.append(not_elements.view([total_batch_size, -1, self.v_vector_size]))  # B * H * V
        constraint_valid.append(his_valid * (-his_pos_neg.view([total_batch_size, -1]) + 1))  # B * H

        elements = his_pos_neg * elements + (-his_pos_neg + 1) * not_elements  # B * H * V
        elements = elements * his_valid.unsqueeze(-1)  # B * H * V

        # # 随机打乱顺序计算
        # # Randomly shuffle the ordering for computing
        if self.seq_rec == 0:
            all_os, all_ovs = [], []
            for i in range(max(history_length)):
                all_os.append(elements[:, i, :])  # B * V
                all_ovs.append(his_valid[:, i].unsqueeze(-1))  # B * 1
            while len(all_os) > 1:
                idx_a, idx_b = 0, 1
                if train:
                    idx_a, idx_b = np.random.choice(len(all_os), size=2, replace=False)
                if idx_a > idx_b:
                    a, av = all_os.pop(idx_a), all_ovs.pop(idx_a)  # B * V,  B * 1
                    b, bv = all_os.pop(idx_b), all_ovs.pop(idx_b)  # B * V,  B * 1
                else:
                    b, bv = all_os.pop(idx_b), all_ovs.pop(idx_b)  # B * V,  B * 1
                    a, av = all_os.pop(idx_a), all_ovs.pop(idx_a)  # B * V,  B * 1
                a_or_b = self.logic_or(a, b, train=train & ~seq_rec)  # B * V
                abv = av * bv  # B * 1
                ab = abv * a_or_b + av * (-bv + 1) * a + (-av + 1) * bv * b  # B * V
                all_os.insert(0, ab)
                all_ovs.insert(0, (av + bv).gt(0).float())
                constraint.append(ab.view([total_batch_size, 1, self.v_vector_size]))
                constraint_valid.append(abv)
            or_vector = all_os[0]
            left_valid = all_ovs[0]
        else:
            # # 按顺序计算
            # # Compute accordingly to the ordering
            tmp_o = None
            for i in range(max(history_length)):
                tmp_o_valid = his_valid[:, i].unsqueeze(-1)  # B * 1
                if tmp_o is None:
                    tmp_o = elements[:, i, :] * tmp_o_valid  # B * V
                else:
                    tmp_o = self.logic_or(tmp_o, elements[:, i, :], train=train & ~seq_rec) * tmp_o_valid + \
                            tmp_o * (-tmp_o_valid + 1)  # B * V
                    constraint.append(tmp_o.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
                    constraint_valid.append(tmp_o_valid)  # B * 1
            or_vector = tmp_o  # B * V
            left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1

        all_valid = feed_dict[IID].gt(0).float().view([total_batch_size, 1])  # B * 1
        #
        # if train:
        #     print('train')
        right_vector = feed_dict[IID]
        right_vector = right_vector.type(torch.float32).unsqueeze(1).expand(right_vector.shape[0],  self.v_vector_size).clone()
        for i in range(feed_dict[IID].shape[0]):
            if self.use_uid:
                right_vector[i] = entity_features[(feed_dict[IID][i] + self.user_num)]
            else:
                right_vector[i] = entity_features[(feed_dict[IID][i])]

        #
        #right_vector = self.feature_embeddings(feed_dict[IID])  # B * V
        embedding_l2.append(right_vector)
        constraint.append(right_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(all_valid)  # B * 1

        sent_vector = self.logic_and(or_vector, right_vector, train=train & ~seq_rec) * left_valid \
                      + (-left_valid + 1) * right_vector  # B * V
        constraint.append(sent_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1

        if feed_dict[RANK] == 1:
            prediction = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])
        else:
            prediction = self.similarity(sent_vector, self.true, sigmoid=True) * \
                         (self.label_max - self.label_min) + self.label_min

        check_list.append(('prediction', prediction))
        check_list.append(('label', feed_dict[Y]))
        check_list.append(('true', self.true))

        constraint = torch.cat(tuple(constraint), dim=1)
        constraint_valid = torch.cat(tuple(constraint_valid), dim=1)
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list,
                    'constraint': constraint,
                    'constraint_valid': constraint_valid,
                    EMBEDDING_L2: embedding_l2}

        every_his = self.logic_and(elements, right_vector.view([total_batch_size, 1, self.v_vector_size]), train=False)
        every_his = self.similarity(every_his, self.true, sigmoid=True)
        target_sim = self.similarity(right_vector, self.true, sigmoid=True)
        every_his = torch.cat([every_his.view([total_batch_size, -1]),
                               target_sim.view([total_batch_size, -1])], dim=1)
        out_dict['sth'] = every_his
        return out_dict

    def predict(self, feed_dict):
        if self.or_and == 1:
            return self.predict_or_and(feed_dict)
        return self.predict_and_or(feed_dict)

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 模型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失

        Except for making predictions, also compute the loss
        :param feed_dict: model input, it's a dict
        :return: output, it's a dict, prediction is the predicted value, check means needs to check the intermediate result, loss is the loss
        """
        out_dict = self.predict(feed_dict)
        out_dict = self.logic_regularizer(out_dict, train=feed_dict[TRAIN])
        prediction, label = out_dict[PREDICTION], feed_dict[Y]
        r_loss = out_dict['r_loss']
        check_list = out_dict[CHECK]

        # loss
        if feed_dict[RANK] == 1:
            # 计算topn推荐的loss，batch前一半是正例，后一半是负例
            # Compute the loss of topn recommendation, the first half of the batch are the positive examples, the second half are negative examples
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            # 计算rating/clicking预测的loss，默认使用mse
            # Compute the loss of rating/clicking prediction, by default using mse
            if self.loss_sum == 1:
                loss = torch.nn.MSELoss(reduction='sum')(out_dict[PREDICTION], feed_dict[Y])
            else:
                loss = torch.nn.MSELoss(reduction='mean')(out_dict[PREDICTION], feed_dict[Y])
        out_dict[LOSS] = loss + r_loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        out_dict[CHECK] = check_list
        return out_dict
