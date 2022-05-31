import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
args = read_args()


class HetAgg(nn.Module):
    def __init__(self, args, feature_list, a_neigh_list_train, b_neigh_list_train,
                 a_train_id_list, b_train_id_list):
        super(HetAgg, self).__init__()
        embed_d = args.embed_d
        in_f_d = args.in_f_d
        self.args = args

        self.a_num = 7919
        self.b_num = 51378

        # self.P_n = args.P_n
        # self.A_n = args.A_n
        # self.V_n = args.V_n
        self.feature_list = feature_list
        self.a_neigh_list_train = a_neigh_list_train
        self.b_neigh_list_train = b_neigh_list_train
        # self.v_neigh_list_train = v_neigh_list_train
        self.a_train_id_list = a_train_id_list
        self.b_train_id_list = b_train_id_list
        # self.v_train_id_list = v_train_id_list

        # self.fc_a_agg = nn.Linear(embed_d * 4, embed_d)

        self.a_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.b_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        # self.v_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional=True)

        self.a_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        self.b_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
        # self.v_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional=True)

        self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.b_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        # self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(embed_d)
        self.embed_d = embed_d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def a_content_agg(self, id_batch):  # heterogeneous content aggregation
        embed_d = self.embed_d
        # print len(id_batch)
        # embed_d = in_f_d, it is flexible to add feature transformer (e.g., FC) here
        # print (id_batch)
        a_net_embed_batch = self.feature_list[6][id_batch]

        a_text_embed_batch_1 = self.feature_list[7][id_batch, :embed_d][0]
        a_text_embed_batch_2 = self.feature_list[7][id_batch, embed_d: embed_d * 2][0]
        a_text_embed_batch_3 = self.feature_list[7][id_batch, embed_d * 2: embed_d * 3][0]

        concate_embed = torch.cat((a_net_embed_batch, a_text_embed_batch_1, a_text_embed_batch_2,
                                   a_text_embed_batch_3), 1).view(len(id_batch[0]), 4, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)

        return torch.mean(all_state, 0)

    def a_a_content_agg(self, id_batch):
        embed_d = self.embed_d
        a_a_edge_embed = self.feature_list[0][id_batch]

        concate_embed = a_a_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_b_content_agg(self, id_batch):
        embed_d = self.embed_d
        a_b_edge_embed = self.feature_list[1][id_batch]

        concate_embed = a_b_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_c_content_agg(self, id_batch):
        embed_d = self.embed_d
        # print(id_batch)
        a_c_edge_embed = self.feature_list[2][id_batch]

        concate_embed = a_c_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_d_content_agg(self, id_batch):
        embed_d = self.embed_d

        a_d_edge_embed = self.feature_list[3][id_batch]

        concate_embed = a_d_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_e_content_agg(self, id_batch):
        embed_d = self.embed_d

        a_e_edge_embed = self.feature_list[4][id_batch]

        concate_embed = a_e_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_f_content_agg(self, id_batch):
        embed_d = self.embed_d

        a_f_edge_embed = self.feature_list[5][id_batch]

        concate_embed = a_f_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_g_content_agg(self, id_batch):
        embed_d = self.embed_d

        a_g_edge_embed = self.feature_list[6][id_batch]

        concate_embed = a_g_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def a_h_content_agg(self, id_batch):
        embed_d = self.embed_d

        a_h_edge_embed = self.feature_list[7][id_batch]

        concate_embed = a_h_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def b_a_content_agg(self, id_batch):
        embed_d = self.embed_d

        b_a_edge_embed = self.feature_list[8][id_batch]

        concate_embed = b_a_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.b_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def b_b_content_agg(self, id_batch):
        embed_d = self.embed_d

        b_b_edge_embed = self.feature_list[9][id_batch]

        concate_embed = b_b_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.b_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def b_c_content_agg(self, id_batch):
        embed_d = self.embed_d

        b_c_edge_embed = self.feature_list[10][id_batch]

        concate_embed = b_c_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.b_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def b_d_content_agg(self, id_batch):
        embed_d = self.embed_d

        b_d_edge_embed = self.feature_list[11][id_batch]

        concate_embed = b_d_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.b_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def b_e_content_agg(self, id_batch):
        embed_d = self.embed_d

        b_e_edge_embed = self.feature_list[12][id_batch]

        concate_embed = b_e_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.b_content_rnn(concate_embed)
        return torch.mean(all_state, 0)

    def b_h_content_agg(self, id_batch):
        embed_d = self.embed_d

        b_h_edge_embed = self.feature_list[13][id_batch]

        concate_embed = b_h_edge_embed.view(len(id_batch[0]), 1, embed_d)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.b_content_rnn(concate_embed)
        return torch.mean(all_state, 0)
    # def p_content_agg(self, id_batch):
    #     embed_d = self.embed_d
    #     p_a_embed_batch = self.feature_list[0][id_batch]
    #     p_t_embed_batch = self.feature_list[1][id_batch]
    #     p_v_net_embed_batch = self.feature_list[2][id_batch]
    #     p_a_net_embed_batch = self.feature_list[3][id_batch]
    #     p_net_embed_batch = self.feature_list[5][id_batch]
    #
    #     concate_embed = torch.cat((p_a_embed_batch, p_t_embed_batch, p_v_net_embed_batch,
    #                                p_a_net_embed_batch, p_net_embed_batch), 1).view(len(id_batch[0]), 5, embed_d)
    #
    #     concate_embed = torch.transpose(concate_embed, 0, 1)
    #     all_state, last_state = self.p_content_rnn(concate_embed)
    #
    #     return torch.mean(all_state, 0)

    # def v_content_agg(self, id_batch):
    #     embed_d = self.embed_d
    #     v_net_embed_batch = self.feature_list[8][id_batch]
    #     v_text_embed_batch_1 = self.feature_list[9][id_batch, :embed_d][0]
    #     v_text_embed_batch_2 = self.feature_list[9][id_batch, embed_d: 2 * embed_d][0]
    #     v_text_embed_batch_3 = self.feature_list[9][id_batch, 2 * embed_d: 3 * embed_d][0]
    #     v_text_embed_batch_4 = self.feature_list[9][id_batch, 3 * embed_d: 4 * embed_d][0]
    #     v_text_embed_batch_5 = self.feature_list[9][id_batch, 4 * embed_d:][0]
    #
    #     concate_embed = torch.cat((v_net_embed_batch, v_text_embed_batch_1, v_text_embed_batch_2, v_text_embed_batch_3,
    #                                v_text_embed_batch_4, v_text_embed_batch_5), 1).view(len(id_batch[0]), 6, embed_d)
    #
    #     concate_embed = torch.transpose(concate_embed, 0, 1)
    #     all_state, last_state = self.v_content_rnn(concate_embed)
    #
    #     return torch.mean(all_state, 0)

    def node_neigh_agg(self, id_batch, node_type):  # type based neighbor aggregation with rnn
        embed_d = self.embed_d

        # if node_type == 1 or node_type == 2:
        batch_s = int(len(id_batch[0]) / 32)
        # else:
        #     #print (len(id_batch[0]))
        #     batch_s = int(len(id_batch[0]) / 3)
        if node_type == 1:
            neigh_agg = self.a_a_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 2:
            neigh_agg = self.a_b_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 3:
            # print(id_batch)
            neigh_agg = self.a_c_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 4:
            neigh_agg = self.a_d_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 5:
            neigh_agg = self.a_e_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 6:
            neigh_agg = self.a_f_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 7:
            neigh_agg = self.a_g_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 8:
            neigh_agg = self.a_h_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.a_neigh_rnn(neigh_agg)
        elif node_type == 9:
            neigh_agg = self.b_a_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.b_neigh_rnn(neigh_agg)
        elif node_type == 10:
            neigh_agg = self.b_b_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.b_neigh_rnn(neigh_agg)
        elif node_type == 11:
            neigh_agg = self.b_c_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.b_neigh_rnn(neigh_agg)
        elif node_type == 12:
            neigh_agg = self.b_d_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.b_neigh_rnn(neigh_agg)
        elif node_type == 13:
            neigh_agg = self.b_e_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.b_neigh_rnn(neigh_agg)
        elif node_type == 14:
            neigh_agg = self.b_h_content_agg(id_batch).view(batch_s, 32, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state = self.b_neigh_rnn(neigh_agg)

        # else:
        #     neigh_agg = self.v_content_agg(id_batch).view(batch_s, 3, embed_d)
        #     neigh_agg = torch.transpose(neigh_agg, 0, 1)
        #     all_state, last_state = self.v_neigh_rnn(neigh_agg)
        neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)

        return neigh_agg

    def node_het_agg(self, id_batch, node_type, triple_index):  # heterogeneous neighbor aggregation
        a_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        b_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        c_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        d_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        e_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        f_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        g_neigh_batch_n1 = [[0] * 10] * len(id_batch)
        h_neigh_batch_n1 = [[0] * 10] * len(id_batch)

        a_neigh_batch_n2 = [[0] * 10] * len(id_batch)
        b_neigh_batch_n2 = [[0] * 10] * len(id_batch)
        c_neigh_batch_n2 = [[0] * 10] * len(id_batch)
        d_neigh_batch_n2 = [[0] * 10] * len(id_batch)
        e_neigh_batch_n2 = [[0] * 10] * len(id_batch)
        h_neigh_batch_n2 = [[0] * 10] * len(id_batch)
        # v_neigh_batch = [[0] * 3] * len(id_batch)
        # for i in range(len(id_batch)):
        #
        #     if node_type == 1:
        #         print(id_batch[i])
        #         print("++++++++++++++++++++++id_batch")
        #         if len(self.a_neigh_list_train[0][id_batch[i]]):
        #             a_neigh_batch_n1[i] = self.a_neigh_list_train[0][id_batch[i]]
        #         if len(self.a_neigh_list_train[1][id_batch[i]]):
        #             b_neigh_batch_n1[i] = self.a_neigh_list_train[1][id_batch[i]]
        #         if len(self.a_neigh_list_train[2][id_batch[i]]):
        #             c_neigh_batch_n1[i] = self.a_neigh_list_train[2][id_batch[i]]
        #         if len(self.a_neigh_list_train[3][id_batch[i]]):
        #             d_neigh_batch_n1[i] = self.a_neigh_list_train[3][id_batch[i]]
        #         if len(self.a_neigh_list_train[4][id_batch[i]]):
        #             e_neigh_batch_n1[i] = self.a_neigh_list_train[4][id_batch[i]]
        #         if len(self.a_neigh_list_train[5][id_batch[i]]):
        #             f_neigh_batch_n1[i] = self.a_neigh_list_train[5][id_batch[i]]
        #         if len(self.a_neigh_list_train[6][id_batch[i]]):
        #             g_neigh_batch_n1[i] = self.a_neigh_list_train[6][id_batch[i]]
        #         if len(self.a_neigh_list_train[7][id_batch[i]]):
        #             h_neigh_batch_n1[i] = self.a_neigh_list_train[7][id_batch[i]]
        #
        #         # v_neigh_batch[i] = self.a_neigh_list_train[2][id_batch[i]]
        #
        #     elif node_type == 2:
        #         if len(self.b_neigh_list_train[0][id_batch[i]]):
        #             a_neigh_batch_n2[i] = self.b_neigh_list_train[0][id_batch[i]]
        #
        #         if len(self.b_neigh_list_train[1][id_batch[i]]):
        #             b_neigh_batch_n2[i] = self.b_neigh_list_train[1][id_batch[i]]
        #
        #         if len(self.b_neigh_list_train[2][id_batch[i]]):
        #             c_neigh_batch_n2[i] = self.b_neigh_list_train[2][id_batch[i]]
        #
        #         if len(self.b_neigh_list_train[3][id_batch[i]]):
        #             d_neigh_batch_n2[i] = self.b_neigh_list_train[3][id_batch[i]]
        #
        #         if len(self.b_neigh_list_train[4][id_batch[i]]):
        #             e_neigh_batch_n2[i] = self.b_neigh_list_train[4][id_batch[i]]
        #
        #         if len(self.b_neigh_list_train[5][id_batch[i]]):
        #             h_neigh_batch_n2[i] = self.b_neigh_list_train[5][id_batch[i]]

        # v_neigh_batch[i] = self.p_neigh_list_train[2][id_batch[i]]
        # else:
        #     a_neigh_batch[i] = self.v_neigh_list_train[0][id_batch[i]]
        #     p_neigh_batch[i] = self.v_neigh_list_train[1][id_batch[i]]
        # v_neigh_batch[i] = self.v_neigh_list_train[2][id_batch[i]]
        # print(id_batch)
        if node_type == 1:
            try:
                a_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
                a_agg_batch = self.node_neigh_agg(a_neigh_batch_n1, 1)
            except Exception as e:
                print(id_batch)
                raise Exception(e)
            b_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            b_agg_batch = self.node_neigh_agg(b_neigh_batch_n1, 2)
            c_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            c_agg_batch = self.node_neigh_agg(c_neigh_batch_n1, 3)
            d_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            d_agg_batch = self.node_neigh_agg(d_neigh_batch_n1, 4)
            e_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            e_agg_batch = self.node_neigh_agg(e_neigh_batch_n1, 5)
            f_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            f_agg_batch = self.node_neigh_agg(f_neigh_batch_n1, 6)
            g_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            g_agg_batch = self.node_neigh_agg(g_neigh_batch_n1, 7)
            h_neigh_batch_n1 = np.reshape(id_batch, (1, -1))
            h_agg_batch = self.node_neigh_agg(h_neigh_batch_n1, 8)
        elif node_type == 2:
            a_neigh_batch_n2 = np.reshape(id_batch, (1, -1))
            a_agg_batch = self.node_neigh_agg(a_neigh_batch_n2, 9)
            b_neigh_batch_n2 = np.reshape(id_batch, (1, -1))
            b_agg_batch = self.node_neigh_agg(b_neigh_batch_n2, 10)
            c_neigh_batch_n2 = np.reshape(id_batch, (1, -1))
            c_agg_batch = self.node_neigh_agg(c_neigh_batch_n2, 11)
            d_neigh_batch_n2 = np.reshape(id_batch, (1, -1))
            d_agg_batch = self.node_neigh_agg(d_neigh_batch_n2, 12)
            e_neigh_batch_n2 = np.reshape(id_batch, (1, -1))
            e_agg_batch = self.node_neigh_agg(e_neigh_batch_n2, 13)
            h_neigh_batch_n2 = np.reshape(id_batch, (1, -1))
            h_agg_batch = self.node_neigh_agg(h_neigh_batch_n2, 14)

        # v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
        # v_agg_batch = self.node_neigh_agg(v_neigh_batch, 3)

# TODO: Continue Here
        # attention module
        print(f'Attention Module node type: {node_type}, triple_index: {triple_index}')
        id_batch = np.reshape(id_batch, (1, -1))
        if node_type == 1 and triple_index == 0:
            cc_agg_batch = self.a_a_content_agg(id_batch)
        elif node_type == 1 and triple_index == 1:
            cc_agg_batch = self.a_b_content_agg(id_batch)
        elif node_type == 1 and triple_index == 2:
            cc_agg_batch = self.a_c_content_agg(id_batch)
        elif node_type == 1 and triple_index == 3:
            cc_agg_batch = self.a_d_content_agg(id_batch)
        elif node_type == 1 and triple_index == 4:
            cc_agg_batch = self.a_e_content_agg(id_batch)
        elif node_type == 1 and triple_index == 5:
            cc_agg_batch = self.a_f_content_agg(id_batch)
        elif node_type == 1 and triple_index == 6:
            cc_agg_batch = self.a_g_content_agg(id_batch)
        elif node_type == 1 and triple_index == 7:
            cc_agg_batch = self.a_h_content_agg(id_batch)

        elif node_type == 2 and triple_index == 8:
            cc_agg_batch = self.b_a_content_agg(id_batch)
        elif node_type == 2 and triple_index == 9:
            cc_agg_batch = self.b_b_content_agg(id_batch)
        elif node_type == 2 and triple_index == 10:
            cc_agg_batch = self.b_c_content_agg(id_batch)
        elif node_type == 2 and triple_index == 11:
            cc_agg_batch = self.b_d_content_agg(id_batch)
        elif node_type == 2 and triple_index == 12:
            cc_agg_batch = self.b_e_content_agg(id_batch)
        else:  # node_type == 2 and triple_index == 14:
            cc_agg_batch = self.b_h_content_agg(id_batch)

        if node_type == 1:
            cc_agg_batch_2 = torch.cat((cc_agg_batch, cc_agg_batch), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            # print(f"++++++++++++++++ a_agg_batch {a_agg_batch.size()}\n {a_agg_batch}")
            # print(f"++++++++++++++++ cc_agg_batch {cc_agg_batch.size()}\n {cc_agg_batch}")
            a_agg_batch_2 = torch.cat((cc_agg_batch, a_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            b_agg_batch_2 = torch.cat((cc_agg_batch, b_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            c_agg_batch_2 = torch.cat((cc_agg_batch, c_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            d_agg_batch_2 = torch.cat((cc_agg_batch, d_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            e_agg_batch_2 = torch.cat((cc_agg_batch, e_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            f_agg_batch_2 = torch.cat((cc_agg_batch, f_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            g_agg_batch_2 = torch.cat((cc_agg_batch, g_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            h_agg_batch_2 = torch.cat((cc_agg_batch, h_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)

            # compute weights
            concate_embed = torch.cat((cc_agg_batch_2, a_agg_batch_2, b_agg_batch_2,
                                       c_agg_batch_2, d_agg_batch_2, e_agg_batch_2,
                                       f_agg_batch_2, g_agg_batch_2, h_agg_batch_2), 1).view(len(cc_agg_batch), 9, self.embed_d * 2)
        else:  # node_type == 2:
            cc_agg_batch_2 = torch.cat((cc_agg_batch, cc_agg_batch), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            a_agg_batch_2 = torch.cat((cc_agg_batch, a_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            b_agg_batch_2 = torch.cat((cc_agg_batch, b_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            c_agg_batch_2 = torch.cat((cc_agg_batch, c_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            d_agg_batch_2 = torch.cat((cc_agg_batch, d_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            e_agg_batch_2 = torch.cat((cc_agg_batch, e_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)
            h_agg_batch_2 = torch.cat((cc_agg_batch, h_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(
                len(cc_agg_batch), self.embed_d * 2)

            # compute weights
            concate_embed = torch.cat((cc_agg_batch_2, a_agg_batch_2, b_agg_batch_2,
                                       c_agg_batch_2, d_agg_batch_2, e_agg_batch_2,
                                       h_agg_batch_2), 1).view(len(cc_agg_batch), 7, self.embed_d * 2)

        if node_type == 1:
            atten_w = self.act(torch.bmm(concate_embed, self.a_neigh_att.unsqueeze(0).expand(len(cc_agg_batch),
                                                                                             *self.a_neigh_att.size())))
            atten_w = self.softmax(atten_w).view(len(cc_agg_batch), 1, 9)
        else:  # node_type == 2:
            atten_w = self.act(torch.bmm(concate_embed, self.b_neigh_att.unsqueeze(0).expand(len(cc_agg_batch),
                                                                                             *self.b_neigh_att.size())))
            atten_w = self.softmax(atten_w).view(len(cc_agg_batch), 1, 7)
        # else:
        #     atten_w = self.act(torch.bmm(concate_embed, self.v_neigh_att.unsqueeze(0).expand(len(cc_agg_batch),
        #                                                                                      *self.v_neigh_att.size())))

        # weighted combination
        if node_type == 1:
            concate_embed = torch.cat((cc_agg_batch,
                                       a_agg_batch.expand(len(cc_agg_batch), -1),
                                       b_agg_batch.expand(len(cc_agg_batch), -1),
                                       c_agg_batch.expand(len(cc_agg_batch), -1),
                                       d_agg_batch.expand(len(cc_agg_batch), -1),
                                       e_agg_batch.expand(len(cc_agg_batch), -1),
                                       f_agg_batch.expand(len(cc_agg_batch), -1),
                                       g_agg_batch.expand(len(cc_agg_batch), -1), h_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(len(cc_agg_batch), 9, self.embed_d)
            weight_agg_batch = torch.bmm(atten_w, concate_embed).view(
                len(cc_agg_batch), self.embed_d)
        else:  # node_type == 2:
            concate_embed = torch.cat((cc_agg_batch,
                                       a_agg_batch.expand(len(cc_agg_batch), -1),
                                       b_agg_batch.expand(len(cc_agg_batch), -1),
                                       c_agg_batch.expand(len(cc_agg_batch), -1),
                                       d_agg_batch.expand(len(cc_agg_batch), -1),
                                       e_agg_batch.expand(len(cc_agg_batch), -1),
                                       h_agg_batch.expand(len(cc_agg_batch), -1)), 1).view(len(cc_agg_batch), 7, self.embed_d)
            weight_agg_batch = torch.bmm(atten_w, concate_embed).view(
                len(cc_agg_batch), self.embed_d)

        return weight_agg_batch

    def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
        embed_d = self.embed_d
        # batch processing
        # nine cases for academic data (author, paper, venue)
        if triple_index == 0:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 1:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 2:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 3:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 4:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 5:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 6:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 7:
            c_agg = self.node_het_agg(c_id_batch, 1, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 1, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 1, triple_index)
        elif triple_index == 8:
            c_agg = self.node_het_agg(c_id_batch, 2, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 2, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 2, triple_index)
        elif triple_index == 9:
            c_agg = self.node_het_agg(c_id_batch, 2, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 2, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 2, triple_index)
        elif triple_index == 10:
            c_agg = self.node_het_agg(c_id_batch, 2, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 2, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 2, triple_index)
        elif triple_index == 11:
            c_agg = self.node_het_agg(c_id_batch, 2, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 2, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 2, triple_index)
        elif triple_index == 12:
            c_agg = self.node_het_agg(c_id_batch, 2, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 2, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 2, triple_index)
        elif triple_index == 13:
            c_agg = self.node_het_agg(c_id_batch, 2, triple_index)
            p_agg = self.node_het_agg(pos_id_batch, 2, triple_index)
            n_agg = self.node_het_agg(neg_id_batch, 2, triple_index)

        elif triple_index == 14:  # save learned node embedding
            embed_file = open(self.args.data_path + "node_embedding.txt", "w")
            save_batch_s = self.args.mini_batch_s
            for i in range(2):
                if i == 0:
                    batch_number = int(len(self.a_train_id_list) / save_batch_s)
                elif i == 1:
                    batch_number = int(len(self.b_train_id_list) / save_batch_s)
                # else:
                #     batch_number = int(len(self.v_train_id_list) / save_batch_s)
                for j in range(batch_number):
                    if i == 0:
                        id_batch = self.a_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        triple_temp_list = []
                        for triple_id_ in range(triple_index):
                            triple_temp_list.append(self.node_het_agg(id_batch, 1, triple_id_))
                        out_temp = torch.cat(triple_temp_list, 1)
                        print(f"++++++++++ out_temp: {out_temp.size()}\n {out_temp}")
                    elif i == 1:
                        id_batch = self.b_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        triple_temp_list = []
                        for triple_id_ in range(triple_index):
                            triple_temp_list.append(self.node_het_agg(id_batch, 2, triple_id_))
                        out_temp = torch.cat(triple_temp_list, 1)
                    # else:
                    #     id_batch = self.v_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                    #     out_temp = self.node_het_agg(id_batch, 3)
                    out_temp = out_temp.data.cpu().numpy()
                    for k in range(len(id_batch)):
                        index = id_batch[k]
                        if i == 0:
                            embed_file.write('a' + str(index) + " ")
                        elif i == 1:
                            embed_file.write('b' + str(index) + " ")
                        # else:
                        #     embed_file.write('v' + str(index) + " ")
                        for l in range(embed_d - 1):
                            embed_file.write(str(out_temp[k][l]) + " ")
                        embed_file.write(str(out_temp[k][-1]) + "\n")

                if i == 0:
                    id_batch = self.a_train_id_list[batch_number * save_batch_s: -1]
                    triple_temp_list = []
                    for triple_id_ in range(triple_index):
                        triple_temp_list.append(self.node_het_agg(id_batch, 1, triple_id_))
                    out_temp = torch.cat(triple_temp_list, 1)
                elif i == 1:
                    id_batch = self.p_train_id_list[batch_number * save_batch_s: -1]
                    triple_temp_list = []
                    for triple_id_ in range(triple_index):
                        triple_temp_list.append(self.node_het_agg(id_batch, 2, triple_id_))
                    out_temp = torch.cat(triple_temp_list, 1)
                # else:
                #     id_batch = self.v_train_id_list[batch_number * save_batch_s: -1]
                #     out_temp = self.node_het_agg(id_batch, 3)
                out_temp = out_temp.data.cpu().numpy()
                for k in range(len(id_batch)):
                    index = id_batch[k]
                    if i == 0:
                        embed_file.write('a' + str(index) + " ")
                    elif i == 1:
                        embed_file.write('b' + str(index) + " ")
                    # else:
                    #     embed_file.write('v' + str(index) + " ")
                    for l in range(embed_d - 1):
                        embed_file.write(str(out_temp[k][l]) + " ")
                    embed_file.write(str(out_temp[k][-1]) + "\n")
            embed_file.close()
            return [], [], []

        return c_agg, p_agg, n_agg

    def aggregate_all(self, triple_list_batch, triple_index):
        c_id_batch = [x[0] for x in triple_list_batch]
        pos_id_batch = [x[1] for x in triple_list_batch]
        neg_id_batch = [x[2] for x in triple_list_batch]

        c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

        return c_agg, pos_agg, neg_agg

    def forward(self, triple_list_batch, triple_index):
        c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
        return c_out, p_out, n_out


def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
    batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]

    c_embed = c_embed_batch.view(batch_size, 1, embed_d)
    pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
    neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

    out_p = torch.bmm(c_embed, pos_embed)
    out_n = - torch.bmm(c_embed, neg_embed)

    sum_p = F.logsigmoid(out_p)
    sum_n = F.logsigmoid(out_n)
    loss_sum = - (sum_p + sum_n)

    #loss_sum = loss_sum.sum() / batch_size

    return loss_sum.mean()
