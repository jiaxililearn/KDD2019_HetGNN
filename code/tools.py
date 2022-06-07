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
    def __init__(self, args, feature_list, graph_train_id_list):
        # a_neigh_list_train, b_neigh_list_train,
        #  a_train_id_list, b_train_id_list):

        super(HetAgg, self).__init__()
        embed_d = args.embed_d

        self.args = args

        self.feature_list = feature_list
        self.graph_train_id_list = graph_train_id_list

        self.fc_a_a_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_b_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_c_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_d_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_e_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_f_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_g_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_a_h_agg = nn.Linear(embed_d * 12, embed_d)

        self.fc_b_a_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_b_b_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_b_c_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_b_d_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_b_e_agg = nn.Linear(embed_d * 12, embed_d)
        self.fc_b_h_agg = nn.Linear(embed_d * 12, embed_d)

        self.fc_het_neigh_agg = nn.Linear(embed_d * 14, embed_d)

        # self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        # self.drop = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(embed_d * 12)
        self.bn2 = nn.BatchNorm1d(embed_d * 14)
        self.embed_d = embed_d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge_content_agg(self, gid_batch, edge_type):

        embed_d = self.embed_d
        if edge_type == 'a_a':
            edge_embed = self.feature_list[0][gid_batch]
            fc_agg = self.fc_a_a_agg
        elif edge_type == 'a_b':
            edge_embed = self.feature_list[1][gid_batch]
            fc_agg = self.fc_a_b_agg
        elif edge_type == 'a_c':
            edge_embed = self.feature_list[2][gid_batch]
            fc_agg = self.fc_a_c_agg
        elif edge_type == 'a_d':
            edge_embed = self.feature_list[3][gid_batch]
            fc_agg = self.fc_a_d_agg
        elif edge_type == 'a_e':
            edge_embed = self.feature_list[4][gid_batch]
            fc_agg = self.fc_a_e_agg
        elif edge_type == 'a_f':
            edge_embed = self.feature_list[5][gid_batch]
            fc_agg = self.fc_a_f_agg
        elif edge_type == 'a_g':
            edge_embed = self.feature_list[6][gid_batch]
            fc_agg = self.fc_a_g_agg
        elif edge_type == 'a_h':
            edge_embed = self.feature_list[7][gid_batch]
            fc_agg = self.fc_a_h_agg
        elif edge_type == 'b_a':
            edge_embed = self.feature_list[8][gid_batch]
            fc_agg = self.fc_b_a_agg
        elif edge_type == 'b_b':
            edge_embed = self.feature_list[9][gid_batch]
            fc_agg = self.fc_b_b_agg
        elif edge_type == 'b_c':
            edge_embed = self.feature_list[10][gid_batch]
            fc_agg = self.fc_b_c_agg
        elif edge_type == 'b_d':
            edge_embed = self.feature_list[11][gid_batch]
            fc_agg = self.fc_b_d_agg
        elif edge_type == 'b_e':
            edge_embed = self.feature_list[12][gid_batch]
            fc_agg = self.fc_b_e_agg
        # elif edge_type == 'b_h':
        else:
            edge_embed = self.feature_list[13][gid_batch]
            fc_agg = self.fc_b_h_agg

        concate_embed = edge_embed.view(len(gid_batch), 1, embed_d * 12)
        concate_embed = torch.transpose(concate_embed, 0, 1)
        output = fc_agg(concate_embed)
        # return torch.mean(output, 0)
        return self.act(output).view(len(gid_batch), embed_d)

    def node_neigh_agg(self, gid_batch, edge_type):  # type based neighbor aggregation with rnn
        # embed_d = self.embed_d
        neigh_agg = self.edge_content_agg(gid_batch, edge_type)
        return neigh_agg

    # heterogeneous neighbor aggregation
    def node_het_agg(self, gid_batch):

        a_a_agg_batch = self.node_neigh_agg(gid_batch, 'a_a')
        a_b_agg_batch = self.node_neigh_agg(gid_batch, 'a_b')
        a_c_agg_batch = self.node_neigh_agg(gid_batch, 'a_c')
        a_d_agg_batch = self.node_neigh_agg(gid_batch, 'a_d')
        a_e_agg_batch = self.node_neigh_agg(gid_batch, 'a_e')
        a_f_agg_batch = self.node_neigh_agg(gid_batch, 'a_f')
        a_g_agg_batch = self.node_neigh_agg(gid_batch, 'a_g')
        a_h_agg_batch = self.node_neigh_agg(gid_batch, 'a_h')

        b_a_agg_batch = self.node_neigh_agg(gid_batch, 'b_a')
        b_b_agg_batch = self.node_neigh_agg(gid_batch, 'b_b')
        b_c_agg_batch = self.node_neigh_agg(gid_batch, 'b_c')
        b_d_agg_batch = self.node_neigh_agg(gid_batch, 'b_d')
        b_e_agg_batch = self.node_neigh_agg(gid_batch, 'b_e')
        b_h_agg_batch = self.node_neigh_agg(gid_batch, 'b_h')

        agg_batch = torch.cat((a_a_agg_batch, a_b_agg_batch, a_c_agg_batch,
                               a_d_agg_batch, a_e_agg_batch, a_f_agg_batch,
                               a_g_agg_batch, a_h_agg_batch, b_a_agg_batch,
                               b_b_agg_batch, b_c_agg_batch, b_d_agg_batch,
                               b_e_agg_batch, b_h_agg_batch),
                              1).view(len(a_a_agg_batch), self.embed_d * 14)

        het_agg_batch = self.act(self.fc_het_neigh_agg(agg_batch))

        # skip attention module
        # atten_w = self.act(
        #     torch.bmm(concat_embed,)
        # )
        return het_agg_batch

    def het_agg(self, gid_batch):
        # aggregate heterogeneous neighbours
        _agg = self.node_het_agg(gid_batch)
        return _agg

    def aggregate_all(self, gid_batch):
        _agg = self.het_agg(gid_batch)
        return _agg

    def forward(self, gid_batch):
        _out = self.aggregate_all(gid_batch)
        return _out


# SVDD Loss
def svdd_batch_loss(model, embed_batch, l2_lambda=0.001):  # nu: {0.1, 0.01}
    l2_lambda = l2_lambda

    _batch_out = embed_batch
    _batch_out_resahpe = _batch_out.view(_batch_out.size()[0] * _batch_out.size()[1], args.embed_d)

    hypersphere_center = torch.mean(_batch_out_resahpe, 0)

    dist = torch.square(_batch_out_resahpe - hypersphere_center)
    loss_ = torch.mean(torch.sum(dist, 1))

    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    loss = loss_ + l2_lambda * l2_norm
    return loss


# Original loss imp
# def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
#     batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
#
#     c_embed = c_embed_batch.view(batch_size, 1, embed_d)
#     pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
#     neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)
#
#     out_p = torch.bmm(c_embed, pos_embed)
#     out_n = - torch.bmm(c_embed, neg_embed)
#
#     sum_p = F.logsigmoid(out_p)
#     sum_n = F.logsigmoid(out_n)
#     loss_sum = - (sum_p + sum_n)
#
#     # loss_sum = loss_sum.sum() / batch_size
#
#     return loss_sum.mean()
