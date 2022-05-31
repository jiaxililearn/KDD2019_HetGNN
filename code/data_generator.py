# import six.moves.cPickle as pickle
import numpy as np
import string
import json
import re
import random
import math
from collections import Counter, defaultdict
from itertools import *


class input_data(object):
    def __init__(self, args):
        self.args = args

        # a_p_list_train = [[] for k in range(self.args.A_n)]
        # p_a_list_train = [[] for k in range(self.args.P_n)]
        # p_p_cite_list_train = [[] for k in range(self.args.P_n)]
        # v_p_list_train = [[] for k in range(self.args.V_n)]

        # number of unique node id by node type
        # a	7919
        # b	51378
        # c	4168479
        # d	1
        # e	816702
        # f	1
        # g	1
        # h	1
        in_f_d = 26
        self.a_num = 7919
        self.b_num = 51378
        c_num = 4168479
        d_num = 1
        e_num = 816702
        f_num = 1
        g_num = 1
        h_num = 1

        list_train = {}

        print('Reading relation files ..')
        relation_f = ['a_a_list.txt', 'a_b_list.txt', 'a_c_list.txt',
                      'a_d_list.txt', 'a_e_list.txt', 'a_f_list.txt',
                      'a_g_list.txt', 'a_h_list.txt', 'b_a_list.txt',
                      'b_b_list.txt', 'b_c_list.txt', 'b_d_list.txt',
                      'b_e_list.txt', 'b_h_list.txt']

        # load node id to vectorized node id mapping
        with open(self.args.data_path + "node_mapping.json", "r") as fin:
            node_type_m = json.loads(fin.read())

        # store system action relation data
        for i, f_name in enumerate(relation_f):
            neigh_f = open(self.args.data_path + f_name, "r")
            src_ = f_name.split('_')[0]
            dst_ = f_name.split('_')[1]

            for line in neigh_f:
                line = line.strip()
                node_id = int(node_type_m[re.split(':', line)[0]][1:])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                link_ = f'{src_}_{dst_}'
                if link_ not in list_train.keys():
                    if src_ == 'a':
                        list_train[link_] = [[] for k in range(self.a_num)]
                    elif src_ == 'b':
                        list_train[link_] = [[] for k in range(self.b_num)]
                for neigh in neigh_list_id:
                    list_train[link_][node_id].append(node_type_m[neigh])

                # if f_name == 'a_p_list_train.txt':
                #     for j in range(len(neigh_list_id)):
                #         a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
                # elif f_name == 'p_a_list_train.txt':
                #     for j in range(len(neigh_list_id)):
                #         p_a_list_train[node_id].append('a'+str(neigh_list_id[j]))
                # elif f_name == 'p_p_citation_list.txt':
                #     for j in range(len(neigh_list_id)):
                #         p_p_cite_list_train[node_id].append('p'+str(neigh_list_id[j]))
                # else:
                #     for j in range(len(neigh_list_id)):
                #         v_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
            neigh_f.close()
        # with open('../data/tmp.json', 'w') as fout:
        #     fout.write(json.dumps(list_train))
        # store paper venue
        # p_v = [0] * self.args.P_n
        # p_v_f = open(self.args.data_path + 'p_v.txt', "r")
        # for line in p_v_f:
        #     line = line.strip()
        #     p_id = int(re.split(',', line)[0])
        #     v_id = int(re.split(',', line)[1])
        #     p_v[p_id] = v_id
        # p_v_f.close()

        # paper neighbor: author + citation + venue
        # p_neigh_list_train = [[] for k in range(self.args.P_n)]
        # for i in range(self.args.P_n):
        #     p_neigh_list_train[i] += p_a_list_train[i]
        #     p_neigh_list_train[i] += p_p_cite_list_train[i]
        #     p_neigh_list_train[i].append('v' + str(p_v[i]))
        # print p_neigh_list_train[11846]

        # self.a_p_list_train = a_p_list_train
        # self.p_a_list_train = p_a_list_train
        # self.p_p_cite_list_train = p_p_cite_list_train
        # self.p_neigh_list_train = p_neigh_list_train
        # self.v_p_list_train = v_p_list_train

        self.a_a_list_train = list_train['a_a']
        self.a_b_list_train = list_train['a_b']
        self.a_c_list_train = list_train['a_c']
        self.a_d_list_train = list_train['a_d']
        self.a_e_list_train = list_train['a_e']
        self.a_f_list_train = list_train['a_f']
        self.a_g_list_train = list_train['a_g']
        self.a_h_list_train = list_train['a_h']

        self.b_a_list_train = list_train['b_a']
        self.b_b_list_train = list_train['b_b']
        self.b_c_list_train = list_train['b_c']
        self.b_d_list_train = list_train['b_d']
        self.b_e_list_train = list_train['b_e']
        self.b_h_list_train = list_train['b_h']

        # skip when train_test_label is 2
        if self.args.train_test_label == 2:
            print('generate random walk neighbours')
            self.het_walk_restart()
            return

        # TODO: What use of the computed sample ratio?
        # self.triple_sample_p = self.compute_sample_p()

        # store paper content pre-trained embedding
        # p_abstract_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        # p_a_e_f = open(self.args.data_path + "p_abstract_embed.txt", "r")
        # for line in islice(p_a_e_f, 1, None):
        #     values = line.split()
        #     index = int(values[0])
        #     embeds = np.asarray(values[1:], dtype='float32')
        #     p_abstract_embed[index] = embeds
        # p_a_e_f.close()

        # p_title_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        # p_t_e_f = open(self.args.data_path + "p_title_embed.txt", "r")
        # for line in islice(p_t_e_f, 1, None):
        #     values = line.split()
        #     index = int(values[0])
        #     embeds = np.asarray(values[1:], dtype='float32')
        #     p_title_embed[index] = embeds
        # p_t_e_f.close()

        # self.p_abstract_embed = p_abstract_embed
        # self.p_title_embed = p_title_embed

        # store pre-defined edge type embedding
        # a_net_embed = np.zeros((self.args.A_n, self.args.in_f_d))
        # p_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        # v_net_embed = np.zeros((self.args.V_n, self.args.in_f_d))

        # TODO: Use a pre-trained network embedding. E.g. Deepwalk
        # TODO: Change the hardcoded input feature dimension
        a_edge_embed = np.zeros((self.a_num, in_f_d))
        b_edge_embed = np.zeros((self.b_num, in_f_d))
        c_edge_embed = np.zeros((c_num, in_f_d))
        d_edge_embed = np.zeros((d_num, in_f_d))
        e_edge_embed = np.zeros((e_num, in_f_d))
        f_edge_embed = np.zeros((f_num, in_f_d))
        g_edge_embed = np.zeros((g_num, in_f_d))
        h_edge_embed = np.zeros((h_num, in_f_d))

        print("Reading initial node embedding ..")
        # net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
        edge_e_f = open(self.args.data_path + "node_edge_embedding.txt", "r")

        for line in islice(edge_e_f, 1, None):
            line = line.strip()
            index = re.split(' ', line)[0]
            if len(index):
                embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
                if node_type_m[index][0] == 'a':
                    a_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'b':
                    b_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'c':
                    c_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'd':
                    d_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'e':
                    e_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'f':
                    f_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'g':
                    g_edge_embed[int(node_type_m[index][1:])] = embeds
                elif node_type_m[index][0] == 'h':
                    h_edge_embed[int(node_type_m[index][1:])] = embeds

        edge_e_f.close()
        print('Reading initial node embedding .. Done')

        # p_v_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        # p_v = [0] * self.args.P_n
        # p_v_f = open(self.args.data_path + "p_v.txt", "r")
        # for line in p_v_f:
        #     line = line.strip()
        #     p_id = int(re.split(',', line)[0])
        #     v_id = int(re.split(',', line)[1])
        #     p_v[p_id] = v_id
        #     p_v_net_embed[p_id] = v_net_embed[v_id]
        # p_v_f.close()

        a_a_edge_embed = self.compute_pair_embedding(
            'a', 'a', self.a_num, in_f_d, self.a_a_list_train, a_edge_embed)
        a_b_edge_embed = self.compute_pair_embedding(
            'a', 'b', self.a_num, in_f_d, self.a_b_list_train, b_edge_embed)
        a_c_edge_embed = self.compute_pair_embedding(
            'a', 'c', self.a_num, in_f_d, self.a_c_list_train, c_edge_embed)
        a_d_edge_embed = self.compute_pair_embedding(
            'a', 'd', self.a_num, in_f_d, self.a_d_list_train, d_edge_embed)
        a_e_edge_embed = self.compute_pair_embedding(
            'a', 'e', self.a_num, in_f_d, self.a_e_list_train, e_edge_embed)
        a_f_edge_embed = self.compute_pair_embedding(
            'a', 'f', self.a_num, in_f_d, self.a_f_list_train, f_edge_embed)
        a_g_edge_embed = self.compute_pair_embedding(
            'a', 'g', self.a_num, in_f_d, self.a_g_list_train, g_edge_embed)
        a_h_edge_embed = self.compute_pair_embedding(
            'a', 'h', self.a_num, in_f_d, self.a_h_list_train, h_edge_embed)

        b_a_edge_embed = self.compute_pair_embedding(
            'b', 'a', self.b_num, in_f_d, self.b_a_list_train, a_edge_embed)
        b_b_edge_embed = self.compute_pair_embedding(
            'b', 'b', self.b_num, in_f_d, self.b_b_list_train, b_edge_embed)
        b_c_edge_embed = self.compute_pair_embedding(
            'b', 'c', self.b_num, in_f_d, self.b_c_list_train, c_edge_embed)
        b_d_edge_embed = self.compute_pair_embedding(
            'b', 'd', self.b_num, in_f_d, self.b_d_list_train, d_edge_embed)
        b_e_edge_embed = self.compute_pair_embedding(
            'b', 'e', self.b_num, in_f_d, self.b_e_list_train, e_edge_embed)
        b_h_edge_embed = self.compute_pair_embedding(
            'b', 'h', self.b_num, in_f_d, self.b_h_list_train, h_edge_embed)

        # p_a_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        # for i in range(self.args.P_n):
        #     if len(p_a_list_train[i]):
        #         for j in range(len(p_a_list_train[i])):
        #             a_id = int(p_a_list_train[i][j][1:])
        #             p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
        #         p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list_train[i])

        # p_ref_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        # for i in range(self.args.P_n):
        #     if len(p_p_cite_list_train[i]):
        #         for j in range(len(p_p_cite_list_train[i])):
        #             p_id = int(p_p_cite_list_train[i][j][1:])
        #             p_ref_net_embed[i] = np.add(p_ref_net_embed[i], p_net_embed[p_id])
        #         p_ref_net_embed[i] = p_ref_net_embed[i] / len(p_p_cite_list_train[i])
        #     else:
        #         p_ref_net_embed[i] = p_net_embed[i]

        # empirically use 3 paper embedding for author content embeding generation
        # a_text_embed = np.zeros((self.args.A_n, self.args.in_f_d * 3))
        # for i in range(self.args.A_n):
        #     if len(a_p_list_train[i]):
        #         feature_temp = []
        #         if len(a_p_list_train[i]) >= 3:
        #             # id_list_temp = random.sample(a_p_list_train[i], 5)
        #             for j in range(3):
        #                 feature_temp.append(p_abstract_embed[int(a_p_list_train[i][j][1:])])
        #         else:
        #             for j in range(len(a_p_list_train[i])):
        #                 feature_temp.append(p_abstract_embed[int(a_p_list_train[i][j][1:])])
        #             for k in range(len(a_p_list_train[i]), 3):
        #                 feature_temp.append(p_abstract_embed[int(a_p_list_train[i][-1][1:])])
        #
        #         feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
        #         a_text_embed[i] = feature_temp

        # empirically use 5 paper embedding for author content embeding generation
        # v_text_embed = np.zeros((self.args.V_n, self.args.in_f_d * 5))
        # for i in range(self.args.V_n):
        #     if len(v_p_list_train[i]):
        #         feature_temp = []
        #         if len(v_p_list_train[i]) >= 5:
        #             # id_list_temp = random.sample(a_p_list_train[i], 5)
        #             for j in range(5):
        #                 feature_temp.append(p_abstract_embed[int(v_p_list_train[i][j][1:])])
        #         else:
        #             for j in range(len(v_p_list_train[i])):
        #                 feature_temp.append(p_abstract_embed[int(v_p_list_train[i][j][1:])])
        #             for k in range(len(v_p_list_train[i]), 5):
        #                 feature_temp.append(p_abstract_embed[int(v_p_list_train[i][-1][1:])])
        #
        #         feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
        #         v_text_embed[i] = feature_temp

        # self.p_v = p_v
        # self.p_v_net_embed = p_v_net_embed
        # self.p_a_net_embed = p_a_net_embed
        # self.p_ref_net_embed = p_ref_net_embed
        # self.p_net_embed = p_net_embed
        # self.a_net_embed = a_net_embed
        # self.a_text_embed = a_text_embed
        # self.v_net_embed = v_net_embed
        # self.v_text_embed = v_text_embed

        self.a_a_edge_embed = a_a_edge_embed
        self.a_b_edge_embed = a_b_edge_embed
        self.a_c_edge_embed = a_c_edge_embed
        self.a_d_edge_embed = a_d_edge_embed
        self.a_e_edge_embed = a_e_edge_embed
        self.a_f_edge_embed = a_f_edge_embed
        self.a_g_edge_embed = a_g_edge_embed
        self.a_h_edge_embed = a_h_edge_embed
        self.b_a_edge_embed = b_a_edge_embed
        self.b_b_edge_embed = b_b_edge_embed
        self.b_c_edge_embed = b_c_edge_embed
        self.b_d_edge_embed = b_d_edge_embed
        self.b_e_edge_embed = b_e_edge_embed
        self.b_h_edge_embed = b_h_edge_embed

        # store neighbor set from random walk sequence
        # a_neigh_list_train = [[[] for i in range(self.args.A_n)] for j in range(3)]
        # p_neigh_list_train = [[[] for i in range(self.args.P_n)] for j in range(3)]
        # v_neigh_list_train = [[[] for i in range(self.args.V_n)] for j in range(3)]

        a_neigh_list_train = [[[] for i in range(self.a_num)] for j in range(8)]  # a-h
        b_neigh_list_train = [[[] for i in range(self.b_num)] for j in range(6)]  # a-e,h

        het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
        for line in het_neigh_train_f:
            line = line.strip()
            node_id = re.split(':', line)[0]
            neigh = re.split(':', line)[1]
            neigh_list = re.split(',', neigh)
            if node_id[0] == 'a' and len(node_id) > 1:
                for neighbour in neigh_list:
                    if neighbour[0] == 'a':
                        a_neigh_list_train[0][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'b':
                        a_neigh_list_train[1][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'c':
                        a_neigh_list_train[2][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'd':
                        a_neigh_list_train[3][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'e':
                        a_neigh_list_train[4][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'f':
                        a_neigh_list_train[5][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'g':
                        a_neigh_list_train[6][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'h':
                        a_neigh_list_train[7][int(node_id[1:])].append(neighbour[1:])
            elif node_id[0] == 'b' and len(node_id) > 1:
                for neighbour in neigh_list:
                    if neighbour[0] == 'a':
                        b_neigh_list_train[0][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'b':
                        b_neigh_list_train[1][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'c':
                        b_neigh_list_train[2][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'd':
                        b_neigh_list_train[3][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'e':
                        b_neigh_list_train[4][int(node_id[1:])].append(neighbour[1:])
                    elif neighbour[0] == 'h':
                        b_neigh_list_train[5][int(node_id[1:])].append(neighbour[1:])
            # elif node_id[0] == 'v' and len(node_id) > 1:
            #     for j in range(len(neigh_list)):
            #         if neigh_list[j][0] == 'a':
            #             v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
            #         if neigh_list[j][0] == 'p':
            #             v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
            #         if neigh_list[j][0] == 'v':
            #             v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
        het_neigh_train_f.close()
        # print a_neigh_list_train[0][1]

        # store top neighbor set (based on frequency) from random walk sequence
        a_neigh_list_train_top = [[[] for i in range(self.a_num)] for j in range(8)]
        b_neigh_list_train_top = [[[] for i in range(self.b_num)] for j in range(6)]
        # v_neigh_list_train_top = [[[] for i in range(self.args.V_n)] for j in range(3)]
        # top_k = [10, 10, 3]  # fix each neighor type size
        top_k = 10
        for i in range(self.a_num):
            for j in range(8):
                a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
                top_list = a_neigh_list_train_temp.most_common(top_k)
                neigh_size = top_k
                # if j == 0 or j == 1:
                #     neigh_size = 10
                # else:
                #     neigh_size = 3
                for k in top_list:
                    a_neigh_list_train_top[j][i].append(int(k[0]))
                if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
                        a_neigh_list_train_top[j][i].append(
                            random.choice(a_neigh_list_train_top[j][i]))

        for i in range(self.b_num):
            for j in range(6):
                b_neigh_list_train_temp = Counter(b_neigh_list_train[j][i])
                top_list = b_neigh_list_train_temp.most_common(top_k)
                neigh_size = top_k
                # if j == 0 or j == 1:
                #     neigh_size = 10
                # else:
                #     neigh_size = 3
                for k in top_list:
                    b_neigh_list_train_top[j][i].append(int(k[0]))

                if len(b_neigh_list_train_top[j][i]) and len(b_neigh_list_train_top[j][i]) < neigh_size:
                    for l in range(len(b_neigh_list_train_top[j][i]), neigh_size):
                        b_neigh_list_train_top[j][i].append(
                            random.choice(b_neigh_list_train_top[j][i]))

        # for i in range(self.args.V_n):
        #     for j in range(3):
        #         v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
        #         top_list = v_neigh_list_train_temp.most_common(top_k[j])
        #         neigh_size = 0
        #         if j == 0 or j == 1:
        #             neigh_size = 10
        #         else:
        #             neigh_size = 3
        #         for k in range(len(top_list)):
        #             v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
        #         if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
        #             for l in range(len(v_neigh_list_train_top[j][i]), neigh_size):
        #                 v_neigh_list_train_top[j][i].append(
        #                     random.choice(v_neigh_list_train_top[j][i]))

        # self.a_neigh_list_train = a_neigh_list_train
        # self.b_neigh_list_train = b_neigh_list_train

        a_neigh_list_train[:] = []
        b_neigh_list_train[:] = []
        # v_neigh_list_train[:] = []

        self.a_neigh_list_train = a_neigh_list_train_top
        self.b_neigh_list_train = b_neigh_list_train_top
        # self.v_neigh_list_train = v_neigh_list_train_top

        # store ids of author/paper/venue used in training
        train_id_list = [[] for i in range(2)]
        for i in range(self.a_num):
            for j in range(8):
                if len(a_neigh_list_train_top[j][i]):
                    train_id_list[0].append(i)
        self.a_train_id_list = np.array(train_id_list[0])

        for i in range(self.b_num):
            for j in range(6):
                if len(b_neigh_list_train_top[j][i]):
                    train_id_list[1].append(i)
        self.b_train_id_list = np.array(train_id_list[1])

        # for i in range(8):
        #     if i == 0:
        #         for l in range(self.a_num):
        #             if len(a_neigh_list_train_top[i][l]):
        #                 train_id_list[i].append(l)
        #         self.a_train_id_list = np.array(train_id_list[i])
        #     elif i == 1:
        #         for l in range(self.b_num):
        #             if len(b_neigh_list_train_top[i][l]):
        #                 train_id_list[i].append(l)
        #         self.b_train_id_list = np.array(train_id_list[i])
        # else:
        #     for l in range(self.args.V_n):
        #         if len(v_neigh_list_train_top[i][l]):
        #             train_id_list[i].append(l)
        #     self.v_train_id_list = np.array(train_id_list[i])
        # print (len(self.v_train_id_list))

    def get_a_neigh_list(self, nodeid):
        return self.a_a_list_train[nodeid] + self.a_b_list_train[nodeid] + self.a_c_list_train[nodeid] + self.a_d_list_train[nodeid] + \
            self.a_e_list_train[nodeid] + self.a_f_list_train[nodeid] + \
            self.a_g_list_train[nodeid] + self.a_h_list_train[nodeid]

    def get_b_neigh_list(self, nodeid):
        return self.b_a_list_train[nodeid] + self.b_b_list_train[nodeid] + self.b_c_list_train[nodeid] + \
            self.b_d_list_train[nodeid] + self.b_e_list_train[nodeid] + self.b_h_list_train[nodeid]

    def het_walk_restart(self):
        a_neigh_list_train = [[] for k in range(self.a_num)]
        b_neigh_list_train = [[] for k in range(self.b_num)]

        # generate neighbor set via random walk with restart
        node_n = [self.a_num, self.b_num]
        for i in range(2):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_temp = self.get_a_neigh_list(j)
                    # print(neigh_temp)
                    neigh_train = a_neigh_list_train[j]
                    curNode = "a" + str(j)
                elif i == 1:
                    neigh_temp = self.get_b_neigh_list(j)
                    neigh_train = b_neigh_list_train[j]
                    curNode = "b" + str(j)

                if len(neigh_temp):
                    neigh_L = 0
                    # a_L = 0
                    # p_L = 0
                    # v_L = 0
                    while neigh_L < 100:  # maximum neighbor size = 100
                        rand_p = random.random()  # return p
                        if rand_p > 0.5:
                            if curNode[0] == "a":
                                # curNode = random.choice(self.a_p_list_train[int(curNode[1:])])
                                curNode = random.choice(self.get_a_neigh_list(int(curNode[1:])))
                                # size constraint (make sure each type of neighobr is sampled)
                                # if p_L < 46:
                                neigh_train.append(curNode)
                                neigh_L += 1
                                # p_L += 1
                            elif curNode[0] == "b":
                                # curNode = random.choice(self.p_neigh_list_train[int(curNode[1:])])
                                if len(self.get_b_neigh_list(int(curNode[1:]))) == 0:
                                    if i == 0:
                                        curNode = ('a' + str(j))
                                    elif i == 1:
                                        curNode = ('b' + str(j))
                                    continue
                                curNode = random.choice(self.get_b_neigh_list(int(curNode[1:])))
                                # if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 46:
                                neigh_train.append(curNode)
                                neigh_L += 1
                            else:
                                if i == 0:
                                    curNode = ('a' + str(j))
                                elif i == 1:
                                    curNode = ('b' + str(j))
                                #     a_L += 1
                                # elif curNode[0] == 'v':
                                #     if v_L < 11:
                                #         neigh_train.append(curNode)
                                #         neigh_L += 1
                                #         v_L += 1
                            # elif curNode[0] == "v":
                            #     curNode = random.choice(self.v_p_list_train[int(curNode[1:])])
                            #     if p_L < 46:
                            #         neigh_train.append(curNode)
                            #         neigh_L += 1
                            #         p_L += 1
                        else:
                            if i == 0:
                                curNode = ('a' + str(j))
                            elif i == 1:
                                curNode = ('b' + str(j))
                            # else:
                            #     curNode = ('v' + str(j))

        for i in range(2):
            for j in range(node_n[i]):
                if i == 0:
                    a_neigh_list_train[j] = list(a_neigh_list_train[j])
                elif i == 1:
                    b_neigh_list_train[j] = list(b_neigh_list_train[j])
                # else:
                #     v_neigh_list_train[j] = list(v_neigh_list_train[j])

        neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
        for i in range(2):
            for j in range(node_n[i]):
                if i == 0:
                    neigh_train = a_neigh_list_train[j]
                    curNode = "a" + str(j)
                elif i == 1:
                    neigh_train = b_neigh_list_train[j]
                    curNode = "b" + str(j)
                else:
                    neigh_train = 0
                # else:
                #     neigh_train = v_neigh_list_train[j]
                #     curNode = "v" + str(j)
                if len(neigh_train):
                    neigh_f.write(curNode + ":")
                    for k in range(len(neigh_train) - 1):
                        neigh_f.write(neigh_train[k] + ",")
                    neigh_f.write(neigh_train[-1] + "\n")
        neigh_f.close()

    def compute_sample_p(self):
        print("computing sampling ratio for each kind of triple ...")
        window = self.args.window
        walk_L = self.args.walk_L
        A_n = self.args.A_n
        P_n = self.args.P_n
        V_n = self.args.V_n

        total_triple_n = [0.0] * 9  # nine kinds of triples
        het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
        centerNode = ''
        neighNode = ''

        for line in het_walk_f:
            line = line.strip()
            path = []
            path_list = re.split(' ', line)
            for i in range(len(path_list)):
                path.append(path_list[i])
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[0] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[1] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[2] += 1
                    elif centerNode[0] == 'p':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[3] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[4] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[5] += 1
                    elif centerNode[0] == 'v':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    total_triple_n[6] += 1
                                elif neighNode[0] == 'p':
                                    total_triple_n[7] += 1
                                elif neighNode[0] == 'v':
                                    total_triple_n[8] += 1
        het_walk_f.close()

        for i in range(len(total_triple_n)):
            total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
        print("sampling ratio computing finish.")

        return total_triple_n

    def sample_het_walk_triple(self):
        print("sampling triple relations ...")
        triple_list = [[] for k in range(14)]
        window = self.args.window
        walk_L = self.args.walk_L
        a_n = self.a_num
        b_n = self.b_num
        # V_n = self.args.V_n
        # triple_sample_p = self.triple_sample_p  # use sampling to avoid memory explosion

        # TODO: Currently using the simple random walk from het_walk_restart. May implement random walk specifically
        # het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
        het_walk_f = open(self.args.data_path + "het_neigh_train.txt", "r")
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            node_src, node_dst = line.split(':')
            node_dst_list = node_dst.split(',')
            path = []
            # path_list = re.split(' ', line)
            # path_list =
            path.append(node_src)
            path = path + node_dst_list
            # print(path)
            # for i in range(len(path_list)):
            #     path.append(path_list[i])
            for j in range(walk_L):
                centerNode = path[j]
                if len(centerNode) > 1:
                    if centerNode[0] == 'a':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_a_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # random negative sampling get similar performance as noise distribution sampling
                                    # TODO: keep neighNode to be same as negNode for now to bypass the exception
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[0].append(triple)
                                elif neighNode[0] == 'b':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_b_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[1].append(triple)
                                elif neighNode[0] == 'c':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_c_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[2].append(triple)
                                elif neighNode[0] == 'd':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_d_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[3].append(triple)
                                elif neighNode[0] == 'e':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_e_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[4].append(triple)
                                elif neighNode[0] == 'f':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_f_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[5].append(triple)
                                elif neighNode[0] == 'g':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_g_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[6].append(triple)
                                elif neighNode[0] == 'h':
                                    negNode = random.randint(0, a_n - 1)
                                    while len(self.a_h_list_train[negNode]) == 0:
                                        negNode = random.randint(0, a_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[7].append(triple)

                    elif centerNode[0] == 'b':
                        for k in range(j - window, j + window + 1):
                            if k and k < walk_L and k != j:
                                neighNode = path[k]
                                if neighNode[0] == 'a':
                                    negNode = random.randint(0, b_n - 1)
                                    while len(self.b_a_list_train[negNode]) == 0:
                                        negNode = random.randint(0, b_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[8].append(triple)
                                elif neighNode[0] == 'b':
                                    negNode = random.randint(0, b_n - 1)
                                    while len(self.b_b_list_train[negNode]) == 0:
                                        negNode = random.randint(0, b_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[9].append(triple)
                                elif neighNode[0] == 'c':
                                    negNode = random.randint(0, b_n - 1)
                                    while len(self.b_c_list_train[negNode]) == 0:
                                        negNode = random.randint(0, b_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[10].append(triple)
                                elif neighNode[0] == 'd':
                                    negNode = random.randint(0, b_n - 1)
                                    while len(self.b_d_list_train[negNode]) == 0:
                                        negNode = random.randint(0, b_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[11].append(triple)
                                elif neighNode[0] == 'e':
                                    negNode = random.randint(0, b_n - 1)
                                    while len(self.b_e_list_train[negNode]) == 0:
                                        negNode = random.randint(0, b_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[12].append(triple)
                                elif neighNode[0] == 'h':
                                    negNode = random.randint(0, b_n - 1)
                                    while len(self.b_h_list_train[negNode]) == 0:
                                        negNode = random.randint(0, b_n - 1)
                                    # triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                                    triple = [int(centerNode[1:]), int(negNode), int(negNode)]
                                    triple_list[13].append(triple)

                    # elif centerNode[0] == 'v':
                    #     for k in range(j - window, j + window + 1):
                    #         if k and k < walk_L and k != j:
                    #             neighNode = path[k]
                    #             if neighNode[0] == 'a':  # and random.random() < triple_sample_p[6]:
                    #                 negNode = random.randint(0, a_n - 1)
                    #                 while len(self.a_p_list_train[negNode]) == 0:
                    #                     negNode = random.randint(0, a_n - 1)
                    #                 triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                    #                 triple_list[6].append(triple)
                    #             # and random.random() < triple_sample_p[7]:
                    #             elif neighNode[0] == 'p':
                    #                 negNode = random.randint(0, b_n - 1)
                    #                 while len(self.p_a_list_train[negNode]) == 0:
                    #                     negNode = random.randint(0, b_n - 1)
                    #                 triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                    #                 triple_list[7].append(triple)
                    #             # and random.random() < triple_sample_p[8]:
                    #             elif neighNode[0] == 'v':
                    #                 negNode = random.randint(0, V_n - 1)
                    #                 while len(self.v_p_list_train[negNode]) == 0:
                    #                     negNode = random.randint(0, V_n - 1)
                    #                 triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
                    #                 triple_list[8].append(triple)
        het_walk_f.close()

        return triple_list

    def compute_pair_embedding(self, src_type, dst_type, x_num, in_f_d, pair_list_train, dst_embed):
        print(f"Comput {src_type}_{dst_type} embedding ..")

        pair_edge_embed = np.zeros((x_num, in_f_d))
        for i in range(x_num):
            if len(pair_list_train[i]):
                for j in pair_list_train[i]:
                    _id = int(j[1:])
                    pair_edge_embed[i] = np.add(pair_edge_embed[i], dst_embed[_id])
                pair_edge_embed[i] = pair_edge_embed[i] / len(pair_list_train[i])

        print(pair_edge_embed.sum())
        return pair_edge_embed
