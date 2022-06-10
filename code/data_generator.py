# import six.moves.cPickle as pickle
import numpy as np
import pandas as pd
import os
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

        self.n_nodes = 5044482
        self.max_num_edge_embeddings = 12
        self.n_graphs = 599226

        # Creating neighbour embedding based on above edge embeddings
        list_train = {}
        relation_f = ['a_a_list.txt', 'a_b_list.txt', 'a_c_list.txt',
                      'a_d_list.txt', 'a_e_list.txt', 'a_f_list.txt',
                      'a_g_list.txt', 'a_h_list.txt', 'b_a_list.txt',
                      'b_b_list.txt', 'b_c_list.txt', 'b_d_list.txt',
                      'b_e_list.txt', 'b_h_list.txt']

        for f_name in relation_f:
            print(f'Reading relation files {f_name}')
            with open(os.path.join(self.args.data_path, f_name), 'r') as fin:
                src_type = f_name.split('_')[0]
                dst_type = f_name.split('_')[1]

                relation_type = f'{src_type}_{dst_type}'
                if relation_type not in list_train.keys():
                    list_train[relation_type] = {}

                for i, line in enumerate(fin):

                    if (i + 1) % 5000 == 0:
                        print(f"\tProcessed {i} lines")

                    line_part = line.strip().split(':')

                    gid = int(line_part[0])
                    src_node_id = int(line_part[1])

                    neigh_list = line_part[2].split(',')

                    if gid not in list_train[relation_type].keys():
                        list_train[relation_type][gid] = defaultdict(list)

                    for neigh in neigh_list:
                        list_train[relation_type][gid][src_node_id].append(int(neigh))

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

        # node-edge embedding
        node_edge_embedding_filename = os.path.join(
            self.args.data_path, "incoming_edge_embedding.csv")

        print(f'Reading Node Edge Embedding file {node_edge_embedding_filename}')
        node_edge_embeddings = pd.read_csv(node_edge_embedding_filename).to_numpy()
        # with open(node_edge_embedding_filename, 'rb') as fin:
        #     node_edge_embeddings = np.loadtxt(fin, delimiter=",", skiprows=1)

        graph_incoming_node_embedding = {}
        self.incoming_node_embedding_size = node_edge_embeddings.shape[1] - 2

        # Getting edge embedding for every node in every graph
        for row in node_edge_embeddings:
            gid = int(row[0])
            dst_id = int(row[1])

            if gid not in graph_incoming_node_embedding.keys():
                graph_incoming_node_embedding[gid] = np.zeros(
                    (self.n_nodes, self.incoming_node_embedding_size))
            graph_incoming_node_embedding[gid][dst_id] += row[2:]

        self.incoming_edge_embeddings = graph_incoming_node_embedding
        # print(graph_incoming_node_embedding[517248])

        # print(self.incoming_edge_embeddings[517248])

        # Create neighbour edge embeddings
        print('Creating Neighbour Edge Embeddings')
        # print(self.a_a_list_train)
        self.a_a_edge_embed = self.compute_edge_embeddings(self.a_a_list_train)
        self.a_b_edge_embed = self.compute_edge_embeddings(self.a_b_list_train)
        self.a_c_edge_embed = self.compute_edge_embeddings(self.a_c_list_train)
        self.a_d_edge_embed = self.compute_edge_embeddings(self.a_d_list_train)
        self.a_e_edge_embed = self.compute_edge_embeddings(self.a_e_list_train)
        self.a_f_edge_embed = self.compute_edge_embeddings(self.a_f_list_train)
        self.a_g_edge_embed = self.compute_edge_embeddings(self.a_g_list_train)
        self.a_h_edge_embed = self.compute_edge_embeddings(self.a_h_list_train)

        self.b_a_edge_embed = self.compute_edge_embeddings(self.b_a_list_train)
        self.b_b_edge_embed = self.compute_edge_embeddings(self.b_b_list_train)
        self.b_c_edge_embed = self.compute_edge_embeddings(self.b_c_list_train)
        self.b_d_edge_embed = self.compute_edge_embeddings(self.b_d_list_train)
        self.b_e_edge_embed = self.compute_edge_embeddings(self.b_e_list_train)
        self.b_h_edge_embed = self.compute_edge_embeddings(self.b_h_list_train)

        # Getting the list of graph ids in the training set
        self.train_graph_id_list = list(self.incoming_edge_embeddings.keys())

    # compute edge embedding for every graph for every source node
    def compute_edge_embeddings(self, list_train_):
        # fix the number of features embeddings in each graph to be maximum 5
        graph_edge_embedding = np.zeros(
            (self.n_graphs, self.max_num_edge_embeddings, self.incoming_node_embedding_size))
        for gid, neigh_dict in list_train_.items():
            # num_src_node = len(neigh_dict.keys())
            # graph_edge_embedding_dict[gid] = np.zeros(
            #     (num_src_node, self.incoming_node_embedding_size))

            for i, (src_id, neigh_list) in enumerate(neigh_dict.items()):
                if i >= self.max_num_edge_embeddings:
                    break
                for neigh in neigh_list:
                    try:
                        graph_edge_embedding[gid][i] += self.incoming_edge_embeddings[gid][neigh]
                    except Exception as e:
                        print(f'i: {i}')
                        print(f'gid: {gid}')
                        print(f'neigh: {neigh}')
                        self.incoming_edge_embeddings[gid][neigh]
        print(f'processed edge.')
        return graph_edge_embedding
