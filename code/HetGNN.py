import os
import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
# from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import random


torch.set_num_threads(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class model_class(object):
    def __init__(self, args):
        super(model_class, self).__init__()
        self.args = args
        self.gpu = args.cuda

        input_data = data_generator.input_data(args=self.args)
        # input_data.gen_het_rand_walk()

        self.input_data = input_data

        if self.args.train_test_label == 2:  # generate neighbor set of each node
            input_data.het_walk_restart()
            print("neighbor set generation finish")
            exit(0)

        feature_list = [
            input_data.a_a_edge_embed,
            input_data.a_b_edge_embed,
            input_data.a_c_edge_embed,
            input_data.a_d_edge_embed,
            input_data.a_e_edge_embed,
            input_data.a_f_edge_embed,
            input_data.a_g_edge_embed,
            input_data.a_h_edge_embed,
            input_data.b_a_edge_embed,
            input_data.b_b_edge_embed,
            input_data.b_c_edge_embed,
            input_data.b_d_edge_embed,
            input_data.b_e_edge_embed,
            input_data.b_h_edge_embed
        ]

        for i, fl in enumerate(feature_list):
            feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

        if self.gpu:
            for i, _ in enumerate(feature_list):
                feature_list[i] = feature_list[i].cuda()

        graph_train_id_list = input_data.train_graph_id_list

        self.model = tools.HetAgg(args, feature_list,
                                  graph_train_id_list)

        if self.gpu:
            self.model.cuda()
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)
        self.model.init_weights()

    def model_train(self):
        print('model training ...')
        if self.args.checkpoint != '':
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        self.model.train()
        batch_s = self.args.batch_s
        mini_batch_s = self.args.mini_batch_s
        embed_d = self.args.embed_d
        epoch_loss_list = []
        eval_list = []
        benign_gid_list, train_gid_list, eval_gid_list, test_gid_list = self.train_eval_test_split()

        for iter_i in range(self.args.train_iter_n):
            self.model.train()
            print('iteration ' + str(iter_i) + ' ...')
            batch_list = benign_gid_list.reshape(int(benign_gid_list.shape[0] / batch_s), batch_s)
            avg_loss_list = []
            for batch_n, k in enumerate(batch_list):
                _out = torch.zeros(int(batch_s / mini_batch_s), mini_batch_s, embed_d)

                mini_batch_list = k.reshape(int(len(k) / mini_batch_s), mini_batch_s)
                for mini_n, mini_k in enumerate(mini_batch_list):
                    _out_temp = self.model(mini_k)
                    _out[mini_n] = _out_temp

                # TODO: perhaps batch norm before fc layer
                batch_loss = tools.svdd_batch_loss(self.model, _out)
                avg_loss_list.append(batch_loss.tolist())
                print(f'\tBatch Loss: {batch_loss}')
                self.optim.zero_grad()
                batch_loss.backward(retain_graph=True)
                self.optim.step()
            print(f'Avg Loss: {np.mean(avg_loss_list)}')
            epoch_loss_list.append(np.mean(avg_loss_list))

            if iter_i % self.args.save_model_freq == 0:
                # Evaluate the model
                print("Evaluating Model ..")
                accuracy, precision, recall, f1 = self.eval_model(train_gid_list, eval_gid_list)
                eval_list.append([accuracy, precision, recall, f1])

                # Save Model
                torch.save(self.model.state_dict(), self.args.model_path +
                           "HetGNN_" + str(iter_i) + ".pt")
                # save current all epoch losses
                with open(f'{self.args.model_path}train_loss.txt', 'w') as fout:
                    for lo in epoch_loss_list:
                        fout.write(f'{lo}\n')

                with open(f'{self.args.model_path}eval_metrics.txt', 'w') as fout:
                    for accuracy, precision, recall, f1 in eval_list:
                        fout.write(f'{accuracy} {precision} {recall} {f1}\n')

            print('iteration ' + str(iter_i) + ' finish.')

    def eval_model(self, train_list, eval_list):
        self.model.eval()

        with torch.no_grad():
            train_X = self.model(train_list)
            train_X = np.array(train_X.tolist())
            train_y = np.where((train_list >= 300) & (train_list < 400), 1, 0)

            eval_X = self.model(eval_list)
            eval_X = np.array(eval_X.tolist())
            eval_y = np.where((eval_list >= 300) & (eval_list < 400), 1, 0)

            # print(f'train_X: {train_X}')
            # print(f'train_y: {train_y}')

            clf = LogisticRegression(random_state=0, solver='liblinear').fit(train_X, train_y)
            pred = clf.predict(eval_X)

            # print(f'eval_y: {eval_y}')
            # print(f'pred: {pred}')

            accuracy = accuracy_score(eval_y, pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                eval_y, pred, average='micro')
            print(f'\tAccuracy:{accuracy}, Precision:{precision}, Recall:{recall}, f1:{f1}')

        return accuracy, precision, recall, f1

    def train_eval_test_split(self):
        all_gid_list = np.array(range(600))
        benign_gid_list = all_gid_list[(all_gid_list < 300) | (all_gid_list > 399)]
        attack_gid_list = np.array(range(300, 400))

        # Train/Eval/Test = 0.6/0.2/0.2
        rep_train_benign_gid_list = np.random.choice(benign_gid_list, 360, replace=False)
        left_benign_gid_list = benign_gid_list[np.in1d(
            benign_gid_list, rep_train_benign_gid_list, invert=True)]

        train_benign_gid_list = np.random.choice(left_benign_gid_list, 84, replace=False)
        left_benign_gid_list = left_benign_gid_list[np.in1d(
            left_benign_gid_list, train_benign_gid_list, invert=True)]

        eval_benign_gid_list = np.random.choice(left_benign_gid_list, 28, replace=False)
        test_benign_gid_list = left_benign_gid_list[np.in1d(
            left_benign_gid_list, eval_benign_gid_list, invert=True)]

        train_attack_gid_list = np.random.choice(attack_gid_list, 60, replace=False)
        left_attack_gid_list = attack_gid_list[np.in1d(
            attack_gid_list, train_attack_gid_list, invert=True)]
        eval_attack_gid_list = np.random.choice(left_attack_gid_list, 20, replace=False)
        test_attack_gid_list = left_attack_gid_list[np.in1d(
            left_attack_gid_list, eval_attack_gid_list, invert=True)]

        train_gid_list = np.concatenate([train_benign_gid_list, train_attack_gid_list], axis=0)
        eval_gid_list = np.concatenate([eval_benign_gid_list, eval_attack_gid_list], axis=0)
        test_gid_list = np.concatenate([test_benign_gid_list, test_attack_gid_list], axis=0)

        np.random.shuffle(train_gid_list)
        np.random.shuffle(eval_gid_list)
        np.random.shuffle(test_gid_list)
        np.random.shuffle(rep_train_benign_gid_list)

        print(f'Representation Model Training Data Size: {rep_train_benign_gid_list.shape}')
        print(f'Clf Training Data Size: {train_gid_list.shape}')
        print(f'Clf Eval Data Size: {eval_gid_list.shape}')
        print(f'Clf Test Data Size: {test_gid_list.shape}')

        print(f'Representation Model Training Set: {rep_train_benign_gid_list}')
        print(f'Clf Training Set: {train_gid_list}')
        print(f'Clf Eval Set: {eval_gid_list}')
        print(f'Clf Test Set: {test_gid_list}')

        # write out current train/eval/test gids
        with open('../data/custom_data_simple/rep_model_train_gid_list.txt', 'w') as fout:
            for i in rep_train_benign_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open('../data/custom_data_simple/clf_train_gid_list.txt', 'w') as fout:
            for i in train_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open('../data/custom_data_simple/clf_eval_gid_list.txt', 'w') as fout:
            for i in eval_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')
        with open('../data/custom_data_simple/clf_test_gid_list.txt', 'w') as fout:
            for i in test_gid_list:
                fout.write(f'{i} ')
            fout.write('\n')

        return rep_train_benign_gid_list, train_gid_list, eval_gid_list, test_gid_list


if __name__ == '__main__':
    args = read_args()
    print("------arguments-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))

    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # model
    model_object = model_class(args)

    if args.train_test_label == 0:
        model_object.model_train()
