from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import json
import math
import sys
import os
import time
np.set_printoptions(threshold=np.nan)
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from function import read_merged_data, extract_feature_and_label, reshape_data_into_2_dim, \
    transform_dataframe2array, filter_missing_values
from evaluation import roc_auc_single, roc_auc_multi, precision_auc_multi, precision_auc_single, \
    enrichment_factor_single


eval_mean_or_median = np.mean


def get_model_precision_auc(true_label,
                            predicted_label,
                            sample_weight):
    eval_indices = np.arange(true_label.shape[1])
    global eval_mean_or_median
    return precision_auc_multi(true_label, predicted_label, sample_weight,
                               eval_indices, eval_mean_or_median)


def get_model_roc_auc(true_label,
                      predicted_label,
                      sample_weight):
    eval_indices = np.arange(true_label.shape[1])
    global eval_mean_or_median
    return roc_auc_multi(true_label, predicted_label, sample_weight,
                         eval_indices, eval_mean_or_median)


class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)


class PCBADataset(Dataset):
    def __init__(self, data_files, feature_name, task_list, transform):
        column_names = [feature_name]
        column_names.extend(task_list)
        pcba_frame = read_merged_data(data_files)
        pcba_frame = pcba_frame[column_names]
        pcba_frame.dropna(inplace=True, subset=task_list, how='all')
        print('Update data shape\t', pcba_frame.shape)
        weight_frame = filter_missing_values(pcba_frame[task_list].copy())

        self.feature_name = feature_name
        self.task_list = task_list
        self.transform = transform
        pcba_frame.fillna(0, inplace=True)
        self.data, self.label = extract_feature_and_label(pcba_frame,
                                                          feature_name=self.feature_name,
                                                          task_list=self.task_list)
        self.weight = transform_dataframe2array(weight_frame)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label_sample = self.label[idx]
        weight_sample = self.weight[idx]
        if self.transform:
            data_sample = self.transform(data_sample)
            label_sample = self.transform(label_sample)
            weight_sample = self.transform(weight_sample)
        return data_sample, label_sample, weight_sample


class NeuralNetwork(nn.Module):
    def __init__(self, conf):
        super(NeuralNetwork, self).__init__()

        layer_config = conf['layers']
        feature_num = 1024
        self.task_number = len(conf['task_list'])
        self.drop_out = float(conf['layers']['dropout'])
        def activation_map(activation):
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'sigmoid':
                return nn.Sigmoid()

        self.batch_normalizer_epsilon = conf['batch']['epsilon']
        self.batch_normalizer_axis = conf['batch']['axis']
        self.batch_normalizer_momentum = conf['batch']['momentum']
        self.batch_normalizer_beta_init = conf['batch']['beta_init']
        self.batch_normalizer_gamma_init = conf['batch']['gamma_init']

        temp = [feature_num] + layer_config['hidden_units'] + [self.task_number]
        shared_depth = len(temp) - 2
        shared_layers = []
        last_layer = nn.ModuleList()
        for layer_id,(in_, out_) in enumerate(zip(temp[:-1], temp[1:])):
            if layer_id < shared_depth:
                shared_layers.append(nn.Linear(in_, out_))
                shared_layers.append(nn.Dropout(p=self.drop_out))
                activation = layer_config['activations'][layer_id]
                shared_layers.append(activation_map(activation))
            else:
                activation = layer_config['activations'][layer_id]
                for task in range(out_):
                    layer_ = nn.Linear(in_, 1)
                    setattr(self, 'task_{}'.format(task), layer_)
                    last_layer.extend(
                        [nn.Sequential(layer_, activation_map(activation))]
                    )
            if layer_id + 1 == shared_depth:
                shared_layers.append(nn.BatchNorm1d(out_, eps=self.batch_normalizer_epsilon, momentum=self.batch_normalizer_momentum))

        self.shared_layers = nn.Sequential(*shared_layers)
        self.last_layer = last_layer

    def forward(self, batch):
        x = self.shared_layers(batch)
        out = []
        for i in range(self.task_number):
            out.append(self.last_layer[i](x).squeeze())
        x = torch.stack(out, dim=1)
        return x

    def share_projection(self, batch):
        batch = self.shared_layers(batch)
        return batch

    def last_projection(self, batch):
        batch = self.last_layer(batch)
        return batch


class MultiTaskModel:
    def __init__(self, conf, **kwargs):
        # this is the threshold value which determined the policy
        self.predefined_lambda = 0.0

        self.conf = conf
        self.model_weight_file = kwargs['file_path']
        self.score_path = kwargs['score_path']
        self.enable_gpu = kwargs['enable_gpu']
        self.task_list = conf['task_list']

        self.training_dataset = kwargs['training_dataset']
        self.validation_dataset = kwargs['validation_dataset']
        self.test_dataset = kwargs['test_dataset']

        self.task_number = len(conf['task_list'])

        self.eval_pr_function = get_model_precision_auc
        self.eval_roc_function = get_model_roc_auc

        self.early_stopping_patience = conf['fitting']['early_stopping']['patience']
        self.early_stopping_option = conf['fitting']['early_stopping']['option']
        self.stop_training = False

        self.fit_nb_epoch = conf['fitting']['nb_epoch']
        self.fit_batch_size = conf['fitting']['batch_size']
        self.fit_verbose = conf['fitting']['verbose']

        self.compile_loss = conf['compile']['loss']
        self.compile_optimizer_option = conf['compile']['optimizer']['option']
        if self.compile_optimizer_option == 'sgd':
            self.sgd_lr = conf['compile']['optimizer']['sgd']['lr']
            self.sgd_momentum = conf['compile']['optimizer']['sgd']['momentum']
            self.sgd_decay = conf['compile']['optimizer']['sgd']['decay']
            self.sgd_nestrov = conf['compile']['optimizer']['sgd']['nestrov']
        else:
            self.adam_lr = conf['compile']['optimizer']['adam']['lr']
            self.adam_beta_1 = conf['compile']['optimizer']['adam']['beta_1']
            self.adam_beta_2 = conf['compile']['optimizer']['adam']['beta_2']
            self.adam_epsilon = conf['compile']['optimizer']['adam']['epsilon']

        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.weight_init_stddevs = 0.02
        self.bias_init_consts = 1.0
        self.penalty = 0.0005

        torch.manual_seed(kwargs['seed'])
        if self.enable_gpu:
            torch.cuda.manual_seed(kwargs['seed'])

    def weights_init(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
            m.weight.data.normal_(0.0, self.weight_init_stddevs).clamp_(min=-2, max=2)
            m.bias.data.fill_(self.bias_init_consts)
        elif 'BatchNorm1d' in classname:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        return

    def build_model(self):
        self.fc_nn = NeuralNetwork(self.conf)
        print('structure\n{}'.format(self.fc_nn))
        self.fc_nn.apply(self.weights_init)
        if self.compile_optimizer_option == 'sgd':
            self.optimizer = optim.SGD(self.fc_nn.parameters(),
                                       lr=self.sgd_lr,
                                       momentum=self.sgd_momentum,
                                       weight_decay=self.sgd_decay,
                                       nestrov=self.sgd_nestrov)
        else:
            self.optimizer = optim.Adam(self.fc_nn.parameters(),
                                        lr=self.adam_lr,
                                        betas=(self.adam_beta_1, self.adam_beta_2),
                                        eps=self.adam_epsilon,
                                        weight_decay=self.penalty)
        return

    def single_task_cost(self, logit, label, sample_weight):
        loss = F.binary_cross_entropy(logit, label, weight=sample_weight)
        return loss

    def multi_task_cost(self, outputs, labels, sample_weights, reduce=True, class_weights=None):
        weighted_costs = []
        for task in range(self.task_number):
            if class_weights is None:
                weighted_cost = self.single_task_cost(outputs[:, task], labels[:, task], sample_weights[:, task])
            else:
                weighted_cost = self.single_task_cost(outputs[:, task], labels[:, task], sample_weights[:, task]) * class_weights[task]
            weighted_costs.append(weighted_cost)
        loss = torch.stack(weighted_costs).squeeze()

        if reduce:
            return loss.sum()
        return loss

    def predict(self, dataset):
        actual = []
        logits = []
        weights = []
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.fit_batch_size,
                                                 shuffle=False, num_workers=1)
        for j, (X_batch, y_batch, sampel_weight_batch) in enumerate(dataloader):
            if self.enable_gpu:
                X_batch = Variable(X_batch.float().cuda())
                y_batch = Variable(y_batch.float().cuda())
                sampel_weight_batch = Variable(sampel_weight_batch.float().cuda())
            else:
                X_batch = Variable(X_batch.float())
                y_batch = Variable(y_batch.float())
                sampel_weight_batch = Variable(sampel_weight_batch.float())

            temp = self.fc_nn(X_batch)
            if self.enable_gpu:
                actual_label = y_batch.data.cpu().numpy()
                batch_output = temp.data.cpu().numpy()
                sample_weight = sampel_weight_batch.data.cpu().numpy()
            else:
                actual_label = y_batch.data.numpy()
                batch_output = temp.data.numpy()
                sample_weight = sampel_weight_batch.data.numpy()

            weights.append(sample_weight)
            actual.append(actual_label)
            logits.append(batch_output)
        return np.vstack(actual), np.vstack(logits), np.vstack(weights)

    def on_train_begin(self):
        y_val, y_pred_on_val, sample_weight_val = self.predict(self.validation_dataset)
        self.best_pr = self.eval_pr_function(y_val, y_pred_on_val, sample_weight_val)
        self.save_model(self.model_weight_file)
        self.start_time = time.time()
        self.counter = 0
        return

    def on_epoch_begin(self):
        # switch to train mode
        self.fc_nn.train()
        self.start_time = time.time()
        return

    def on_epoch_end(self, avg_loss, focussed=False):
        end_time = time.time()
        duration = end_time - self.start_time
        print('Duration for one data pass is {},\taverage loss is {}.'.
              format(duration, avg_loss))

        # start to validate
        self.fc_nn.eval()

        # compute output
        y_train, logits_train, sample_weight_train = self.predict(self.training_dataset)
        y_val, logits_val, sample_weight_val = self.predict(self.validation_dataset)

        if focussed:
            print('In focussed learning, truncatation')
            print(y_train.shape)
            print(logits_train.shape)
            print(y_val.shape)
            y_train = y_train[:, 0:1]
            logits_train = logits_train[:, 0:1]
            sample_weight_train = sample_weight_train[:, 0:1]
            y_val = y_val[:, 0:1]
            logits_val = logits_val[:, 0:1]
            sample_weight_val = sample_weight_val[:, 0:1]
            print(y_train.shape)
            print(logits_train.shape)
            print(y_val.shape)

        train_precision = self.eval_pr_function(y_train, logits_train, sample_weight_train)
        train_roc = self.eval_roc_function(y_train, logits_train, sample_weight_train)
        val_precision = self.eval_pr_function(y_val, logits_val, sample_weight_val)
        val_roc = self.eval_roc_function(y_val, logits_val, sample_weight_val)

        print('train precision, {}'.format(train_precision))
        print('train roc, {}'.format(train_roc))
        print('validation precision, {}'.format(val_precision))
        print('validation roc, {}'.format(val_roc))
        print()

        if val_precision > self.best_pr:
            self.counter = 0
            self.best_pr = val_precision
            self.save_model(self.model_weight_file)
        else:
            self.counter += 1
            if self.counter > self.early_stopping_patience:
                self.stop_training = True
        return

    def on_train_end(self):
        self.fc_nn = self.load_best_model()
        # start to validate
        self.fc_nn.eval()

        # start to validate
        y_train, logits_train, sample_weight_train = self.predict(self.training_dataset)
        y_val, logits_val, sample_weight_val = self.predict(self.validation_dataset)
        y_test, logits_test, sample_weight_test = self.predict(self.test_dataset)

        print()
        y_true = y_train
        y_pred = logits_train
        sample_weight = sample_weight_train
        print('train precision, {}'.format(self.eval_pr_function(y_true, y_pred, sample_weight)))
        print('train roc, {}'.format(self.eval_roc_function(y_true, y_pred, sample_weight)))
        y_true = y_val
        y_pred = logits_val
        sample_weight = sample_weight_val
        print('validation precision, {}'.format(self.eval_pr_function(y_true, y_pred, sample_weight)))
        print('validation roc, {}'.format(self.eval_roc_function(y_true, y_pred, sample_weight)))
        y_true = y_test
        y_pred = logits_test
        sample_weight = sample_weight_test
        print('test precision, {}'.format(self.eval_pr_function(y_true, y_pred, sample_weight)))
        print('test roc, {}'.format(self.eval_roc_function(y_true, y_pred, sample_weight)))
        print()

        train_pr_auc, train_roc_auc = [], []
        val_pr_auc, val_roc_auc = [], []
        test_pr_auc, test_roc_auc = [], []

        # Store all the target evaluation.
        if not os.path.exists(self.score_path):
            os.mkdir(self.score_path)
        for i in range(len(self.task_list)):
            label_name = self.task_list[i]
            file_ = open(self.score_path + '/{}.out'.format(label_name), 'w')
            print('{}'.format(label_name), file=file_)
            print('', file=file_)

            y_true = reshape_data_into_2_dim(y_train[:, i])
            y_pred = reshape_data_into_2_dim(logits_train[:, i])
            sample_weight = reshape_data_into_2_dim(sample_weight_train[:, i])
            pr_auc = precision_auc_single(y_true, y_pred, sample_weight)
            roc_auc = roc_auc_single(y_true, y_pred, sample_weight)
            train_pr_auc.append(pr_auc)
            train_roc_auc.append(roc_auc)
            print('train precision, {}'.format(pr_auc), file=file_)
            print('train roc, {}'.format(roc_auc), file=file_)
            print('', file=file_)

            y_true = reshape_data_into_2_dim(y_val[:, i])
            y_pred = reshape_data_into_2_dim(logits_val[:, i])
            sample_weight = reshape_data_into_2_dim(sample_weight_val[:, i])
            pr_auc = precision_auc_single(y_true, y_pred, sample_weight)
            roc_auc = roc_auc_single(y_true, y_pred, sample_weight)
            val_pr_auc.append(pr_auc)
            val_roc_auc.append(roc_auc)
            print('validation precision, {}'.format(pr_auc), file=file_)
            print('validation roc, {}'.format(roc_auc), file=file_)
            print('', file=file_)

            y_true = reshape_data_into_2_dim(y_test[:, i])
            y_pred = reshape_data_into_2_dim(logits_test[:, i])
            sample_weight = reshape_data_into_2_dim(sample_weight_test[:, i])
            pr_auc = precision_auc_single(y_true, y_pred, sample_weight)
            roc_auc = roc_auc_single(y_true, y_pred, sample_weight)
            test_pr_auc.append(pr_auc)
            test_roc_auc.append(roc_auc)
            print('test precision, {}'.format(pr_auc), file=file_)
            print('test roc, {}'.format(roc_auc), file=file_)
            print('', file=file_)

            for EF_ratio in self.EF_ratio_list:
                n_actives, ef, ef_max = enrichment_factor_single(y_true, y_pred, EF_ratio, sample_weight)
                print('EF ratio, {}, EF, {}, active, {}'.format(EF_ratio, ef, n_actives), file=file_)

        print('Training AUC[PR]\n{}'.format(train_pr_auc))
        print('average is: {}'.format(np.mean(train_pr_auc)))
        print()
        print('Training AUC[ROC]\n{}'.format(train_roc_auc))
        print('average is: {}'.format(np.mean(train_roc_auc)))
        print()

        print('Validation AUC[PR]\n{}'.format(val_pr_auc))
        print('average is: {}'.format(np.mean(val_pr_auc)))
        print()
        print('Validation AUC[ROC]\n{}'.format(val_roc_auc))
        print('average is: {}'.format(np.mean(val_roc_auc)))
        print()

        print('Test AUC[PR]\n{}'.format(test_pr_auc))
        print('average is: {}'.format(np.mean(test_pr_auc)))
        print()
        print('Test AUC[ROC]\n{}'.format(test_roc_auc))
        print('average is: {}'.format(np.mean(test_roc_auc)))
        print()

        return

    def predict_with_existing(self):
        self.fc_nn = self.load_best_model()
        self.on_train_end()

    def save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.fc_nn, f_)
        return

    def load_best_model(self):
        with open(self.model_weight_file, "rb") as f_:
            self.fc_nn = torch.load(f_)
        return self.fc_nn

    def load_model(self, file_path):
        with open(file_path, "rb") as f_:
            self.fc_nn = torch.load(f_)
        return self.fc_nn

    def get_best_pr(self):
        return self.best_pr