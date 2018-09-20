from __future__ import print_function
from __future__ import absolute_import

import argparse
import pandas as pd
import numpy as np
import json
import math
import sys
import os
import time
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as auto
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from pcba_model import *


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())


def variable_to_tensor(x):
    return x.data


def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x


def load_data(conf):
    task_list = conf['task_list']
    data_directory = given_args.data_dir
    k = 5
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv.gz'.format(i))

    data_file_list = [data_directory + 'file_{}.csv.gz'.format(i) for i in range(5)]

    train_dataset = PCBADataset(data_files=data_file_list[0:3],
                                feature_name='Fingerprints',
                                task_list=task_list,
                                transform=ToTensor())
    val_dataset = PCBADataset(data_files=data_file_list[3:4],
                              feature_name='Fingerprints',
                              task_list=task_list,
                              transform=ToTensor())
    test_dataset = PCBADataset(data_files=data_file_list[4:5],
                               feature_name='Fingerprints',
                               task_list=task_list,
                               transform=ToTensor())
    print('Done Data Preparation!')
    return train_dataset, val_dataset, test_dataset


def train_and_test_MTL(dp_classifier, max_epoch, training_dataset, **kwargs):
    train_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                   batch_size=dp_classifier.fit_batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    enable_gpu = kwargs['enable_gpu']
    dp_classifier.fc_nn.eval()

    N = len(training_dataset)
    dp_classifier.on_train_begin()
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        dp_classifier.on_epoch_begin()
        training_loss_value = 0
        for i, (X_batch, y_batch, sample_weight_batch) in enumerate(train_dataloader):
            if enable_gpu:
                X_batch = Variable(X_batch.float().cuda())
                y_batch = Variable(y_batch.float().cuda())
                sample_weight_batch = Variable(sample_weight_batch.float().cuda())
            else:
                X_batch = Variable(X_batch.float())
                y_batch = Variable(y_batch.float())
                sample_weight_batch = Variable(sample_weight_batch.float())

            dp_classifier.optimizer.zero_grad()
            y_pred = dp_classifier.fc_nn(X_batch)
            loss = dp_classifier.multi_task_cost(y_pred, y_batch, sample_weight_batch)
            training_loss_value += loss.data[0]
            # backward prop
            loss.backward()
            # update the model
            dp_classifier.optimizer.step()

            logging.debug('{}/{}: Loss is {:.6f}'.format(i, N/dp_classifier.fit_batch_size, loss.data[0]))
        avg_loss = training_loss_value / (1.0*N/dp_classifier.fit_batch_size)
        dp_classifier.on_epoch_end(avg_loss)
        if dp_classifier.stop_training:
            break
    dp_classifier.on_train_end()
    return


def train_and_test_RMTL(dp_classifier, max_epoch, training_dataset, **kwargs):
    train_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                   batch_size=dp_classifier.fit_batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    enable_gpu = kwargs['enable_gpu']
    task_list = kwargs['task_list']
    focused_task = kwargs['focused_task']
    if focused_task not in task_list:
        raise ValueError('Focused Cosine task ({}) not in task list ({}).'.format(focused_task, task_list))
    print('Focused Cosine task ({}) in task list ({}).'.format(focused_task, task_list))
    dp_classifier.fc_nn.eval()
    final_layer_names = ['task_{}'.format(i) for i in range(len(task_list))]

    N = len(training_dataset)
    dp_classifier.on_train_begin()
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        dp_classifier.on_epoch_begin()
        training_loss_value = 0
        for i, (X_batch, y_batch, sample_weight_batch) in enumerate(train_dataloader):
            if enable_gpu:
                X_batch = Variable(X_batch.float().cuda())
                y_batch = Variable(y_batch.float().cuda())
                sample_weight_batch = Variable(sample_weight_batch.float().cuda())
            else:
                X_batch = Variable(X_batch.float())
                y_batch = Variable(y_batch.float())
                sample_weight_batch = Variable(sample_weight_batch.float())

            dp_classifier.optimizer.zero_grad()
            y_pred = dp_classifier.fc_nn(X_batch)
            loss_list = dp_classifier.multi_task_cost(y_pred, y_batch, sample_weight_batch, reduce=False)

            class_weights = np.ones((len(task_list)))
            for i, layer_name in enumerate(final_layer_names):
                layer = getattr(dp_classifier.fc_nn, layer_name)
                logging.debug(layer, '\t', layer.weight.size())
                grad = auto.grad(loss_list[i], layer.weight, retain_graph=True)[0]
                if i == 0:
                    focused_grad = grad
                else:
                    cosine_score = F.cosine_similarity(focused_grad, grad)
                    class_weights[i] = max(0, cosine_score.data[0])
            print('updated class weight:\t{}'.format(class_weights))
            class_weights = tensor_to_variable(torch.from_numpy(class_weights))
            logging.info('loss is: {}'.format(loss_list))
            loss_list = torch.mul(loss_list, class_weights)
            logging.info('loss is: {}'.format(loss_list))

            loss = loss_list.sum()
            training_loss_value += loss.data[0]
            # backward prop
            loss.backward(retain_graph=True)
            # update the model
            dp_classifier.optimizer.step()

            logging.debug('{}/{}: Loss is {:.6f}'.format(i, N/dp_classifier.fit_batch_size, loss.data[0]))
        avg_loss = training_loss_value / (1.0*N/dp_classifier.fit_batch_size)
        dp_classifier.on_epoch_end(avg_loss)
        if dp_classifier.stop_training:
            break
    dp_classifier.on_train_end()
    return


def train_and_test_LBTW(dp_classifier, max_epoch, training_dataset, **kwargs):
    train_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                   batch_size=dp_classifier.fit_batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    enable_gpu = kwargs['enable_gpu']
    task_list = kwargs['task_list']
    alpha = kwargs['alpha']
    dp_classifier.fc_nn.eval()

    N = len(training_dataset)
    dp_classifier.on_train_begin()
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        dp_classifier.on_epoch_begin()
        training_loss_value = 0
        for i, (X_batch, y_batch, sample_weight_batch) in enumerate(train_dataloader):
            if enable_gpu:
                X_batch = Variable(X_batch.float().cuda())
                y_batch = Variable(y_batch.float().cuda())
                sample_weight_batch = Variable(sample_weight_batch.float().cuda())
            else:
                X_batch = Variable(X_batch.float())
                y_batch = Variable(y_batch.float())
                sample_weight_batch = Variable(sample_weight_batch.float())

            dp_classifier.optimizer.zero_grad()
            y_pred = dp_classifier.fc_nn(X_batch)
            loss_list = dp_classifier.multi_task_cost(y_pred, y_batch, sample_weight_batch, reduce=False)

            if i == 0:
                initial_task_loss_list = loss_list.data
                logging.info('initial loss\t{}'.format(initial_task_loss_list))

            loss_ratio = loss_list.data / initial_task_loss_list
            inverse_traing_rate = loss_ratio
            class_weights = inverse_traing_rate.pow(alpha)
            class_weights = class_weights / sum(class_weights) * len(task_list)
            class_weights = Variable(class_weights, requires_grad=False)
            logging.info('class weight is: {}'.format(class_weights))
            logging.info('loss is: {}'.format(loss_list))
            loss_list = torch.mul(loss_list, class_weights)
            logging.debug('loss is: {}'.format(loss_list))

            loss = loss_list.sum()
            training_loss_value += loss.data[0]
            # backward prop
            loss.backward(retain_graph=True)
            # update the model
            dp_classifier.optimizer.step()

            logging.debug('{}/{}: Loss is {:.6f}'.format(i, N/dp_classifier.fit_batch_size, loss.data[0]))
        avg_loss = training_loss_value / (1.0*N/dp_classifier.fit_batch_size)
        dp_classifier.on_epoch_end(avg_loss)
        if dp_classifier.stop_training:
            break
    dp_classifier.on_train_end()
    return


def train_and_test_FineTuning(dp_classifier, max_epoch, training_dataset, **kwargs):
    if kwargs['pre_train']:
        print('Pre Training on {}'.format(kwargs['task_list']))
        train_and_test_MTL(dp_classifier, max_epoch, training_dataset, **kwargs)
    else:
        print('Fine Tuning on {}'.format(kwargs['task_list']))
        pre_train_model_weight_file = dp_classifier.model_weight_file.replace('fine_tune', 'pre_train')
        pre_trained_model = torch.load(pre_train_model_weight_file)

        model_dict = dp_classifier.fc_nn.state_dict()
        for name, param in pre_trained_model.named_parameters():
            if 'task' not in name:
                model_dict[name] = param
        dp_classifier.fc_nn.load_state_dict(model_dict)

        train_and_test_MTL(dp_classifier, max_epoch, training_dataset, **kwargs)

    return


def main(args):
    config_json_file = args.config_json_file
    with open(config_json_file, 'r') as f:
        conf = json.load(f)

    task_list = args.task_list
    task_list = map(lambda x: 'pcba-aid{}'.format(x), task_list.strip().split(' '))

    if args.mode in ['RMTL', 'FineTuning']:
        focused_task = 'pcba-aid'+ args.focused_task
        if args.mode == 'FineTuning':
            if args.pre_train:
                task_list.remove(focused_task)
            else:
                task_list = [focused_task]
        elif args.mode == 'RMTL':
            assert focused_task is not None, 'RMTL Task shouldn\'t be None'
            task_list.remove(focused_task)
            task_list = [focused_task] + task_list

    conf['task_list'] = task_list
    print('task_list ', task_list)

    train_dataset, val_dataset, test_dataset = load_data(conf)

    kwargs = {'file_path': args.model_weight_file, 'score_path': args.score_path,
              'enable_gpu': args.enable_gpu, 'seed': args.seed,
              'training_dataset': train_dataset, 'validation_dataset':val_dataset, 'test_dataset':test_dataset}
    dp_classifier = MultiTaskModel(conf=conf, **kwargs)
    dp_classifier.build_model()

    if args.enable_gpu:
        dp_classifier.fc_nn.cuda()

    if args.mode == 'MTL':
        print('Running Multi-Task Leaning.')
        kwargs = {'score_path': args.score_path, 'task_list': task_list,
                  'enable_gpu': args.enable_gpu}
        train_and_test_MTL(dp_classifier, max_epoch=conf['fitting']['nb_epoch'],
                           training_dataset=train_dataset, **kwargs)
    elif args.mode == 'RMTL':
        print('Running Reinforced Multi-Task Learning.')
        kwargs = {'score_path': args.score_path, 'task_list': task_list, 'focused_task': focused_task,
                  'enable_gpu': args.enable_gpu}
        train_and_test_RMTL(dp_classifier, max_epoch=conf['fitting']['nb_epoch'],
                               training_dataset=train_dataset, **kwargs)
    elif args.mode == 'LBTW':
        print('Running Loss-Balanced Task Weighting.')
        kwargs = {'score_path': args.score_path, 'task_list': task_list,
                  'alpha': conf['alpha'], 'enable_gpu': args.enable_gpu}
        train_and_test_LBTW(dp_classifier, max_epoch=conf['fitting']['nb_epoch'],
                            training_dataset=train_dataset, **kwargs)
    elif args.mode == 'FineTuning':
        print('Running FineTuning.')
        kwargs = {'score_path': args.score_path, 'task_list': task_list,
                  'enable_gpu': args.enable_gpu, 'pre_train': args.pre_train}
        train_and_test_FineTuning(dp_classifier, max_epoch=conf['fitting']['nb_epoch'],
                                  training_dataset=train_dataset, **kwargs)
    else:
        raise ValueError('No such mode. Must be in [{}, {}, {}, {}].'.
                         format('MTL', 'RMTL', 'LBTW', 'FineTuning'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_list', action='store', dest='task_list', required=True)
    parser.add_argument('--config_json_file', action='store', dest='config_json_file', required=True)
    parser.add_argument('--model_weight_file', action='store', dest='model_weight_file', required=True)
    parser.add_argument('--score_path', action='store', dest='score_path', required=True)
    parser.add_argument('--mode', action='store', dest='mode', required=True)
    parser.add_argument('--enable-gpu', action='store_true', dest='enable_gpu')
    parser.add_argument('--data_dir', action='store', dest='data_dir',  required=False, default='../dataset/pcba/')
    parser.set_defaults(enable_gpu=False)
    parser.add_argument('--seed', action='store', dest='seed', default=123, required=False)

    parser.add_argument('--focused_task', action='store', dest='focused_task', required=False)
    parser.add_argument('--pre-train', action='store_true', dest='pre_train')
    parser.set_defaults(enable_gpu=False)
    given_args = parser.parse_args()
    main(given_args)