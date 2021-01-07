#!/usr/bin/env python
# pylint: disable=W0201
# import sys
import argparse
# import yaml
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
from torchlight import str2bool
from .processor_ensemble import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, argv=None):
        Processor.__init__(self, argv)
        if self.arg.model2:
            print('init model 2')
            if not self.arg.model_args2:
                self.arg.model_args2 = self.arg.model_args
            self.model2 = self.io.load_model(self.arg.model2, **(self.arg.model_args2))
            if self.arg.weights2:
                self.model2 = self.io.load_weights(self.model2, self.arg.weights2, self.arg.ignore_weights)
            self.model2 = self.model2.to(self.dev)
            for name, value in vars(self).items():
                cls_name = str(value.__class__)
                if cls_name.find('torch.nn.modules') != -1:
                    setattr(self, name, value.to(self.dev))

            # model parallel
            if self.arg.use_gpu and len(self.gpus) > 1:
                self.model2 = nn.DataParallel(self.model2, device_ids=self.gpus)

        if self.arg.model3:
            print('init model 3')
            if not self.arg.model_args3:
                self.arg.model_args3 = self.arg.model_args
            self.model3 = self.io.load_model(self.arg.model3, **(self.arg.model_args3))
            if self.arg.weights3:
                self.model3 = self.io.load_weights(self.model3, self.arg.weights3, self.arg.ignore_weights)
            self.model3 = self.model3.to(self.dev)
            for name, value in vars(self).items():
                cls_name = str(value.__class__)
                if cls_name.find('torch.nn.modules') != -1:
                    setattr(self, name, value.to(self.dev))

            # model parallel
            if self.arg.use_gpu and len(self.gpus) > 1:
                self.model3 = nn.DataParallel(self.model3, device_ids=self.gpus)

        if self.arg.model4:
            print('init model 4')
            if not self.arg.model_args4:
                self.arg.model_args4 = self.arg.model_args
            self.model4 = self.io.load_model(self.arg.model4, **(self.arg.model_args4))
            if self.arg.weights4:
                self.model4 = self.io.load_weights(self.model4, self.arg.weights4, self.arg.ignore_weights)
            self.model4 = self.model4.to(self.dev)
            for name, value in vars(self).items():
                cls_name = str(value.__class__)
                if cls_name.find('torch.nn.modules') != -1:
                    setattr(self, name, value.to(self.dev))

            # model parallel
            if self.arg.use_gpu and len(self.gpus) > 1:
                self.model4 = nn.DataParallel(self.model4, device_ids=self.gpus)

        self.load_optimizer()

    def param_optimized(self):
        print('check parameters ...')
        params_list = []
        if not self.arg.residual:
            params_list = [{'params': self.model.parameters()}]
        if self.arg.model2:
            params_list.append({'params': self.model2.parameters()})
        if self.arg.model3:
            params_list.append({'params': self.model3.parameters()})
        if self.arg.model4:
            params_list.append({'params': self.model4.parameters()})
        return params_list

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                # self.model.parameters(),
                self.param_optimized(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self):
        if self.arg.residual:
            self.model.eval()
        else:
            self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, motion, label in tqdm(loader):

            # get data
            data = data.float().to(self.dev)
            motion = motion.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            if self.arg.residual:
                # with torch.no_grad():
                output = self.model([data, motion])
            else:
                output = self.model([data, motion])
            if self.arg.model2:
                output += self.model2([data, motion])
            if self.arg.model3:
                output += self.model3([data, motion])
            if self.arg.model4:
                output += self.model4([data, motion])
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            print(np.mean(loss_value))
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        if self.arg.model2:
            self.model2.eval()
        if self.arg.model3:
            self.model3.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, motion, label in tqdm(loader):
            # get data
            data = data.float().to(self.dev)
            motion = motion.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model([data, motion])
                if self.arg.model2:
                    output += self.model2([data, motion])
                if self.arg.model3:
                    output += self.model3([data, motion])

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.eval_info['eval_mean_loss'] = np.mean(loss_value)
            self.show_eval_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
            self.show_best(1)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='MV-IGNet')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--residual',  type=str2bool, default=False, help='residual manner or not')
        # endregion yapf: enable

        return parser
