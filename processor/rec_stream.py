#!/usr/bin/env python
# pylint: disable=W0201

# import sys
import argparse
# import yaml
import time
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
from torchlight import str2bool
from .processor import Processor


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

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
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
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, motion, label in tqdm(loader):
            self.global_step += 1
            # get data
            data = data.float().to(self.dev)
            motion = motion.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model([data, motion])
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info(print_flag=False)
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('train_loss', self.epoch_info['train_mean_loss'], self.global_step)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, epoch, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        infer_time = data_time = device_time = log_time = 0.0
        t4 = time.time()
        t1 = t2 = t3 = 0.0
        for data, motion, label in tqdm(loader):
            torch.cuda.synchronize()
            t1 = time.time()
            data_time += t1 - t4

            # get data
            data = data.float().to(self.dev)
            motion = motion.float().to(self.dev)
            label = label.long().to(self.dev)
            torch.cuda.synchronize()
            t2 = time.time()
            device_time += t2 - t1

            # inference
            with torch.no_grad():
                output = self.model([data, motion])
                # output = self.model([data, []])
            torch.cuda.synchronize()
            t3 = time.time()
            infer_time += t3 - t2

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
            torch.cuda.synchronize()
            t4 = time.time()
            log_time += t4 - t3

        print('>>> data_time: %.2fs, device_time: %.2fs, infer_time: %.2fs, log_time: %.2fs' % (
            data_time, device_time, infer_time, log_time))
        self.result = np.concatenate(result_frag)

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.eval_info['eval_mean_loss'] = np.mean(loss_value)
            self.show_eval_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
            self.show_best(1)
            self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
