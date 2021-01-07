import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import collections
from itertools import repeat
from net.utils.graph import Graph


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def conv_dw(inp, oup, kernel_size, stride, padding, dilation, bias):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, dilation=dilation, bias=bias),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class SPGConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, A=None, bias=False):
        super().__init__()

        A_size = A.shape
        self.register_buffer('A', A)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.dw_gcn_weight = nn.Parameter(torch.Tensor(in_channels, A_size[1], A_size[2]))
        self.pw_gcn_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.dw_gcn_weight.data.uniform_(-stdv, stdv)
        self.pw_gcn_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        dw_gcn_weight = self.dw_gcn_weight.mul(self.A)
        x = torch.einsum('nctv,cvw->nctw', (x, dw_gcn_weight))
        x = torch.einsum('nctw,cd->ndtw', (x, self.pw_gcn_weight))
        return x


class TemporalConv(nn.Module):
    def __init__(self, out_channels, t_kernel_size, stride, padding, dilation, bias, dropout):
        super().__init__()

        layers = []
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            conv_dw(out_channels,
                    out_channels,
                    (t_kernel_size, 1),
                    (stride, 1),
                    padding,
                    dilation=(dilation, 1),
                    bias=bias,)
        )
        if dropout:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class STCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, dropout=0, residual=True, bias=False, A=None):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (((kernel_size[0] - 1) * dilation) // 2, 0)

        self.gcn = SPGConv(in_channels, out_channels, kernel_size[1], A=A, bias=bias)
        self.dilation = dilation
        self.t_kernel_size = kernel_size[0]
        self.tcn = TemporalConv(out_channels, self.t_kernel_size, stride, padding, dilation, bias, dropout)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                conv_dw(
                    in_channels,
                    out_channels,
                    kernel_size=(self.t_kernel_size, 1),   # init: 1, my: kernel_size
                    stride=(stride, 1),
                    padding=padding,
                    dilation=(self.dilation, 1),
                    bias=bias
                )
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res

        return self.relu(x)


class SPGNet(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, in_plane=16, dilation=1, topology='physical', **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        if topology == 'complete':
            A = torch.ones(self.graph.A.shape)
        elif topology == 'complement':
            graph_path = 'complement_graph_1.npz'
            print('load graph_path: %s' % graph_path)
            seleted_graph = np.load(graph_path)
            A = seleted_graph['na']
            A = torch.tensor(A, dtype=torch.float32, requires_grad=False).unsqueeze(0)
            print('complement graph.')
        else:
            print('physical graph.')

        self.register_buffer('A', A)
        # graph_size = A.shape

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.in_plane = in_plane
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        # self.motion_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.skeleton_net = nn.Sequential(
            STCBlock(in_channels, self.in_plane, kernel_size, 2, dilation, A=A, residual=False, **kwargs),
            STCBlock(self.in_plane, self.in_plane, kernel_size, 2, dilation, A=A, **kwargs),
            STCBlock(self.in_plane, self.in_plane * 2 ** 1, kernel_size, 2, dilation, A=A, **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A, **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 2, kernel_size, 2, dilation, A=A, **kwargs),
            STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A, **kwargs)
        )

        # fcn for prediction
        self.fcn = nn.Conv2d(self.in_plane * 2 ** 2, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        x = self.skeleton_net(x)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])

        # multi-person
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class SPGNetFusion(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, in_plane=16, dilation=1, topology='physical', **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        if topology == 'complete':
            A = torch.ones(self.graph.A.shape)
        elif topology == 'complement':
            graph_path = 'complement_graph_1.npz'
            print('load graph_path: %s' % graph_path)
            seleted_graph = np.load(graph_path)
            A = seleted_graph['na']
            A = torch.tensor(A, dtype=torch.float32, requires_grad=False).unsqueeze(0)
            print('complement graph.')
        else:
            print('physical graph.')

        self.register_buffer('A', A)
        # graph_size = A.shape

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # config graph
        A_config = [A, A, A, A, A, A]

        self.in_plane = in_plane
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.motion_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.skeleton_net = nn.Sequential(
            STCBlock(in_channels, self.in_plane, kernel_size, 2, dilation, A=A_config[0], residual=False, **kwargs),
            STCBlock(self.in_plane, self.in_plane, kernel_size, 2, dilation, A=A_config[1], **kwargs),
            STCBlock(self.in_plane, self.in_plane * 2 ** 1, kernel_size, 2, dilation, A=A_config[2], **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], **kwargs),
        )
        self.motion_net = nn.Sequential(
            STCBlock(in_channels, self.in_plane, kernel_size, 2, dilation, A=A_config[0], residual=False, **kwargs),
            STCBlock(self.in_plane, self.in_plane, kernel_size, 2, dilation, A=A_config[1], **kwargs),
            STCBlock(self.in_plane, self.in_plane * 2 ** 1, kernel_size, 2, dilation, A=A_config[2], **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], **kwargs),
        )
        self.fusion = nn.Sequential(
            STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 2, dilation, A=A_config[4], **kwargs),
            STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A_config[5], **kwargs)
        )

        # fcn for prediction
        self.fcn = nn.Conv2d(self.in_plane * 2 ** 2, num_class, kernel_size=1)

    def forward(self, x):
        motion = x[1]
        x = x[0]

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # motion nornalization
        motion = motion.permute(0, 4, 3, 1, 2).contiguous()
        motion = motion.view(N * M, V * C, T)
        motion = self.motion_bn(motion)
        motion = motion.view(N, M, V, C, T)
        motion = motion.permute(0, 1, 3, 4, 2).contiguous()
        motion = motion.view(N * M, C, T, V)

        # forward
        x = torch.cat([self.skeleton_net(x), self.motion_net(motion)], 1).contiguous()
        x = self.fusion(x)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])

        # multi-person
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, in_plane, dilation, topology, **kwargs):
        super().__init__()
        self.position_net = SPGNet(in_channels, num_class, graph_args, in_plane, dilation, topology, **kwargs)

    def forward(self, x):
        position = x[0]
        out = self.position_net(position)
        return out


class ModelFusion(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, in_plane, dilation, topology, **kwargs):
        super().__init__()
        self.fusion_net = SPGNetFusion(in_channels, num_class, graph_args, in_plane, dilation, topology, **kwargs)

    def forward(self, x):
        return self.fusion_net(x)
