import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN1, LeakyReLU as LReLU, Module as Mod
import torch.nn.functional as F
import math
from collections import OrderedDict


def check_dataparallel(state_dict):
    res = False
    for k, _ in state_dict.items():
        if str(k).startswith('module'):
            res = True
            break
    return res


def remove_prefix_dict(state_dict, to_remove_str='module.'):
    new_state_dict = OrderedDict()
    remove_len = len(to_remove_str)
    for k, v in state_dict.items():
        if str(k).startswith(to_remove_str):
            name = k[remove_len:]  # remove to_remove_str
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    print(f'Refactored dict keys: {new_state_dict.keys()}')
    return new_state_dict



def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9:
            # normals not considered in graph construction
            idx = knn(x[:, 6:], k=k)
        else:
            idx = knn(x, k=k)  # (batch_size, in_points, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature  # (batch_size, 2*num_dims, in_points, k)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def batched_index_select(x, dim, index):
    for i in range(1, len(x.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)


class EdgePoolingLayer(Mod):
    """ Dynamic Edge Pooling layer - Relued Scores before TopK"""

    def __init__(self, in_channels, k, ratio=0.5, scoring_fun="tanh", num_points=-1):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.ratio = ratio
        self.score_layer = nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True)
        self.scoring_fun = scoring_fun
        self.num_points = num_points

    def __str__(self):
        return 'EdgePoolingLayer(in_channels=' + str(self.in_channels) + ', k=' + str(self.k) + ', ratio=' + str(
            self.ratio) + ', scoring_fun=' + str(self.scoring_fun) + ', num_points=' + str(self.num_points) + ')'

    def forward(self, feat, idx=None):
        batch_size, dim, in_points = feat.size()  # (batch_size, α, in_points)
        assert dim == self.in_channels

        # Dynamic Edge Conv
        # Re-computing graph before pooling
        x = get_graph_feature(feat, k=self.k, idx=idx)  # (batch_size, α * 2, in_points, self.k)
        x = self.score_layer(x)  # (batch_size, 1, in_points, self.k)

        # scores = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1, in_points)
        # ReLU applied to Scores
        scores = F.relu(x.max(dim=-1, keepdim=False)[0])  # (batch_size, 1, in_points)

        if self.num_points < 0:
            # default - 'num_points' not specified, use 'ratio'
            num_keypoints = math.floor(in_points * self.ratio)
        else:
            # do not use ratio but sample a fixed number of points 'num_points'
            assert self.num_points < in_points, \
                "Pooling more points (%d) than input ones (%d) !" % (self.num_points, in_points)
            num_keypoints = self.num_points

        top, top_idxs = torch.topk(scores.squeeze(), k=num_keypoints, dim=1)
        new_feat = batched_index_select(feat.permute(0, 2, 1), 1, top_idxs)  # (batch, num_keypoints, α)
        top = top.unsqueeze(2)

        # Apply scoring function!
        if self.scoring_fun == 'tanh':
            new_feat = new_feat * torch.tanh(top)
        elif self.scoring_fun == 'softmax':
            new_feat = new_feat * F.softmax(top, dim=1)
        elif self.scoring_fun == 'leaky-relu':
            new_feat = new_feat * F.leaky_relu(top, negative_slope=0.001)

        return new_feat


class GlobalFeat(Mod):
    def __init__(self, k=30, emb_dims=1024):
        super(GlobalFeat, self).__init__()
        self.k = k
        self.conv1 = Seq(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = Seq(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = Seq(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                         nn.BatchNorm2d(128),
                         nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = Seq(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                         nn.BatchNorm2d(256),
                         nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = Seq(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                         nn.BatchNorm1d(emb_dims),
                         nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, repeat=False):
        B, dim, N = x.size()
        assert dim == 3

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        feat = self.conv5(x)
        x1 = F.adaptive_max_pool1d(feat, 1).view(B, -1)

        # x2 = F.adaptive_avg_pool1d(feat, 1).view(B, -1)
        # x = torch.cat((x1, x2), 1)

        if repeat:
            return x1.unsqueeze(-1).repeat_interleave(N, 2)
        else:
            return x1


class Projector(Mod):
    def __init__(self, in_feat=1024, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_feat, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim, bias=True)
        )

    def forward(self, x):
        return self.projector(x)


###################################

class LocalFeat(Mod):
    def __init__(self, k=6, emb_dims=1024):
        super(LocalFeat, self).__init__()
        self.k = k

        self.conv1 = Seq(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Conv2d(64, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2), )

        self.conv2 = Seq(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Conv2d(64, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2), )

        self.conv3 = Seq(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Conv2d(64, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2), )

        self.conv4 = Seq(nn.Conv1d(64 * 3, emb_dims, kernel_size=1, bias=False),
                         nn.BatchNorm1d(emb_dims),
                         nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, in_points)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, in_points)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, in_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64 * 3, in_points)
        x = self.conv4(x)  # (batch_size, emb_dims, in_points)
        return x


class Classifier(Mod):
    def __init__(self, k, emb_dims, out_channels=13):
        super().__init__()

        # Global GLEncoder
        self.global_encoder = GlobalFeat(k=k, emb_dims=emb_dims)

        # Classification Head
        self.cls_mlp = Seq(
            self.mlp([emb_dims, 512]),
            nn.Dropout(0.5),
            self.mlp([512, 256]),
            nn.Dropout(0.5),
            Lin(256, out_channels))

        self._initialize_weights()

    def mlp(self, channels):
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]),
                BN1(channels[i]),
                LReLU(negative_slope=0.2),
                ) for i in range(1, len(channels))
        ])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # very similar to resnet
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pos):
        feat = self.global_encoder(pos)
        x = self.cls_mlp(feat)
        return x
