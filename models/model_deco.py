"""
@Author: Antonio Alliegro
@Contact: antonio.alliegro@polito.it
@File: model_deco.py
@Time: July 14th - going on
"""

from __future__ import print_function
from models.GPDNet import *
from model_utils import *


class GPDFeat(nn.Module):
    """
    Our main local encoder branch
    https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650103.pdf
    """
    def __init__(self, configuration):
        super(GPDFeat, self).__init__()

        # PRE
        self.pre_nn_layers = get_mlp_1d([3, 33, 66, 99])

        # RESIDUAL BLOCKS
        self.residual_block1 = Seq()
        for j in range(configuration["conv_n_layers"]):
            self.residual_block1.add_module(name="conv_{}_layer".format(j),
                                            module=ConvLayer(layer_conf=configuration["conv_layer"]))
        self.residual_block2 = Seq()
        for j in range(configuration["conv_n_layers"]):
            self.residual_block2.add_module(name="conv_{}_layer".format(j),
                                            module=ConvLayer(layer_conf=configuration["conv_layer"]))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # very similar to resnet
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, points):
        assert points.size(1) == 3, "Wrong input shape at GPDFeat"
        # pre
        h = self.pre_nn_layers(points)  # [B, in_feat, N]
        h = h.permute(0, 2, 1)

        # residual block 1
        x1, _ = self.residual_block1((h, None))  # compute graph 1, [B, N, in_feat] -> [B, N, out_feat]
        h = x1 + h  # residual 1

        # residual block 2
        x2, _ = self.residual_block2((h, None))  # computing graph 2
        h = x2 + h  # residual 2
        return h.permute(0, 2, 1)


class GLEncoder(Mod):
    """
    Global + Local encoder (DeCo) used throughout all paper experiments
    """
    def __init__(self, conf):
        super(GLEncoder, self).__init__()

        self.global_encoder = GlobalFeat(
            k=conf['global_encoder']['nearest_neighboors'],
            emb_dims=conf['global_encoder']['latent_dim'])

        self.local_encoder = GPDFeat(
            configuration=conf['GPD_local'])

        global_emb_dim = conf['global_encoder']['latent_dim']
        out_dim = conf['generator']['latent_dim']

        self.nn_local = nn.Conv1d(
            in_channels=99,
            out_channels=out_dim,
            kernel_size=1,
            bias=False)

        self.nn_global = nn.Linear(
            in_features=global_emb_dim,
            out_features=out_dim,
            bias=False)

        self.nn_aggr_bn = nn.BatchNorm1d(out_dim)
        self.nn_aggr_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, points):
        l_feat = self.local_encoder(points)  # [B, 99, N]
        g_feat = self.global_encoder(points)  # [B, global_emb]

        l_feat = self.nn_local(l_feat)  # [B,out_dim,N]
        g_feat = self.nn_global(g_feat).unsqueeze(-1)  # [B,out_dim,1]

        feat = l_feat + g_feat  # local + global aggregation
        feat = self.nn_aggr_bn(feat)
        feat = self.nn_aggr_relu(feat)
        return feat


class Generator(Mod):
    def __init__(self, conf, pool1_points, pool2_points):
        super(Generator, self).__init__()
        self.k = conf['generator']['nearest_neighboors']
        self.emb_dim = conf['generator']['latent_dim']
        self.scoring_fun = conf['generator']['scoring_fun']
        self.pool1_nn = conf['generator']['pool1_nn']
        self.pool2_nn = conf['generator']['pool2_nn']

        self.conv1 = Seq(
            nn.Conv2d(self.emb_dim * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2))

        self.pool1 = EdgePoolingLayer(
            in_channels=128, k=self.pool1_nn,
            ratio=0.5, scoring_fun=self.scoring_fun,
            num_points=pool1_points)

        self.conv2 = Seq(
            nn.Conv2d(128 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = Seq(
            nn.Conv2d(128 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.pool2 = EdgePoolingLayer(
            in_channels=64, k=self.pool2_nn,
            ratio=0.5, scoring_fun=self.scoring_fun,
            num_points=pool2_points)

        # Intermediate Head
        self.raw_head = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=True))

        # Final Head
        self.fine_head = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=True))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = self.pool1(x1).permute(0, 2, 1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # [B, 128, pool1_points]

        # Intermediate Pred: frame + coarse missing part
        x_raw = self.raw_head(x2)  # [B, 128, pool1_points] => [B, 3, pool1_points]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # [B, 64, pool1_points]
        x = self.pool2(x3).permute(0, 2, 1)

        # Final Pred: missing part
        x = self.fine_head(x)  # [B, 64, pool2_points] => [B, 3, pool2_points]
        return x.permute(0, 2, 1), x_raw.permute(0, 2, 1)


class LocalFeat_dgcnn(Mod):
    """
    DGCNN local feature extractor: returns 64*3 per-point feature embedding
    For 'Local Denoising Variants' (Table 8 arxiv) experiments in which we compared different GCN networks at the local encoder
    """

    def __init__(self, k=8):
        super(LocalFeat_dgcnn, self).__init__()
        self.k = k
        self.conv1 = Seq(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Conv2d(64, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = Seq(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Conv2d(64, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = Seq(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2),
                         nn.Conv2d(64, 64, kernel_size=1, bias=False),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = torch.cat((x1, x2, x3), dim=1)  # (B, 64*3, N)
        return x


class GLEncoder_dgcnn(Mod):
    """
    At the local encoder branch DGCNN instead of GPD.
    For 'Local Denoising Variants' (Table 8 arxiv) experiments in which we compared different GCN networks at the local encoder
    """

    def __init__(self, conf):
        super(GLEncoder_dgcnn, self).__init__()
        self.config = conf

        self.global_encoder = GlobalFeat(k=conf['global_encoder']['nearest_neighboors'],
                                         emb_dims=conf['global_encoder']['latent_dim'])

        self.local_encoder = LocalFeat_dgcnn(k=conf['local_encoder']['nearest_neighboors'])

        global_emb_dim = conf['global_encoder']['latent_dim']
        out_dim = conf['generator']['latent_dim']

        self.nn_local = nn.Conv1d(
            in_channels=64 * 3,
            out_channels=out_dim,
            kernel_size=1,
            bias=False)

        self.nn_global = nn.Linear(
            in_features=global_emb_dim,
            out_features=out_dim,
            bias=False)

        self.nn_aggr_bn = nn.BatchNorm1d(out_dim)
        self.nn_aggr_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, points):
        l_feat = self.local_encoder(points)  # BS, 64*3, n_points
        g_feat = self.global_encoder(points)  # BS, global_emb

        l_feat = self.nn_local(l_feat)  # [BS, out_dim, n_points]
        g_feat = self.nn_global(g_feat).unsqueeze(-1)  # [BS, out_dim, 1]

        feat = l_feat + g_feat  # local + global aggregation
        feat = self.nn_aggr_bn(feat)
        feat = self.nn_aggr_relu(feat)
        return feat
