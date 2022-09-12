import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from nets.encoders import video_encoder
#from nets.CNN_GCN import kpts_decoder, SpiralConv
from nets.EFNet import MLP
#from nets.EFGCN import EFGCN

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1, tsteps=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * (self.seq_length), out_channels)
        #print(in_channels * (self.seq_length), out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):

        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
            x = self.layer(x)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
            x = self.layer(x)

        else:
            raise RuntimeError(
                'x.dim() is expected to be 2, 3 and 4, but received {}'.format(
                    x.dim()))
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

# class kpts_decoder_temporal(nn.Module):       # TODO: use inheritance.
#
#     def __init__(self, features_size, kpt_channels, gcn_channels, num_kpts=8, tsteps=1, is_gpu=True):
#         super().__init__(features_size, kpt_channels, gcn_channels, num_kpts, tsteps, is_gpu)
#
#     def create_graph(self, num_nodes: int, tsteps: int) -> np.ndarray:
#         # construct nodes for graph CNN decoder:
#         self.num_nodes_single = int(num_nodes / tsteps)
#         adjacency = []
#
#         for ii in range(self.num_nodes_single):
#             xx = list(range(self.num_nodes_single))
#             xx.insert(0, xx.pop(ii))
#             adjacency.append(xx)
#         adjacency = np.array(adjacency)
#
#         # if more than one time step is present
#         if self.tsteps != 1:
#             indices = np.concatenate((adjacency, adjacency + self.num_nodes_single), 0)
#             time_edges = np.roll(indices[:, 0], self.num_nodes_single)
#             adjacency = np.concatenate((indices, time_edges[:,np.newaxis]),1)
#             # note that this is a special case for two frames.
#             # more than to frames requires:
#             # handling for end and start frames versus intermediate frames
#             # quick fix would be to add a temporal connection to itself in case of the endpoints
#             # otherwise the SpiralConv must be changed, too
#             # currently only implemented for two timesteps because more than 2 most be treated differently
#
#         return adjacency


class kpts_decoder_temporal(nn.Module):

    def __init__(self, features_size, kpt_channels, gcn_channels, num_kpts=8, tsteps=1, is_gpu=True):

        super(kpts_decoder_temporal, self).__init__()

        self.kpt_channels = kpt_channels
        self.gcn_channels = gcn_channels

        # construct nodes for graph CNN decoder:
        self.num_nodes = num_kpts
        self.tsteps = tsteps
        self.num_nodes_single = int(self.num_nodes / self.tsteps)
        adjacency = []

        for ii in range(self.num_nodes_single):
            xx = list(range(self.num_nodes_single))
            xx.insert(0, xx.pop(ii))
            adjacency.append(xx)
        adjacency = np.array(adjacency)

        # if more than one time step is present
        if self.tsteps != 1:
            indices = np.concatenate((adjacency, adjacency + self.num_nodes_single), 0)
            time_edges = np.roll(indices[:, 0], self.num_nodes_single)
            adjacency = np.concatenate((indices, time_edges[:,np.newaxis]),1)
            # note that this is a special case for two frames.
            # more than to frames requires:
            # handling for end and start frames versus intermediate frames
            # quick fix would be to add a temporal connection to itself in case of the endpoints
            # otherwise the SpiralConv must be changed, too
            # currently only implemented for two timesteps because more than 2 most be treated differently

        self.spiral_indices = torch.from_numpy(adjacency)
        if not is_gpu:
            self.spiral_indices = self.spiral_indices.cpu()
        else:
            self.spiral_indices = self.spiral_indices.cuda()

        # construct graph CNN layers:
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(features_size, self.num_nodes * self.gcn_channels[-1]))
        for idx in range(len(self.gcn_channels)):
            if idx == 0:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx - 1],
                               self.gcn_channels[-idx - 1],
                               self.spiral_indices, tsteps=tsteps))
            else:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx], self.gcn_channels[-idx - 1],
                               self.spiral_indices, tsteps=tsteps))
        self.decoder_layers.append(
            SpiralConv(self.gcn_channels[0], self.kpt_channels, self.spiral_indices, tsteps=tsteps))

    def forward(self, x):
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_nodes, self.gcn_channels[-1])
            elif i != num_layers - 1:
                x = F.elu(layer(x))
            else:
                x = layer(x)
        return x


class EFGCNTmp(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """
    def __init__(self, backbone: str = 'r3d_18', MLP_channels_ef: List = [16, 32, 32, 48], GCN_channels_kpts: List = [16, 32, 32, 48], is_gpu=False):

        super(EFGCNTmp, self).__init__()

        self.MLP_channels_ef = MLP_channels_ef
        self.GCN_channels_kpts = GCN_channels_kpts

        self.encoder = video_encoder(backbone=backbone)
        self.ef_regressor = MLP(features_size=self.encoder.img_feature_size, channels=self.MLP_channels_ef, output_channels=1)
        self.kpts_regressor = kpts_decoder_temporal(features_size=self.encoder.img_feature_size,
                                         kpt_channels=2,
                                         gcn_channels=self.GCN_channels_kpts,
                                         tsteps=2,
                                         num_kpts=40 * 2,
                                         is_gpu=is_gpu)
        # This is a sparser graph, including full spatial connections for each frame, and temporal connections for each associated node.

    def forward(self, x: torch.Tensor, is_feat: bool = False) -> torch.Tensor:
        features = self.encoder(x)
        ef, guiding_layer_ef = self.ef_regressor(features)
        kpts = self.kpts_regressor(features)

        if is_feat:
            return ef, kpts, guiding_layer_ef
        else:
            return ef, kpts


if __name__ == '__main__':
    frame_size = 112
    num_frames = 2
    num_kpts = 40
    num_batches = 4
    kpt_channels = 2 # kpts dim
    print('load model')
    img = torch.rand(num_batches, 3, num_frames, frame_size, frame_size).cuda()
    m = EFGCNTmp(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 16, 32, 32], is_gpu=True)
    m = m.cuda()
    print('model loaded')
    ef_pred, kpts_pred = m(img)
    ef, kpts = torch.rand(num_batches).cuda(), torch.rand(num_batches, num_kpts, num_frames).cuda()
    # loss = nn.L1Loss()
    # print(loss(ef_pred, ef) + loss(kpts_pred, kpts))
    # optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    # optimizer.step()
    print("hi")
    pass
