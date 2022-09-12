import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple

from nets.encoders import image_encoder
from nets.CNN_GCN import kpts_decoder

class Reduce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = x.mean(dim=1)
        return x


class kpts_decoder_temporal(kpts_decoder):

    def __init__(self, features_size, kpt_channels, gcn_channels, num_kpts=8, tsteps=2, is_gpu=True):
        super(kpts_decoder_temporal, self).__init__(features_size, kpt_channels, gcn_channels, num_kpts, tsteps, is_gpu)
        # self.regression_layer = nn.Sequential(nn.Linear(self.gcn_channels[0]*num_kpts,1), nn.Sigmoid())

    def forward(self, x):
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                for idx in range(0, x.size(2)):
                    x1 = layer(x[:, :, idx])
                    x1 = x1.view(-1, self.num_nodes, self.gcn_channels[-1]).unsqueeze(3)
                    if idx == 0:
                        x_out = x1
                    else:
                        x_out = torch.cat((x_out, x1), dim=3)
                x = x_out
            elif i != num_layers - 1:
                x = F.elu(layer(x))
            else:
                x = layer(x)

        return x


class CNN_GCN_EFV2(nn.Module):
    def __init__(self, kpt_channels, gcn_channels, backbone=18, num_kpts=8, tsteps=2, volume_channels: List = [ 8, 16, 32, 32, 48], is_gpu=True):
        """
        Network for predicting kpts and volumes. Consists of a cnn encoder and gcn decoder.
        The volumes are predicted with a regression module.
        :param kpt_channels:
        :param gcn_channels:
        :param backbone:
        :param num_kpts:
        :param volume_channels:
        :param is_gpu:
        """
        super(CNN_GCN_EFV2, self).__init__()

        self.num_kpts = num_kpts
        self.output_features = kpt_channels*num_kpts
        self.volume_channels = volume_channels
        self.tsteps = tsteps
        self.image_encoder1 = image_encoder(backbone=backbone)

        self.regression_output = nn.Sequential(
                    nn.Linear(in_features=self.image_encoder1.img_feature_size, out_features=self.image_encoder1.img_feature_size//2, bias=True),
                    nn.LayerNorm(self.image_encoder1.img_feature_size//2),
                    nn.LeakyReLU(negative_slope=0.05, inplace=True),
                    nn.Linear(in_features=self.image_encoder1.img_feature_size//2, out_features=1, bias=True),
                    Reduce(),
                    nn.Sigmoid()
                )

        self.kpts_decoder1 = kpts_decoder_temporal(features_size=self.image_encoder1.img_feature_size,
                                                         kpt_channels=kpt_channels,
                                                         gcn_channels=gcn_channels,
                                                         num_kpts=num_kpts,
                                                         tsteps=self.tsteps,
                                                         is_gpu=is_gpu)

    def forward(self, x):
        #swap axes so that we have (B,C,T,H,W) instead of (B,C,H,W,T)
        x=x.permute(0,4,1,2,3)

        for idx in range(0,x.size(1)):
            if idx == 0:
                features = self.image_encoder1(x[:,idx,:,:,:]).unsqueeze(2)

            else:
                features1 = self.image_encoder1(x[:, idx, :, :, :]).unsqueeze(2)
                features = torch.cat((features, features1), dim=2)
        # concatenate feature vectors
        kpts = self.kpts_decoder1(features)
        ef = self.regression_output(features.permute(0,2,1))

        return kpts, ef


if __name__ == '__main__':
    num_batches = 64
    num_kpts = 40
    kpt_channels = 2 # kpts dim
    img = torch.rand(num_batches, 3, 112, 112, 2).cpu()
    #m = CNN_GCN(kpt_channels=kpt_channels, gcn_channels=[16, 32, 32, 48], backbone='mobilenet2_quantize', num_kpts=num_kpts)

    m = CNN_GCN_EFV2(kpt_channels=2, gcn_channels=[16, 32, 32, 48], backbone='resnet50', num_kpts=num_kpts,tsteps=2,
                       volume_channels=[4, 8, 8, 16, 16, 32, 32, 48], is_gpu=False)

    m = m.cpu()


    o = m(img)
    kpts = torch.rand(num_batches, num_kpts, kpt_channels).cuda()
    l2loss = nn.L1Loss()
    print(l2loss(o, kpts))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    pass
