import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple

class Reduce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x = x.mean(dim=1)
        return x


class volume_regressor(nn.Module):
    """ Sequence of linear layers to regress a single value from a latent feature vector. """
    """ Taken from pose regressor. Output channels are the number of outputs (volume = 1) """

    def __init__(self, features_size: int, decoding_channels: List = [16, 32, 32, 48], output_channels: int = 6):
        super(volume_regressor, self).__init__()

        self.decoding_channels = decoding_channels
        self.output_channels = output_channels

        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(features_size, self.decoding_channels[-1]))
        for idx in range(len(self.decoding_channels)):
            if idx == 0:
                self.decoder_layers.append(
                    nn.Linear(self.decoding_channels[-idx -1],
                              self.decoding_channels[-idx -1]))
            else:
                self.decoder_layers.append(
                    nn.Linear(self.decoding_channels[-idx],
                              self.decoding_channels[-idx - 1]))
        self.decoder_layers.append(
            nn.Linear(self.decoding_channels[0], self.output_channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                x = layer(x)
            elif i != num_layers - 1:
                x = F.elu(layer(x))
            else:
                x = layer(x)
        return x


class CNN_Regressor(nn.Module):
    def __init__(self, kpts_extractor=None, volume_channels: List = [ 8, 16, 32, 32, 48], is_gpu=True):
        """
        Network for predicting ejection fraction based on pretrained encoder. Feature vectors of the encoder are inserted in a regression layer. This layer is taken from https://github.com/HReynaud/UVT/blob/main/Network/model.py
        The ef predicted with a regression module.
        :param backbone:
        :param volume_channels:
        :param is_gpu:
        """
        super(CNN_Regressor, self).__init__()

        self.volume_channels = volume_channels #if we use the volume regression part
        self.kpts_extractor = kpts_extractor
        self.image_encoder = kpts_extractor.image_encoder
        self.regression_output = nn.Sequential(
                    nn.Linear(in_features=self.image_encoder.img_feature_size, out_features=self.image_encoder.img_feature_size//2, bias=True),
                    nn.LayerNorm(self.image_encoder.img_feature_size//2),
                    nn.LeakyReLU(negative_slope=0.05, inplace=True),
                    nn.Linear(in_features=self.image_encoder.img_feature_size//2, out_features=1, bias=True),
                    Reduce(),
                    nn.Sigmoid()
                )

    def forward(self, x):
        #swap axes so that we have (B,C,T,H,W) instead of (B,C,H,W,T)
        x=x.permute(0,4,1,2,3).contiguous()
        bs,frames,c,w,h = x.shape
        x = x.view(bs*frames,c,w,h)
        features = self.image_encoder(x)
        kpts = self.kpts_extractor(x)
        features = features.view(-1,frames,self.image_encoder.img_feature_size)
        kpts = kpts.view(-1,frames,40,2)
        kpts = kpts.permute(0,2,3,1)
        #use encoder output to regress ef directly
        ef = self.regression_output(features)
        return kpts, ef


if __name__ == '__main__':
    num_batches = 1
    tsteps = 2
    img = torch.rand(num_batches, 3, 112, 112,tsteps).cpu()
    m = CNN_Regressor(backbone='freezed_encoder',weight_file='/scratch/sarinat/code/gcn_up/gcn_ultrasound/tmp/logs/echonet40/CNNGCN/resnext50/143/weights_echonet40_CNNGCN_best_kptsErr.pth',
                       volume_channels=[4, 8, 8, 16, 16, 32, 32, 48], is_gpu=False)
    m = m.cpu()
    kpts,ef = m(img)
    l2loss = nn.L1Loss()
    print(kpts,ef)
    print(l2loss(m, kpts))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    pass
