import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from nets.encoders import video_encoder
from nets.EFNet import MLP


class EFKptsNet(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """
    def __init__(self, backbone: str = 'r3d_18', MLP_channels_ef: List = [16, 32, 32, 48], MLP_channels_kpts: List = [16, 32, 32, 48]):

        super(EFKptsNet, self).__init__()

        self.MLP_channels_ef = MLP_channels_ef
        self.MLP_channels_kpts = MLP_channels_kpts

        self.encoder = video_encoder(backbone=backbone)
        self.ef_regressor = MLP(features_size=self.encoder.img_feature_size, channels=self.MLP_channels_ef, output_channels=1)
        self.kpts_regressor = MLP(features_size=self.encoder.img_feature_size, channels=self.MLP_channels_kpts, output_channels=40*2*2)

    def forward(self, x: torch.Tensor, is_feat: bool = False) -> torch.Tensor:
        features = self.encoder(x)
        ef, guiding_layer_ef = self.ef_regressor(features)
        kpts, guiding_layer_kpts = self.kpts_regressor(features)

        if is_feat:
            return ef, kpts, guiding_layer_ef, guiding_layer_kpts
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
    m = EFKptsNet(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[96, 96, 124])
    m = m.cuda()
    print('model loaded')
    ef_pred, kpts_pred = m(img)
    ef, kpts = torch.rand(num_batches).cuda(), torch.rand(num_batches, 160).cuda()
    loss = nn.L1Loss()
    print(loss(ef_pred, ef) + loss(kpts_pred, kpts))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    print("hi")
    pass
