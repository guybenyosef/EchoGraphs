import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from nets.encoders import video_encoder
from nets.EFNet import MLP

class SDNet(nn.Module):

    def __init__(self, features_size: int = 512, num_label_frames: int = 16+1, num_sd_frames: int = 3):

        super(SDNet, self).__init__()

        self.net = nn.Sequential(
                   nn.Linear(in_features=features_size, out_features=features_size//2, bias=True),
                   nn.LayerNorm(features_size//2),
                   nn.LeakyReLU(negative_slope=0.05, inplace=True),
                   nn.Linear(in_features=features_size//2, out_features=features_size//4, bias=True),
                   nn.LayerNorm(features_size//4),
                   nn.LeakyReLU(negative_slope=0.05, inplace=True),
                   nn.Linear(in_features=features_size//4, out_features=num_sd_frames * num_label_frames, bias=True)
                   )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EFKptsNetSDNet(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """
    def __init__(self, backbone: str = 'r3d_18', MLP_channels_ef: List = [16, 32, 32, 48], MLP_channels_kpts: List = [16, 32, 32, 48]):

        super(EFKptsNetSDNet, self).__init__()

        self.MLP_channels_ef = MLP_channels_ef
        self.MLP_channels_kpts = MLP_channels_kpts
        self.num_frames = 16
        self.num_label_frames = self.num_frames + 1     # also include label 0 for 'transition' (or None ES or ED)
        self.num_SD_frames = 2  # ES, ED, or 'transition' (None)

        self.encoder = video_encoder(backbone=backbone)
        self.ef_regressor = MLP(features_size=self.encoder.img_feature_size, channels=self.MLP_channels_ef, output_channels=1)
        self.kpts_regressor = MLP(features_size=self.encoder.img_feature_size + self.num_SD_frames * self.num_label_frames,
                                  channels=self.MLP_channels_kpts, output_channels=40*2*2)
        self.sd_regressor = SDNet(features_size=self.encoder.img_feature_size, num_label_frames=self.num_label_frames, num_sd_frames=self.num_SD_frames)

    def forward(self, x: torch.Tensor, is_feat: bool = False) -> torch.Tensor:
        features = self.encoder(x)

        ef, guiding_layer_ef = self.ef_regressor(features)
        flatten_sd = self.sd_regressor(features)
        features_and_sd = torch.cat((features, flatten_sd), dim=1)
        kpts, guiding_layer_kpts = self.kpts_regressor(features_and_sd)

        num_batches = flatten_sd.shape[0]
        sd = torch.reshape(flatten_sd, (num_batches, self.num_label_frames, self.num_SD_frames))

        if is_feat:
            return ef, kpts, sd, guiding_layer_ef, guiding_layer_kpts
        else:
            return ef, kpts, sd


if __name__ == '__main__':
    frame_size = 112
    num_frames = 2
    num_kpts = 40
    num_batches = 4
    kpt_channels = 2 # kpts dim
    is_gpu = False
    print('load model')
    img = torch.rand(num_batches, 3, num_frames, frame_size, frame_size)
    if is_gpu:
        img = img.cuda()
    m = EFKptsNetSDNet(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[96, 96, 124])
    if is_gpu:
        m = m.cuda()
    print('model loaded')
    ef_pred, kpts_pred, sd_pred = m(img)
    ef, kpts, sd = torch.rand(num_batches, 1), torch.rand(num_batches, 160), torch.randint(low=0, high=16+1, size=(num_batches, 2))
    if is_gpu:
        ef, kpts, sd = ef.cuda(), kpts.cuda(), sd.cuda()
    loss = [nn.L1Loss(), nn.CrossEntropyLoss()]
    print(loss[0](ef_pred, ef) + loss[0](kpts_pred, kpts) + loss[1](sd_pred, sd))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    print("hi")
    pass
