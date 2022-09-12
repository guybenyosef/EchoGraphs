import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from nets.encoders import video_encoder
from nets.EFNet import MLP
from nets.EFKptsSDNet import SDNet
from nets.EFGCNTmp import kpts_decoder_temporal

class EFGCNSD(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """
    def __init__(self, backbone: str = 'r3d_18', MLP_channels_ef: List = [16, 32, 32, 48], GCN_channels_kpts: List = [16, 32, 32, 48], is_gpu=False):

        super(EFGCNSD, self).__init__()

        self.MLP_channels_ef = MLP_channels_ef
        self.GCN_channels_kpts = GCN_channels_kpts
        self.num_frames = 16
        self.num_label_frames = self.num_frames + 1     # also include label 0 for 'transition' (or None ES or ED)
        self.num_SD_frames = 2  # ES, ED, or 'transition' (None)

        self.encoder = video_encoder(backbone=backbone)
        self.ef_regressor = MLP(features_size=self.encoder.img_feature_size, channels=self.MLP_channels_ef, output_channels=1)
        # self.kpts_regressor = MLP(features_size=self.encoder.img_feature_size + self.num_SD_frames * self.num_label_frames,
        #                           channels=self.MLP_channels_kpts, output_channels=40*2*2)
        self.kpts_regressor = kpts_decoder_temporal(features_size=self.encoder.img_feature_size + self.num_SD_frames * self.num_label_frames,
                                         kpt_channels=2,
                                         gcn_channels=self.GCN_channels_kpts,
                                         tsteps=2,
                                         num_kpts=40 * 2,
                                         is_gpu=is_gpu)


        self.sd_regressor = SDNet(features_size=self.encoder.img_feature_size, num_label_frames=self.num_label_frames, num_sd_frames=self.num_SD_frames)

    def forward(self, x: torch.Tensor, is_feat: bool = False) -> torch.Tensor:
        features = self.encoder(x)

        ef, guiding_layer_ef = self.ef_regressor(features)
        flatten_sd = self.sd_regressor(features)
        features_and_sd = torch.cat((features, flatten_sd), dim=1)
        kpts = self.kpts_regressor(features_and_sd)

        num_batches = flatten_sd.shape[0]
        sd = torch.reshape(flatten_sd, (num_batches, self.num_label_frames, self.num_SD_frames))

        if is_feat:
            return ef, kpts, sd, guiding_layer_ef
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
    m = EFGCNSD(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
    if is_gpu:
        m = m.cuda()
    print('model loaded')
    ef_pred, kpts_pred, sd_pred = m(img)
    ef, kpts, sd = torch.rand(num_batches, 1), torch.rand(num_batches, num_kpts * 2, num_frames), torch.randint(low=0, high=16+1, size=(num_batches, 2))
    if is_gpu:
        ef, kpts, sd = ef.cuda(), kpts.cuda(), sd.cuda()
    loss = [nn.L1Loss(), nn.CrossEntropyLoss()]
    print(loss[0](ef_pred, ef) + loss[0](kpts_pred, kpts) + loss[1](sd_pred, sd))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    print("hi")
    pass
