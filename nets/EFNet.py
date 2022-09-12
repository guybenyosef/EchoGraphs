import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from nets.encoders import video_encoder

class MLP(nn.Module):
    """ Keypoints decoder from a latent feature vector. Based on a sequence of linear layers. """

    def __init__(self, features_size: int, channels: List = [16, 32, 32, 48], output_channels: int = 1):
        super(MLP, self).__init__()

        self.channels = channels
        self.output_channels = output_channels

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(features_size, self.channels[-1]))
        for idx in range(len(self.channels)):
            if idx == 0:
                self.layers.append(
                    nn.Linear(self.channels[-idx -1],
                              self.channels[-idx -1]))
            else:
                self.layers.append(
                    nn.Linear(self.channels[-idx],
                              self.channels[-idx - 1]))
        self.layers.append(
            nn.Linear(self.channels[0], self.output_channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
                guiding_layer = x
            elif i != num_layers - 1:
                x = F.elu(layer(x))
            else:
                x = layer(x)
        return x, guiding_layer


class EFNet(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """
    def __init__(self, backbone: str = 'r3d_18', MLP_channels: List = [16, 32, 32, 48]):

        super(EFNet, self).__init__()

        self.MLP_channels = MLP_channels

        self.encoder = video_encoder(backbone=backbone)
        self.ef_regressor = MLP(features_size=self.encoder.img_feature_size, channels=self.MLP_channels, output_channels=1)

    def forward(self, x: torch.Tensor, is_feat: bool = False) -> torch.Tensor:
        features = self.encoder(x)
        ef, guiding_layer = self.ef_regressor(features)

        if is_feat:
            return ef, guiding_layer
        else:
            return ef


if __name__ == '__main__':
    frame_size = 112
    num_frames = 2
    num_kpts = 40
    num_batches = 4
    kpt_channels = 2 # kpts dim
    print('load model')
    img = torch.rand(num_batches, 3, num_frames, frame_size, frame_size).cuda()
    m = EFNet(backbone='r3d_18', MLP_channels=[16, 32, 32, 48])
    m = m.cuda()
    print('model loaded')
    o = m(img)
    ef = torch.rand(num_batches).cuda()
    loss = nn.L1Loss()
    print(loss(o, ef))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    pass
