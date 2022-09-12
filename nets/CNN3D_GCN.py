import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from nets.CNN_GCN import SpiralConv
from nets.encoders import image_encoder, video_encoder

class spatiotemporal_kpts_decoder(nn.Module):

    def __init__(self, features_size, kpt_channels, gcn_channels, num_kpts=8, num_tpts=2, is_gpu=True):

        super(spatiotemporal_kpts_decoder, self).__init__()

        self.kpt_channels = kpt_channels
        self.gcn_channels = gcn_channels

        # construct nodes for graph CNN decoder:
        self.num_nodes = num_kpts

        adjacency = []
        for ii in range(self.num_nodes):
            x = list(range(self.num_nodes))
            x.insert(0, x.pop(ii))
            adjacency.append(x)
        adjacency = np.array(adjacency)

        if is_gpu:
            self.spiral_indices = torch.from_numpy(adjacency).cuda()
        else:
            self.spiral_indices = torch.from_numpy(adjacency).cpu()

        time_steps = num_tpts
        # if more than one time steps are present
        #if time_steps !=1:
        #indices = torch.cat((indices, indices[0,:].unsqueeze(1)+num_nodes), dim = 1)

        indices = self.spiral_indices

        for tpts in range(1,time_steps):
            indices = torch.cat((indices,self.spiral_indices+self.num_nodes*tpts),0)

        time_edges = torch.roll(indices[:,0], self.num_nodes)
        indices = torch.cat((indices,time_edges.unsqueeze(1)), dim=1)

        #add cycle connections if enough timesteps are present
        if time_steps > 3:
            time_edges= torch.roll(indices[:,0], -self.num_nodes)
            indices = torch.cat((indices,time_edges.unsqueeze(1)), dim=1)

        # could be used for disentangle spatial and temporal connections
        indices_spatial = indices[:,0:self.num_nodes]
        indices_temporal = indices[:,self.num_nodes:self.num_nodes+time_steps]


        self.spiral_indices = indices

        self.num_nodes = num_kpts*num_tpts

        # construct graph CNN layers:
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(features_size, self.num_nodes * self.gcn_channels[-1]))
        for idx in range(len(self.gcn_channels)):
            if idx == 0:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx - 1],
                               self.gcn_channels[-idx - 1],
                               self.spiral_indices))
            else:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx], self.gcn_channels[-idx - 1],
                               self.spiral_indices))
        self.decoder_layers.append(
            SpiralConv(self.gcn_channels[0], self.kpt_channels, self.spiral_indices))

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

class CNN3D_GCN(nn.Module):
    def __init__(self, kpt_channels, gcn_channels, backbone=18, num_kpts=8, num_tpts=2,is_gpu=True):

        super(CNN3D_GCN, self).__init__()
        self.video_encoder = video_encoder(backbone=backbone)

        self.kpts_decoder = spatiotemporal_kpts_decoder(features_size=self.video_encoder.img_feature_size,
                                                         kpt_channels=kpt_channels,
                                                         gcn_channels=gcn_channels,
                                                         num_kpts=num_kpts,
                                                         num_tpts=num_tpts,
                                                         is_gpu=is_gpu)

    def forward(self, x):
        #print(x.size())
        x=x.permute(0,1,4,2,3)
        #swap axes so that we have (B,C,T,H,W) instead of (B,C,H,W,T)
        features = self.video_encoder(x)

        # encoder splited by timepoints and then either given to num_tpts*encoder or one encoder with all timepoints
        kpts = self.kpts_decoder(features)

        return kpts

class CNN25D_GCN(nn.Module):
    def __init__(self, kpt_channels, gcn_channels, backbone=18, num_kpts=8, num_tpts=2,is_gpu=True):

        super(CNN25D_GCN, self).__init__()

        self.image_encoder1 = image_encoder(backbone=backbone)
        self.image_encoder2 = image_encoder(backbone=backbone)
        self.regression_output = nn.Linear(in_features=2*self.image_encoder1.img_feature_size, out_features=1, bias=True)
        self.kpts_decoder = spatiotemporal_kpts_decoder(features_size=2*self.image_encoder1.img_feature_size,
                                                         kpt_channels=kpt_channels,
                                                         gcn_channels=gcn_channels,
                                                         num_kpts=num_kpts,
                                                         num_tpts=num_tpts,
                                                         is_gpu=is_gpu)

    def forward(self, x):
        x=x.permute(0,1,4,2,3)
        #swap axes so that we have (B,C,T,H,W) instead of (B,C,H,W,T)

        features1 = self.image_encoder1(x[:,:,0,:,:])
        features2 = self.image_encoder2(x[:,:,1,:,:])
        # encoder splited by timepoints and then either given to num_tpts*encoder or one encoder with all timepoints
        ef = self.regression_output(torch.cat((features1,features2),1))
        # concatenate feature vectors
        kpts = self.kpts_decoder(torch.cat((features1,features2),1))

        return kpts, ef


class CNN_GCN_C(nn.Module):
    def __init__(self, kpt_channels, gcn_channels, backbone=18, num_kpts=8, num_tpts=2,is_gpu=True):

        super(CNN_GCN_C, self).__init__()

        self.image_encoder = image_encoder(backbone=backbone)

        self.num_tpts = 3 #use channel size as timepoint size
        self.num_kpts = num_kpts
        self.kpts_decoder = spatiotemporal_kpts_decoder(features_size=self.image_encoder.img_feature_size,
                                                         kpt_channels=kpt_channels,
                                                         gcn_channels=gcn_channels,
                                                         num_kpts=self.num_kpts,
                                                         num_tpts=self.num_tpts,
                                                         is_gpu=is_gpu)

    def forward(self, x):

        features = self.image_encoder(x)
        # encoder splited by timepoints and then either given to num_tpts*encoder or one encoder with all timepoints
        kpts_flatten = self.kpts_decoder(features)
        kpts = kpts_flatten.reshape([x.shape[0],self.num_tpts,self.num_kpts,2])

        return kpts

if __name__ == '__main__':
    frame_size = 112
    num_frames = 2
    num_kpts = 40
    num_batches = 4
    kpt_channels = 2 # kpts dim
    print('load model')
    img = torch.rand(num_batches, 3, num_frames, frame_size, frame_size).cuda()
    m = CNN3D_GCN(kpt_channels=kpt_channels, gcn_channels=[16, 32, 32, 48], backbone='r3d_18', num_kpts=num_kpts, num_tpts=num_frames)
    m = m.cuda()
    print('model loaded')
    o = m(img)
    kpts = torch.rand(num_batches, num_kpts*num_frames, kpt_channels).cuda()
    l2loss = nn.L1Loss()
    print(l2loss(o, kpts))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    pass
