import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as torchvision_models

## the following block replaces the above: (a better workaround to the certificate problem")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class image_encoder(nn.Module):

    def __init__(self, backbone):
        super(image_encoder, self).__init__()

        # cnn encoder: use resnet features from the last hidden layer:
        ################
        hub_name = 'pytorch/vision:v0.10.0'#'pytorch/vision:v0.10.0'#'pytorch/vision:v0.6.0'
        if backbone == 'alexnet':
            self.base_net = torch.hub.load(hub_name, 'alexnet', pretrained=True)
            self.base_net.classifier = nn.Sequential(*list(self.base_net.classifier.children())[:-1])
            self.img_feature_size = 4096
        elif backbone == 'mobilenet2':
            self.base_net = torch.hub.load(hub_name, 'mobilenet_v2', pretrained=True)
            self.base_net.classifier = nn.Sequential(*list(self.base_net.classifier.children())[:-1])
            self.img_feature_size = 1280
        elif backbone == 'resnet18' or backbone == 18:
            self.base_net = torch.hub.load(hub_name, 'resnet18', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 512
        elif backbone == 'resnet50' or backbone == 50:
            self.base_net =torch.hub.load(hub_name, 'resnet50', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 2048
        elif backbone == 'densenet161' or backbone == 161:
            self.base_net = torch.hub.load(hub_name, 'densenet161', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1],
                                          nn.ReLU(inplace=True),
                                          nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten())
            self.img_feature_size = 2208
        elif backbone == 'densenet201' or backbone == 201:
            self.base_net = torch.hub.load(hub_name, 'densenet201', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1],
                                          nn.ReLU(inplace=True),
                                          nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Flatten())
            self.img_feature_size = 1920
        elif backbone == 'resnext50':
            self.base_net = torch.hub.load(hub_name, 'resnext50_32x4d', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 2048
        elif backbone == 'resnext101':
            self.base_net = torch.hub.load(hub_name, 'resnext101_32x8d', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 2048
        elif backbone == 'wide_resnet50':
            self.base_net = torch.hub.load(hub_name, 'wide_resnet50_2', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 2048
        elif backbone == 'wide_resnet101':
            self.base_net = torch.hub.load(hub_name, 'wide_resnet101_2', pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 2048
        else:
            raise NotImplementedError(
                "Backbone model {} is not supported".format(backbone)
            )


    def forward(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.base_net(x)
        x = torch.flatten(x, start_dim=1)
        return x


class video_encoder(nn.Module):

    def __init__(self, backbone):
        super(video_encoder, self).__init__()

        if backbone == 'r3d_18':
            self.base_net = torchvision_models.video.r3d_18(pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 512
        elif backbone == 'mc3_18':
            self.base_net = torchvision_models.video.mc3_18(pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 512
        elif backbone == 'r2plus1d_18':
            self.base_net = torchvision_models.video.mc3_18(pretrained=True)
            self.base_net = nn.Sequential(*list(self.base_net.children())[:-1])
            self.img_feature_size = 512
        else:
            raise NotImplementedError(
                "Backbone model {} is not supported".format(backbone)
            )


    def forward(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.base_net(x)
        x = torch.flatten(x, start_dim=1)

        return x


if __name__ == '__main__':
    num_train_imgs = 8
    batch_size = 1
    img = torch.rand(batch_size, 3, 16, 112, 112).cuda()
    kpts_gt = torch.rand(batch_size, 8, 2).cuda()
    m = video_encoder(backbone="r2plus1d_18").cuda()

    out = m(img)
    print(out.shape)
    # recon_loss = m.loss_function(reconstruction, recon_gt, mu, log_var, M_N=batch_size / num_train_imgs)
    # l2loss = nn.MSELoss()
    # loss = recon_loss['loss']
    # print(loss)
    # optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    # optimizer.step()
    pass

