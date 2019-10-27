import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchsummary import summary
import numpy as np


class X_Enc(nn.Module):
    def __init__(self, layers, num_classes=1000, init_weights=True):
        super(X_Enc, self).__init__()

        self.features = nn.Sequential(*layers)  # layers
        print(self.features)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        all_maxpools = []
        for l in self.features:
            if isinstance(l, nn.MaxPool1d) == False:
                x = l(x)
            else:
                x, pool_indices = l(x)
                all_maxpools.append(pool_indices)
        return x, all_maxpools

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_enc(cfg):
    layers = []
    conv_layers = []
    in_channels = cfg[0]
    cfg = cfg[1:]
    for v in cfg:
        if v == 'M':
            layers += conv_layers  # [nn.Sequential(*conv_layers)]
            conv_layers = []
            layers += [nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            conv_layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    if len(conv_layers) > 0:
        layers += conv_layers  # [nn.Sequential(*conv_layers)]
    return layers


configs_enc = [
    [128, 128],
    [128, 128, 128, 'M', 256],
    [128, 128, 128, 'M', 256, 256, 'M', 512],
    [128, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512]
]

configs_dec = [
    [128, 128],
    [256, 128, 'M', 128, 128],
    [512, 256, 'M', 256, 128, 'M', 128, 128],
    [512, 512, 'M', 512, 256, 'M', 256, 128, 'M', 128, 128]
]


def encoder(x, pretrained_path=None, **kwargs):
    if pretrained_path is not None:
        kwargs['init_weights'] = False
    model = X_Enc(make_layers_enc(configs_enc[x-1]), **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    return model


class X_Dec(nn.Module):
    def __init__(self, layers, num_classes=1000, init_weights=True):
        super(X_Dec, self).__init__()

        self.layers = nn.Sequential(*layers)
        print(self.layers)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, all_maxpools):
        ct = -1
        for l in self.layers:
            if isinstance(l, nn.MaxUnpool1d) == False:
                x = l(x)
            else:
                x = l(x, all_maxpools[ct])
                ct -= 1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_dec(cfg):
    layers = []
    conv_layers = []
    in_channels = cfg[0]
    cfg = cfg[1:]
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += conv_layers  # [nn.Sequential(*conv_layers)]
            conv_layers = []
            layers += [nn.MaxUnpool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.ConvTranspose1d(in_channels, v, kernel_size=3, padding=1)
            if i != len(cfg) - 1:
                conv_layers += [conv1d, nn.ReLU(inplace=True)]
            else:
                conv_layers += [conv1d]
            in_channels = v
    if len(conv_layers) > 0:
        layers += conv_layers  # [nn.Sequential(*conv_layers)]
    return layers


def decoder(x, pretrained_path=None, **kwargs):
    if pretrained_path is not None:
        kwargs['init_weights'] = False
    model = X_Dec(make_layers_dec(configs_dec[x-1]), **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    encoder = vgg16_enc(x=3, pretrained=True)  # .to(device)
    for k in encoder.state_dict():
        print(k)
    summary(encoder, (3, 224, 224), device="cpu")
    z, all_maxpools = encoder(torch.from_numpy(np.zeros([1, 3, 224, 224])).float())

    decoder = vgg16_dec(x=3, pretrained=False)  # .to(device)
    for k in decoder.state_dict():
        print(k)
    x_rebuild = decoder(z, all_maxpools)
    # summary(decoder, (256, 56, 56), device="cpu")
