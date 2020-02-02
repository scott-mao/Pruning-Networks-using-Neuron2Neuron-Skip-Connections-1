import torch.nn as nn
import torch.nn.init as init
import math
from models.layers.expandergraphlayer import ExpanderLinear,ExpanderConv2d

__all__ = [
    'VGGexpander', 'vggexpander11', 'vggexpander11_bn', 'vggexpander13', 'vggexpander13_bn', 'vggexpander16', 'vggexpander16_bn',
    'vggexpander19_bn', 'vggexpander19',
]


class VGGexpander(nn.Module):
    def __init__(self, features,sparsity):
        super(VGGexpander, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            ExpanderLinear(512, 512, expandSize=int(512*sparsity/200)),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, sparsity,batch_norm=False):
    layers = [nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True)]
    in_channels = 64
    for i in range(len(cfg)):
        if cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if int(sparsity*cfg[i]/100) < in_channels:
                conv2d = ExpanderConv2d(in_channels, cfg[i], expandSize=int(sparsity*cfg[i]/200), kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[i]
    return nn.Sequential(*layers)


cfg = {
    'A': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
expandcfg = {
    'A': ['M', 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'B': [16, 'M', 32, 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'D': [64, 'M', 64, 64, 'M', 16, 16, 16, 'M', 16, 16, 16, 'M', 16, 16, 16, 'M'],
    'E': [16, 'M', 32, 32, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 64, 64, 64, 64, 'M'],
}


def vggexpander11(sparsity):
    """VGG 11-layer model (configuration "A")"""
    return VGGexpander(make_layers(cfg['A'], sparsity=10),sparsity)


def vggexpander11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGGexpander(make_layers(cfg['A'], sparsity=10, batch_norm=True))


def vggexpander13():
    """VGG 13-layer model (configuration "B")"""
    return VGGexpander(make_layers(cfg['B'], expandcfg['B']))


def vggexpander13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGGexpander(make_layers(cfg['B'], expandcfg['B'], batch_norm=True))


def vggexpander16(sparsity):
    """VGG 16-layer model (configuration "D")"""
    return VGGexpander(make_layers(cfg['D'], sparsity=sparsity),sparsity=sparsity)


def vggexpander16_bn(sparsity):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGGexpander(make_layers(cfg['D'], expandcfg['D'], batch_norm=True),sparsity=sparsity)


def vggexpander19():
    """VGG 19-layer model (configuration "E")"""
    return VGGexpander(make_layers(cfg['E'], expandcfg['E']))


def vggexpander19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGGexpander(make_layers(cfg['E'], expandcfg['E'], batch_norm=True))
