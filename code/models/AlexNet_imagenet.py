import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        #print(x.size())
        return x

'''
def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Net(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
'''

from models.layers.expandergraphlayer import ExpanderLinear, ExpanderConv2d
#COmpression ratio for 256-256-512 : 12.03%
__all__ = ['AlexNet', 'alexnet']


class AlexNetexpander(nn.Module):
    def __init__(self, sparsity,num_classes=1000):
        super(AlexNetexpander, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.05),
            ExpanderLinear(256 * 6 * 6, 4096, expandSize=int(256*6*6*sparsity/100)),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.05),
            ExpanderLinear(4096, 4096, expandSize=int(4096*sparsity/100)),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            ExpanderLinear(4096, num_classes, expandSize=int(4096*sparsity/100)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class AlexNetSkip(nn.Module):
    def __init__(self, sparsity,num_classes=1000):
        super(AlexNetSkip, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc1 = ExpanderLinear(256 * 6 * 6, 4096, expandSize=int(256*6*6*sparsity/100))
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = ExpanderLinear(4096, 4096, expandSize=int(4096*sparsity/100))
        self.fc3 = ExpanderLinear(256 * 6 * 6, num_classes, expandSize=int(256*6*6*sparsity/100))
        self.fc4 = ExpanderLinear(4096, num_classes, expandSize=int(4096*sparsity/100))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        out1 = self.bn1(self.relu(self.fc1(x)))
        out2 = self.bn1(self.relu(self.fc2(out1) + self.bn1(self.fc1(x))))   ##Check if adding relu helps
        out3 = (self.relu(self.fc4(out2) + self.bn2(self.fc4(out1)) + self.bn2(self.fc3(x))))  #Check if adding relu helps 
        return x

