import math
import torch
import torch.nn as nn
import torch.nn.init as init

def conv3x3(inplanes,outplanes):
  return nn.Conv2d(inplanes,outplanes,kernel_size=3, padding=1)


class vgg11(nn.Module):
    def __init__(self):
        super(vgg11,self).__init__()
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv3x3(64,128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128,256)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = conv3x3(256,256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = conv3x3(256,512)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = conv3x3(512,512)
        self.bn6 = nn.BatchNorm2d(512)     
        self.conv7 = conv3x3(512,512)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = conv3x3(512,512)
        self.bn8 = nn.BatchNorm2d(512)  
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,10)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(x)))))))
        x = self.pool(self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(x)))))))
        x = self.pool(self.relu(self.bn8(self.conv8(self.relu(self.bn7(self.conv7(x)))))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class vgg13(nn.Module):    
    def __init__(self):
        super(vgg13,self).__init__()
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1b = conv3x3(64,64)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv3x3(64,128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2b = conv3x3(128,128)
        self.bn2b = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128,256)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = conv3x3(256,256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = conv3x3(256,512)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = conv3x3(512,512)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = conv3x3(512,512)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = conv3x3(512,512)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self,x):
        x = self.pool(self.relu(self.bn1b(self.conv1b(self.relu(self.bn1(self.conv1(x)))))))
        x = self.pool(self.relu(self.bn2b(self.conv2b(self.relu(self.bn2(self.conv2(x)))))))
        x = self.pool(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(x)))))))
        x = self.pool(self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(x)))))))
        x = self.pool(self.relu(self.bn8(self.conv8(self.relu(self.bn7(self.conv7(x)))))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16,self).__init__()
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1d = conv3x3(64,64)
        self.bn1d = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv3x3(64,128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2d = conv3x3(128,128)
        self.bn2d = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128,256)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = conv3x3(256,256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4d = conv3x3(256,256)
        self.bn4d = nn.BatchNorm2d(256)
        self.conv5 = conv3x3(256,512)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = conv3x3(512,512)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv6d = conv3x3(512,512)
        self.bn6d = nn.BatchNorm2d(512)
        self.conv7 = conv3x3(512,512)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = conv3x3(512,512)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv8d = conv3x3(512,512)
        self.bn8d = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self,x):
        x = self.pool(self.relu(self.bn1d(self.conv1d(self.relu(self.bn1(self.conv1(x)))))))
        x = self.pool(self.relu(self.bn2d(self.conv2d(self.relu(self.bn2(self.conv2(x)))))))
        x = self.pool(self.relu(self.bn4d(self.conv4d(self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(x))))))))))
        x = self.pool(self.relu(self.bn6d(self.conv6d(self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(x))))))))))
        x = self.pool(self.relu(self.bn8d(self.conv8d(self.relu(self.bn8(self.conv8(self.relu(self.bn7(self.conv7(x))))))))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class vgg19(nn.Module):
    def __init__(self):
        super(vgg19,self).__init__()
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1e = conv3x3(64,64)
        self.bn1e = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv3x3(64,128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2e = conv3x3(128,128)
        self.bn2e = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128,256)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = conv3x3(256,256)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4d = conv3x3(256,256)
        self.bn4d = nn.BatchNorm2d(256)
        self.conv4e = conv3x3(256,256)
        self.bn4e = nn.BatchNorm2d(256)
        self.conv5 = conv3x3(256,512)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = conv3x3(512,512)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv6d = conv3x3(512,512)
        self.bn6d = nn.BatchNorm2d(512)
        self.conv6e = conv3x3(512,512)
        self.bn6e = nn.BatchNorm2d(512)
        self.conv7 = conv3x3(512,512)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = conv3x3(512,512)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv8d = conv3x3(512,512)
        self.bn8d = nn.BatchNorm2d(512)
        self.conv8e = conv3x3(512,512)
        self.bn8e = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,10)


        for m in self.modules():
          if isinstance(m, nn.Conv2d):
              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
              m.weight.data.normal_(0, math.sqrt(2. / n))
              m.bias.data.zero_()

    def forward(self,x):
        x = self.pool(self.relu(self.bn1e(self.conv1e(self.relu(self.bn1(self.conv1(x)))))))
        x = self.pool(self.relu(self.bn2e(self.conv2e(self.relu(self.bn2(self.conv2(x)))))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(self.relu(self.bn4e(self.conv4e(self.relu(self.bn4d(self.conv4d(self.relu(self.bn4(self.conv4(x))))))))))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(self.relu(self.bn6e(self.conv6e(self.relu(self.bn6d(self.conv6d(self.relu(self.bn6(self.conv6(x))))))))))
        x = self.pool(self.relu(self.bn7(self.conv7(x))))
        x = self.pool(self.relu(self.bn8e(self.conv8e(self.relu(self.bn8d(self.conv8d(self.relu(self.bn8(self.conv8(x))))))))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


