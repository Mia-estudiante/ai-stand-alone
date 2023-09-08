import torch
import torch.nn as nn

_NUMS_18 = [2, 2, 2, 2]
_NUMS_34 = [3, 4, 6, 3]
_NUMS_50 = [3, 4, 6, 3]
_NUMS_101 = [3, 4, 23, 3]
_NUMS_152 = [3, 8, 36, 3]

_CHANNELS_33 = [64, 128, 256, 512]
_CHANNELS_131 = [256, 512, 1024, 2048]

class EMPTY(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
class InputPart(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, filter_size=7, stride=2, padding=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, filter_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class OutputPart(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        in_feature = 512 if config in [18,34] else 2048
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_feature, num_classes)
    def forward(self, x):
        c = x.shape[0]
        x = self.pool(x)
        x = torch.reshape(x, (c, -1))
        x = self.fc(x)
        return x

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size, stride=1, use_relu=True):
        super().__init__()
        padding = 1 if filter_size==3 else 0
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False):
        super().__init__()
        stride = 1
        self.down_sample = down_sample
        
        if(self.down_sample):
            stride = 2              #h, w 크기를 맞춰줘야 하므로
            self.down_sample_net = conv(in_channel, out_channel, 3, stride=stride) 
        
        self.conv1 = conv(in_channel, out_channel, 3, stride=stride)
        self.conv2 = conv(out_channel, out_channel, 3, use_relu=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x_skip = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down_sample:
            x_skip = self.down_sample_net(x_skip)
        x = x + x_skip
        x = self.relu(x)
        return x    

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False): #h, w의 크기가 줄어드는 유무에 따라 down_sample 설정
        super().__init__()

        middle_channel = out_channel//4
        stride = 2 if down_sample else 1
        
        self.down_sample_net = conv(in_channel, out_channel, filter_size=3, stride=stride) 
        
        self.conv1 = conv(in_channel, middle_channel, filter_size=1, stride=stride)
        self.conv2 = conv(middle_channel, middle_channel, filter_size=3)
        self.conv3 = conv(middle_channel, out_channel, filter_size=1, use_relu=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x_skip = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_skip = self.down_sample_net(x_skip)
        x = x + x_skip
        x = self.relu(x)
        return x  

class MidPart(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config==18:
            _nums = _NUMS_18
            _channels = _CHANNELS_33
            self.TARGET = Block
        elif config==34:
            _nums = _NUMS_34
            _channels = _CHANNELS_33
            self.TARGET = Block
        elif config==50:
            _nums = _NUMS_50
            _channels = _CHANNELS_131
            self.TARGET = BottleNeck
        elif config==101:
            _nums = _NUMS_101
            _channels = _CHANNELS_131
            self.TARGET = BottleNeck
        elif config==152:
            _nums = _NUMS_152
            _channels = _CHANNELS_131
            self.TARGET = BottleNeck

        self.layer1 = self.make_layer(_nums[0], 64, _channels[0])
        self.layer2 = self.make_layer(_nums[1], _channels[0], _channels[1], down_sample=True)
        self.layer3 = self.make_layer(_nums[2], _channels[1], _channels[2], down_sample=True)
        self.layer4 = self.make_layer(_nums[3], _channels[2], _channels[3], down_sample=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def make_layer(self, num_blocks, in_channel, out_channel, down_sample=False):
        layers = [
            self.TARGET(in_channel, out_channel, down_sample=down_sample)
        ]
        
        for _ in range(num_blocks-1):
            layers.append(self.TARGET(out_channel, out_channel))
        
        layer = nn.Sequential(*layers)
        return layer

class ResNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.inputpart = InputPart()
        self.middlepart = MidPart(config)
        self.outputpart = OutputPart(config, num_classes)
    def forward(self, x):
        x = self.inputpart(x)
        x = self.middlepart(x)
        x = self.outputpart(x)
        return x