import torch
import torch.nn as nn

_RES18_BLOCKS = [2,2,2,2]
_RES34_BLOCKS = [3,4,6,3]
_RES50_BLOCKS = [3,4,6,3]
_RES101_BLOCKS = [3,4,23,3]
_RES152_BLOCKS = [3,8,36,3]

_CHANNELS_BLOCK = [64,128,256,512]
_CHANNELS_BOTTLE = [256,512,1024,2048]

class EMPTY(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
#InputPart   
class InputPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,64,7,2,3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3,2,1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x 
    
#OutputPart    
class OutputPart(nn.Module):
    def __init__(self, num_classes, config=18):
        super().__init__()
        in_feature = 512 if config in [18,34] else 2048
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_feature, num_classes)
    def forward(self, x):
        x = self.pool(x)
        b,c,h,w = x.shape
        x = torch.reshape(x, (-1, c*h*w))
        x = self.fc(x)
        return x
        
#conv - relu 사용 유무 확인 
class conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, filter_size=3, use_relu=True):
        super().__init__()
        self.use_relu = use_relu
        padding = 1 if filter_size==3 else 0
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        if self.use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x    
    
#Block - 3-3 구조
#h, w 크기를 맞출 필요성 존재
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=True):
        super().__init__()
        stride = 1
        self.relu = nn.ReLU()
        self.down_sample = down_sample

        if down_sample:
            stride = 2
            self.down_sample_net = conv(in_channel, out_channel, stride)

        self.conv1 = conv(in_channel, out_channel, stride=stride)
        self.conv2 = conv(out_channel, out_channel, use_relu=False)

    def forward(self, x):
        x_clone = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down_sample:
            x_clone = self.down_sample_net(x_clone)
        x += x_clone
        x = self.relu(x) 
        return x    
    
#BottleNeck - 1-3-1 구조
#h, w의 크기가 줄어드는 유무에 따라 down_sample_net 설정
class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=True):
        super().__init__()
        mid_channel = out_channel//4
        stride = 2 if down_sample else 1
        self.conv1 = conv(in_channel, mid_channel, stride=stride, filter_size=1)
        self.conv2 = conv(mid_channel, mid_channel)
        self.conv3 = conv(mid_channel, out_channel, filter_size=1, use_relu=False)
        self.relu = nn.ReLU()
        self.down_sample = down_sample
        self.down_sample_net = conv(in_channel, out_channel, stride=stride, filter_size=3)
    
    def forward(self, x):
        x_clone = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_clone = self.down_sample_net(x_clone)
        x += x_clone
        x = self.relu(x) 
        return x   
    
#MidPart        
class MidPart(nn.Module):
    def __init__(self, config=18):
        super().__init__()
        if config==18:
            self.block_nums = _RES18_BLOCKS
            self.TARGET = Block 
            self.channels = _CHANNELS_BLOCK
        elif config==34:
            self.block_nums = _RES34_BLOCKS
            self.TARGET = Block 
            self.channels = _CHANNELS_BLOCK
        elif config==50:
            self.block_nums = _RES50_BLOCKS
            self.TARGET = BottleNeck 
            self.channels = _CHANNELS_BOTTLE
        elif config==101:
            self.block_nums = _RES101_BLOCKS
            self.TARGET = BottleNeck 
            self.channels = _CHANNELS_BOTTLE
        elif config==152:
            self.block_nums = _RES152_BLOCKS
            self.TARGET = BottleNeck 
            self.channels = _CHANNELS_BOTTLE
        
        self.layer1 = self.make_layers(64, self.channels[0], self.block_nums[0], down_sample=False)
        self.layer2 = self.make_layers(self.channels[0], self.channels[1], self.block_nums[1])
        self.layer3 = self.make_layers(self.channels[1], self.channels[2], self.block_nums[2])
        self.layer4 = self.make_layers(self.channels[2], self.channels[3], self.block_nums[3])

    def make_layers(self, in_channel, out_channel, nums, down_sample=True):
        layer = [self.TARGET(in_channel, out_channel, down_sample)]
        
        for _ in range(nums-1):
            layer.append(self.TARGET(out_channel, out_channel, down_sample=False))
        layers = nn.Sequential(*layer)
        return layers
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
#ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes, config=18):
        super().__init__()
        self.inputpart = InputPart()
        self.outputpart = OutputPart(num_classes, config)
        self.midpart = MidPart(config)
    def forward(self, x):
        x = self.inputpart(x)
        x = self.midpart(x)
        x = self.outputpart(x)
        return x
