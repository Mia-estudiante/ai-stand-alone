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
    def __init__(self, in_channel=3, out_channel=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 7, 2, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(3, 2, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class OutputPart(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.in_channel = 512 if config in [18, 34] else 2048 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channel, num_classes)        
    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        return x

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size, use_relu=True):
        super().__init__()
        padding = 1 if filter_size==3 else 0
        self.conv = nn.Conv2d(in_channel, 
                                out_channel, 
                                filter_size, 1, padding)
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
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = conv(in_channel, out_channel, filter_size=3)
        self.conv2 = conv(out_channel, out_channel, filter_size=3, use_relu=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x_skip = x.copy()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + x_skip
        x = self.relu(x)
        return x   

class MiddlePart(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config==18:
            _nums = _NUMS_18
            _channels = _CHANNELS_33
        elif config==34:
            _nums = _NUMS_34
            _channels = _CHANNELS_33
        elif config==50:
            _nums = _NUMS_50
            _channels = _CHANNELS_131
        elif config==101:
            _nums = _NUMS_101
            _channels = _CHANNELS_131
        elif config==152:
            _nums = _NUMS_152
            _channels = _CHANNELS_131

        self.layer1 = self.make_layer(_nums[0], 64, _channels[0])
        self.layer2 = self.make_layer(_nums[1], _channels[0], _channels[1])
        self.layer3 = self.make_layer(_nums[2], _channels[1], _channels[2])
        self.layer4 = self.make_layer(_nums[3], _channels[2], _channels[3])

    #Block으로 이뤄진 한 층 제작
    def make_layer(self, num_blocks, in_channel, out_channel):
        layer = [
            Block(in_channel, out_channel)
        ]

        for _ in range(num_blocks-1):
            layer.append(
                Block(out_channel, out_channel)
            )

        layer = nn.Sequential(*layer)
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, num_classes, config=18):
        super().__init__()
        self.input_part = InputPart()
        self.middle_part = MiddlePart(config)
        self.output_part = OutputPart(config, num_classes)

    def forward(self, x):
        x = self.input_part(x)
        x = self.middle_part(x)
        x = self.output_part(x)
        return x
    