import torch
import torch.nn as nn

_BLOCKS_A = [1,1,2,2,2]
_BLOCKS_B = [2,2,2,2,2]
_BLOCKS_C = [2,2,3,3,3]
_BLOCKS_D = [2,2,3,3,3]
_BLOCKS_E = [2,2,4,4,4]
CHANNELS = [3,64,128,256,512]

class EMPTY(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

#VGG_conv 구조 제작    
class VGG_conv(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size=3, use_one=False):
        super().__init__()
        padding = 0 if use_one else 1
        filter_size = 1 if use_one else filter_size
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, 1, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
#VGG_Block 구조 제작
class VGG_Block(nn.Module):
    def __init__(self, conv_nums, in_channel, out_channel, filter_size=3, use_one=False):
        super().__init__()

        #conv_list 제작
        module = [VGG_conv(in_channel, out_channel, filter_size)]
        for _ in range(conv_nums-1):
            module.append(
                VGG_conv(out_channel, out_channel)
            )

        #1by1 filter가 있는 경우
        if use_one:
            module.pop()
            module.append(
                VGG_conv(out_channel, out_channel, use_one=True)
            )
        
        self.conv_list = nn.ModuleList(module)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        for module in self.conv_list:
            x = module(x)
        x = self.pool(x)
        return x

#VGG_A  
class VGG_A(nn.Module):
    def __init__(self, num_blocks, channels):
        super().__init__()
        self.block1 = VGG_Block(num_blocks[0], channels[0], channels[1])
        self.block2 = VGG_Block(num_blocks[1], channels[1], channels[2])
        self.block3 = VGG_Block(num_blocks[2], channels[2], channels[3])
        self.block4 = VGG_Block(num_blocks[3], channels[3], channels[4])
        self.block5 = VGG_Block(num_blocks[4], channels[4], channels[4])
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
#VGG_B
class VGG_B(VGG_A):
    def __init__(self, num_blocks, channels):
        super().__init__()
        self.block1 = VGG_Block(num_blocks[0], channels[0], channels[1])
        self.block2 = VGG_Block(num_blocks[1], channels[1], channels[2])

#VGG_C
class VGG_C(VGG_B):
    def __init__(self, num_blocks, channels):
        super().__init__()
        self.block3 = VGG_Block(num_blocks[2], channels[2], channels[3], use_one=True)
        self.block4 = VGG_Block(num_blocks[3], channels[3], channels[4], use_one=True)
        self.block5 = VGG_Block(num_blocks[4], channels[4], channels[4], use_one=True)

#VGG_D
class VGG_D(VGG_B):
    def __init__(self, num_blocks, channels):
        super().__init__()
        self.block3 = VGG_Block(num_blocks[2], channels[2], channels[3])
        self.block4 = VGG_Block(num_blocks[3], channels[3], channels[4])
        self.block5 = VGG_Block(num_blocks[4], channels[4], channels[4])

#VGG_E
class VGG_E(VGG_D):
    def __init__(self, num_blocks, channels):
        super().__init__()
        self.block3 = VGG_Block(num_blocks[2], channels[2], channels[3])
        self.block4 = VGG_Block(num_blocks[3], channels[3], channels[4])
        self.block5 = VGG_Block(num_blocks[4], channels[4], channels[4])

#VGG
class VGG(nn.Module):
    def __init__(self, config, image_size, num_classes=1000):
        super().__init__()
        if config=='a':
            self.net = VGG_A(_BLOCKS_A, CHANNELS)
        elif config=='b':
            self.net = VGG_B(_BLOCKS_B, CHANNELS)
        elif config=='c':
            self.net = VGG_C(_BLOCKS_C, CHANNELS)
        elif config=='d':
            self.net = VGG_D(_BLOCKS_D, CHANNELS)
        elif config=='e':
            self.net = VGG_E(_BLOCKS_E, CHANNELS)
        self.classifier = VGG_Classifier(image_size, num_classes)
    def forward(self, x):
        x = self.net(x)
        _, c, h, w = x.shape
        x = torch.reshape(x, (_, -1, ))
        x = self.classifier(x)
        return x
    
#VGG_Classifier
class VGG_Classifier(nn.Module):
    def __init__(self, image_size, num_classes):
        super().__init__()
        in_features = 25088 if image_size==224 else 512
        self.fc1 = nn.Linear(in_features, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x