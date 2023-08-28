import torch
import torch.nn as nn

class EMPTY(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

#VGG_conv 구조 제작    
class VGG_conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, one_filter=False):
        super().__init__()
        kernel_size = 1 if one_filter else 3
        padding = 0 if one_filter else 1
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#VGG_Block 구조 제작
class VGG_Block(nn.Module):
    def __init__(self, num_convs, in_channel, out_channel, one_filter=False):
        super().__init__()
        
        #conv_list 제작
        self.conv_list = [VGG_conv(in_channel, out_channel)]

        for _ in range(num_convs-1):
            self.conv_list.append(
                VGG_conv(out_channel, out_channel)
            )

        #1by1 filter가 있는 경우
        if one_filter:
            self.conv_list.pop()
            self.conv_list.append(
                VGG_conv(out_channel, out_channel, one_filter=True)
            )

        self.module = nn.ModuleList(self.conv_list)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for module in self.module:
            x = module(x)
        x = self.maxpool(x)
        return x

#VGG_A  
class VGG_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.vb1 = VGG_Block(1, 3, 64)
        self.vb2 = VGG_Block(1, 64, 128)
        self.vb3 = VGG_Block(2, 128, 256)
        self.vb4 = VGG_Block(2, 256, 512)
        self.vb5 = VGG_Block(2, 512, 512)

    def forward(self, x):
        x = self.vb1(x)
        x = self.vb2(x)
        x = self.vb3(x)
        x = self.vb4(x)
        x = self.vb5(x)   
        return x

#VGG_B
class VGG_B(VGG_A):
    def __init__(self):
        super().__init__()
        self.vb1 = VGG_Block(2, 3, 64)
        self.vb2 = VGG_Block(2, 64, 128)    

#VGG_C
class VGG_C(VGG_B):
    def __init__(self):
        super().__init__()
        self.vb3 = VGG_Block(3, 128, 256, one_filter=True)
        self.vb4 = VGG_Block(3, 256, 512, one_filter=True)
        self.vb5 = VGG_Block(3, 512, 512, one_filter=True)

#VGG_D
class VGG_D(VGG_B):
    def __init__(self):
        super().__init__()
        self.vb3 = VGG_Block(3, 128, 256)
        self.vb4 = VGG_Block(3, 256, 512)
        self.vb5 = VGG_Block(3, 512, 512)

#VGG_E
class VGG_E(VGG_D):
    def __init__(self):
        super().__init__()
        self.vb3 = VGG_Block(4, 128, 256)
        self.vb4 = VGG_Block(4, 256, 512)
        self.vb5 = VGG_Block(4, 512, 512)

#VGG
class VGG(nn.Module):  
    def __init__(self, image_size, num_classes, config='a'):
        super().__init__()
        
        if config=='a':
            self.net = VGG_A()
        elif config=='b':
            self.net = VGG_B()
        elif config=='c':
            self.net = VGG_C()
        elif config=='d':
            self.net = VGG_D()
        elif config=='e':
            self.net = VGG_E() 

        self.classifier = VGG_Classifier(num_classes, image_size)                        
    def forward(self, x):
        x = self.net(x)
        b, c, h, w = x.shape
        x = torch.reshape(x, (b, -1))
        # x = torch.reshape(x, (-1, c*h*w))
        x = self.classifier(x)
        return x   

#VGG_Classifier
class VGG_Classifier(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        in_feature = 512 if image_size==32 else 25088

        self.fc1 = nn.Linear(in_feature, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):

        return x