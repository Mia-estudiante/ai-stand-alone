import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 400)) #[batch_size, 5*5*16]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

#1. LeNet_Linear
class LeNet_Linear(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        #Linear 주입####
        self.inj_linear1 = nn.Sequential(
            nn.Linear(1176, 2048),
            nn.ReLU()
        ) 
        self.inj_linear2 = nn.Sequential(
            nn.Linear(2048, 1176),
            nn.ReLU()
        )
        #Linear 주입####

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.pool1(x)
        
        _, c, h, w = x.shape

        x = torch.reshape(x, (-1, (c*h*w)))
        x = self.inj_linear1(x)
        x = self.inj_linear2(x)
        x = torch.reshape(x, (-1, c, h, w))
        
        x = self.conv2(x)
        x = self.pool2(x)  

        x = torch.reshape(x, (-1, 400))

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

#2. LeNet_MultiConv
class LeNet_MultiConv(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.conv_block1_1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv_block1_2 = nn.Sequential(
            nn.Conv2d(6, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv_block1_3 = nn.Sequential(
            nn.Conv2d(6, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv_block1_4 = nn.Sequential(
            nn.Conv2d(6, 6, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.conv_blocks1 = nn.Sequential(
            self.conv_block1_1,
            self.conv_block1_2,
            self.conv_block1_3,
            self.conv_block1_4
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_block2_1 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_block2_2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_block2_3 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv_blocks2 = nn.Sequential(
            self.conv_block2_1,
            self.conv_block2_2,
            self.conv_block2_3,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_blocks1(x)
        x = self.pool1(x)
        x = self.conv_blocks2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 400)) #[batch_size, 5*5*16]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

#3. LeNet_Merge
class LeNet_Merge(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.img_size = img_size

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 6, 1, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 6, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)

        merg_x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv1(merg_x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 400)) #[batch_size, 5*5*16]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
