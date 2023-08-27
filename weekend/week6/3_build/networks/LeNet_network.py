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

        ##inj_linear
        self.inj_linear1 = nn.Linear(1176, 2048)
        self.inj_linear2 = nn.Linear(2048, 1176)
        ##inj_linear

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

        ##inj_linear
        _, c, h, w = x.shape
        x = torch.reshape(x, (-1, c*h*w))
        x = self.inj_linear1(x)
        x = self.inj_linear2(x)
        x = torch.reshape(x, (-1, c, h, w))
        ##inj_linear        

        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 400)) #[batch_size, 5*5*16]
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
        
#2. LeNet_MultiConv
class LeNet_MultiConv(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(6, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(6, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(6, 6, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.conv_blocks1 = nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.conv1_3,
            self.conv1_4
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv_blocks2 = nn.Sequential(
            self.conv2_1,
            self.conv2_2,
            self.conv2_3,
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
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 400)) #[batch_size, 5*5*16]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

#4. LeNet_Merge2
# LeNet_Merge 코드를 추가 및 변형하였음
# conv1, 2 앞에 conv를 병합하는 부분을 각각 넣음
# 기존 LeNet_Merge에서 conv를 병합하고 채널이 18개로 늘어났는데 이를 다시 6개로 줄이기에
# 채널 수를 줄이지 않고 다 활용하고자 했음
# 하지만 그러면 연산량이 많이 늘어날 것이기에 다시 조금씩 줄임
class LeNet_Merge2(nn.Module):
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
            nn.Conv2d(18, 9, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(9),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(9, 6, 1, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(9, 6, 3, 1, 1),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(9, 6, 5, 1, 2),    #in, out, f, s, p
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(18, 16, 5, 1, 0),    #in, out, f, s, p
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x1_3 = self.conv1_3(x)
        x = torch.cat([x1_1, x1_2, x1_3], dim=1)

        x = self.conv1(x)
        x = self.pool1(x)

        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x = torch.cat([x2_1, x2_2, x2_3], dim=1)

        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 400)) #[batch_size, 5*5*16]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
