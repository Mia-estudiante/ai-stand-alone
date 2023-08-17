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
