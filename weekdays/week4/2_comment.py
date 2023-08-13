#필요한 모듈 import
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader #, Dataset
from torch.optim import Adam 

#하이퍼파라미터
img_size = 28
hidden_size = 500
num_classes = 10
batch_size = 100

lr = 0.001
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#모델 설계
class MLP(nn.Module):
    def __init__(self, img_size, hidden_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.mlp1 = nn.Linear(img_size*img_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.mlp4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x): #x: [batch_size, 1, 28, 28]
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, self.img_size*self.img_size)) #[batch_size, 1*28*28]
        x = self.mlp1(x) #[batch_size, 500]
        x = self.mlp2(x) #[batch_size, 500]
        x = self.mlp3(x) #[batch_size, 500]
        x = self.mlp4(x) #[batch_size, 10]

        return x

#1. 모델 인스턴스 생성(by using Hyperparameters)
myMLP = MLP(img_size, hidden_size, num_classes).to(device)

#2. data load
#2-1) Dataset 설정
train_mnist = MNIST(root="../../data/mnist", train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root="../../data/mnist", train=False, transform=ToTensor(), download=True)

#2-2) Dataloader 설정
train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

#3. loss function 설정
loss_fn = nn.CrossEntropyLoss()
#4. Optimizer 설정
optim = Adam(params=myMLP.parameters(), lr=lr)

#5. train
for epoch in range(epochs):
    for idx, (images, targets) in enumerate(train_loader): #batch size 만큼 데이터로더에서 데이터 꺼내오기
        images = images.to(device)
        targets = targets.to(device)

        #5-1) input -> output
        output = myMLP(images)
        #5-2) 모델 output과 정답 비교
        loss = loss_fn(output, targets)
        #5-3) loss => parameters update
        loss.backward() #loss 토대로 역전파 진행
        optim.step()    #매개변수 조정(optimizer 실행)
        
        optim.zero_grad() #이전 업데이트값이 현재 process에 영향을 주지 않기 위함

        if idx%100==0: #100개의 batch마다 loss 찍어보기 
            print(loss)