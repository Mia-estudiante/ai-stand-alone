# Debug mode를 이용해 상해있는 이 코드를 정상적으로 작동시켜보세요. 
# 총 11가지의 강제 error를 만들었습니다. 
# 에러를 고치면 에러가 발생한 line 뒤쪽에 주석으로 error의 원인을 적어주세요. 
# Debug mode의 call stack, debug console, variable 등의 과정을 충분히 활용해보세요 ^^ 
# 제출은 고친 파일(error의 원인이 적혀있는)과 
# debug mode를 활용해 디버깅하는 과정의 스크린샷을 찍어서 보내주세요!

import torch # 임포트 에러 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

lr = 0.001          #1. int형이 아닌 string형으로 변수 만듦
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 500 
total_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #2. if문이 참인 경우, cuda를 사용하도록 설정 안 함

class MLP(nn.Module): 
    def __init__(self, image_size, hidden_size, num_classes) : 
        super().__init__()    #3. init 빠짐 -> 부모 클래스 상속을 통한 초기화
        self.image_size = image_size
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size) #4. out_features를 mlp4의 input 사이즈와 동일하게 맞춰주지 않음
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self, x) : 
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, self.image_size * self.image_size)) #5. (-1, self.image_size * self.image_size) 로 고쳐줌으로써 Linear함수 input 크기에 맞춰줌
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

myMLP = MLP(image_size, hidden_size, num_classes)

train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True) #6-1. batch_size를 변수명이 아닌 string 형태로 삽입
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=True)   #6-2. batch_size를 변수명이 아닌 string 형태로 삽입

loss_fn = nn.CrossEntropyLoss()                #7. class가 아닌 instance를 받아오지 않음

optim = Adam(params=myMLP.parameters(), lr=lr) #8. myMLP 내 존재하는 함수 parameters가 아닌 변수를 삽입

for epoch in range(total_epochs): 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device)

        output = myMLP(image) 

        loss = loss_fn(output, label)

        loss.backward()
        optim.step()             #9. optimizer를 통한 파라미터 업데이트 진행
        optim.zero_grad()

        if idx // 100 == 0 : 
            print(loss)