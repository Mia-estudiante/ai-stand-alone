import os
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

results_folder = 'results_hw'

#저장
#상위 저장 폴더 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
#하위 저장 폴더 생성
target_folder_name = max([0]+[int(e) for e in os.listdir(results_folder)])+1
target_folder = os.path.join(results_folder, str(target_folder_name))
os.makedirs(target_folder)    

#타겟 폴더 밑에 hparam 저장
with open(os.path.join(target_folder, 'hparam.txt'), 'w') as f:
    f.write(f'{img_size}\n')    
    f.write(f'{hidden_size}\n')
    f.write(f'{num_classes}\n')
    f.write(f'{batch_size}\n')
    f.write(f'{lr}\n')
    f.write(f'{epochs}\n')
    f.write(f'{results_folder}\n')

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

#평가 함수 구현
def evaluate(model, loader, device):
    with torch.no_grad(): #evaluate 시, gradient를 계산할 필요 없음
        model.eval()      #평가중이라는 것을 모델에게 알려줌
        total = 0
        correct = 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            output_index = torch.argmax(output, dim=1)
            total += targets.shape[0]
            correct += (output_index==targets).sum().item()
    acc = correct/total*100
    model.train()#평가가 끝나면 다시 학습으로 돌아가도록 알려줌
    return acc

#클래스별 평가 함수
def evaluate_by_class(model, loader, device, num_classes):
    with torch.no_grad(): #evaluate 시, gradient를 계산할 필요 없음
        model.eval()      #평가중이라는 것을 모델에게 알려줌
        total = torch.zeros(num_classes)
        correct = torch.zeros(num_classes)
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            output_index = torch.argmax(output, dim=1)
            
            for _class in range(num_classes):
                total[_class] += (targets==_class).sum().item()
                correct[_class] += ((targets==_class)*(output_index==_class)).sum().item()
    acc = correct/total*100
    model.train()#평가가 끝나면 다시 학습으로 돌아가도록 알려줌
    return acc

#5. train
_max = -1
for epoch in range(epochs):
    for idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        #5-1) input -> output
        output = myMLP(images)
        #5-2) 모델 output과 정답 비교
        loss = loss_fn(output, targets)
        #5-3) loss => parameters update
        loss.backward()
        
        optim.step()
        optim.zero_grad()

        if idx%100==0:
            print(loss)
            #평가
            acc = evaluate(myMLP, test_loader, device)
            print('클래스별 acc', evaluate_by_class(myMLP, test_loader, device, num_classes))
            #평가 결과가 좋다면, 타겟 폴더에 모델 weight 저장
            #과거 결과(_max) < 지금 결과(acc)
            if _max < acc:
                print('새로운 acc 등장, 모델 weight update', acc)
                _max = acc
                torch.save(
                    myMLP.state_dict(),
                    os.path.join(target_folder, 'myMLP_best.ckpt')
                )
