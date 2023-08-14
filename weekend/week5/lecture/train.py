import os
import torch
import json
import argparse
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader #, Dataset
from torch.optim import Adam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--do_save', action='store_true')
    parser.add_argument('--data', nargs='+', type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    #저장
    #상위 저장 폴더를 만들어야 함
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    
    # 내가 저장을 할 하위 폴더를 만들어야 함 (하위 폴더가 앞으로 사용될 타겟 폴더가 됨) 
    target_folder_name = max([0]+[int(e) for e in os.listdir(args.results_folder)])+1
    save_path = os.path.join(args.results_folder, str(target_folder_name))
    os.makedirs(save_path)

    # 타겟 폴더 밑에 hparam 저장 (text의 형태로)
    with open(os.path.join(save_path, 'hparam.json'), 'w') as f:
        write_args = args.__dict__  #namespace -> dict
        del write_args['device']
        json.dump(write_args, f, indent=4)

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
    myMLP = MLP(args.img_size, args.hidden_size, args.num_classes).to(args.device)

    #2. data load
    #2-1) Dataset 설정
    train_mnist = MNIST(root="../../data/mnist", train=True, transform=ToTensor(), download=True)
    test_mnist = MNIST(root="../../data/mnist", train=False, transform=ToTensor(), download=True)

    #2-2) Dataloader 설정
    train_loader = DataLoader(dataset=train_mnist, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_mnist, batch_size=args.batch_size, shuffle=False)

    #3. loss function 설정
    loss_fn = nn.CrossEntropyLoss()
    #4. Optimizer 설정
    optim = Adam(params=myMLP.parameters(), lr=args.lr)

    #평가 함수 구현 
    def evaluate(model, loader, device):
        with torch.no_grad(): #evaluate 시, gradient 를 구할 필요 없음
            model.eval() #평가라는 것을 모델에게 알려줌
            total = 0
            correct = 0
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                
                output_index = torch.argmax(output, dim=1)
                total += targets.shape[0]
                correct += (targets==output_index).sum().item()

        acc = correct/total*100
        model.train() #평가가 끝나면 다시 학습으로 돌아가도록 알려줌
        return acc

    #클래스별 평가함수
    def evaluate_by_class(model, loader, device, num_classes):
        with torch.no_grad():
            model.eval() 
            correct = torch.zeros(num_classes) 
            total = torch.zeros(num_classes) 
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                output_index = torch.argmax(output, dim=1)
                
                for _class in range(num_classes):
                    total[_class] += (targets==_class).sum().item()
                    correct[_class] += ((targets==_class) * (output_index==_class)).sum().item()

        acc = correct/total*100
        model.train() #평가가 끝나면 다시 학습으로 돌아가도록 알려줌
        return acc
        
    #5. train
    _max = -1
    for epoch in range(args.epochs):
        for idx, (images, targets) in enumerate(train_loader):
            images = images.to(args.device)
            targets = targets.to(args.device)

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
                #평가(로깅, print), 저장
                acc = evaluate(myMLP, test_loader, args.device)
                acc_class = evaluate_by_class(myMLP, test_loader, args.device, args.num_classes)
                
                # 평가 결과가 좋으면 타겟 폴더에 모델 weight 저장을 진행 
                # 평가 결과가 좋다는게 무슨 의미지? -> 과거의 평가 결과보다 좋은 수치가 나오면 결과가 좋다고 얘기합니다. 
                # 과거 결과(max) < 지금 결과(acc) 
                if _max < acc:
                    print('새로운 acc 등장, 모델 weight udpate ', acc)
                    _max = acc
                    torch.save(
                        myMLP.state_dict(), #모델 weight 저장
                        os.path.join(args.target_folder, 'myMLP_best.ckpt')
                    )

if __name__ == '__main__':
    main()