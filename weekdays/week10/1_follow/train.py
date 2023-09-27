import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.parser import parse_args
from utils.save_folder import save_path_name
from torchvision.models import resnet18, ResNet18_Weights
from utils.get_loader import get_loaders
from utils.eval import evaluate, evaluate_class

def main():
    #Step1. 파라미터 로드
    args = parse_args()

    #1-1. json 파일 형태로 하이퍼파라미터 저장 및 weight 파일 저장할 폴더 받아오기
    save_path = save_path_name(args)

    #Step2. 모델 인스턴스 생성
    model = resnet18(ResNet18_Weights)
    model.fc = nn.Linear(512, args.num_classes) #학습이 필요한 부분
    model = model.to(args.device)

    #Step3. dataloader
    train_loader, test_loader = get_loaders(args)

    #Step4. Loss function & Optimizer 
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam([
                    {'params': [weight for name, weight in model.named_parameters() \
                                if 'fc' not in name], 'lr': args.lr*0.01},
                    {'params': model.fc.parameters()}    
                ], lr=args.lr)
    
    #Step5. train
    _max = -1
    durations = [] 

    for _ in range(args.epochs): 
        for idx, (image, targets) in enumerate(train_loader) : 
            image, targets = image.to(args.device), targets.to(args.device)

            start = time.time() 
            #5-1. model output
            output = model(image)
            duration = time.time() - start
            durations.append(duration)

            #5-2. loss 계산
            loss = loss_fn(output, targets)

            #5-3. parameters update
            loss.backward()
            optim.step()
            optim.zero_grad()

            if idx%3==0:
                print(loss)

                #Step6. evaluate
                acc = evaluate(model, test_loader, args.device)
                acc_class = evaluate_class(model, test_loader, args.device, args.num_classes)
                
                #6-1. 이전 평가 결과와 비교해서 더 좋다면, save model parameters
                if _max < acc :
                    _max = acc 
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, 'model_best.ckpt')
                    )
                print("best model weight 저장!!", acc)
                print("duration", sum(durations) / len(durations))
                durations = [] 

if __name__ == '__main__' : 
    main() 