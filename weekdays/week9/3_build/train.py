import sys
sys.path.append('.')

import os
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.parser import parse_args
from utils.save_folder import save_path_name
from networks.ResNet_network import ResNet
from utils.get_loader import get_loaders
from utils.eval import evaluate, evaluate_class

def main():
    #Step1. 파라미터 로드
    args = parse_args()

    #1-1. json 파일 형태로 하이퍼파라미터 저장 및 weight 파일 저장할 폴더 받아오기
    save_path = save_path_name(args)

    #Step2. 모델 인스턴스 생성
    model = ResNet(args.num_classes, args.resnet_config).to(args.device)

    #Step3. dataloader
    train_loader, test_loader = get_loaders(args)

    #Step4. Loss function & Optimizer 
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)

    #Step5. train
    _max = -1

    for _ in range(args.epochs):
        for idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(args.device), targets.to(args.device)
            
            #5-1. model output
            output = model(images)
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
                if _max < acc:
                    _max = acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "model_best.ckpt"),
                    )
                    print("best model weight 저장!!", acc)

if __name__ == '__main__':
    main()


