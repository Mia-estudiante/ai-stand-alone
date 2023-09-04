import os
import torch
import torch.nn as nn
from torch.optim import Adam

from networks.ResNet_network import ResNet
from utils.parser import parse_args
from utils.save_folder import get_save_path
from utils.get_loader import get_loaders
from utils.eval import evaluate, evaluate_by_class

def main():
    args = parse_args()

    save_path = get_save_path(args)
    
    #1. 모델 인스턴스 생성(by using Hyperparameters)
    model = ResNet(args.num_classes, args.resnet_config).to(args.device)

    #2. data load
    train_loader, test_loader = get_loaders(args)

    #3. loss function 설정
    loss_fn = nn.CrossEntropyLoss()
    #4. Optimizer 설정
    optim = Adam(params=model.parameters(), lr=args.lr)

    #5. train
    _max = -1
    for epoch in range(args.epochs):
        for idx, (images, targets) in enumerate(train_loader):
            images = images.to(args.device)
            targets = targets.to(args.device)

            #5-1) input -> output
            output = model(images)
            #5-2) 모델 output과 정답 비교
            loss = loss_fn(output, targets)
            #5-3) loss => parameters update
            loss.backward()
            
            optim.step()
            optim.zero_grad()

            if idx%3==0:
                print(loss)
                #평가(로깅, print), 저장
                acc = evaluate(model, test_loader, args.device)
                acc_class = evaluate_by_class(model, test_loader, args.device, args.num_classes)
                
                # 평가 결과가 좋으면 타겟 폴더에 모델 weight 저장을 진행 
                # 평가 결과가 좋다는게 무슨 의미지? -> 과거의 평가 결과보다 좋은 수치가 나오면 결과가 좋다고 얘기합니다. 
                # 과거 결과(max) < 지금 결과(acc) 
                if _max < acc:
                    print('새로운 acc 등장, 모델 weight udpate ', acc)
                    _max = acc
                    torch.save(
                        model.state_dict(), #모델 weight 저장
                        os.path.join(args.target_folder, 'model_best.ckpt')
                    )

if __name__ == '__main__':
    main()