import os
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.parser import resume_parse_args
from utils.save_folder import get_save_path
from networks.LeNet_network import LeNet
from utils.get_loader import get_loaders
from utils.eval import evaluate, evaluate_by_class

def main():
    #Step1. 파라미터 로드
    args = resume_parse_args()
    #1-1. json 파일 형태로 하이퍼파라미터 저장
    save_path = get_save_path(args) 

    #Step2. 모델 인스턴스 생성
    model = LeNet(args.img_size, args.num_classes).to(args.device)

    #Step3. dataloader
    train_loader, test_loader = get_loaders(args)

    #Step5. Loss function & Optimizer 
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)

    #시작과 끝 epoch 설정
    start_epoch = 0
    end_epoch =  args.epochs
    
    ###################################################
    
    if args.resume_from:
        ckpt_path = os.path.join(args.results_folder, args.sub_results_folder)
        ckpt = torch.load(
            os.path.join(ckpt_path, "model_best.ckpt")
        )
        model.load_state_dict(ckpt['model']) 
        optim.load_state_dict(ckpt['optim'])

        start_epoch = ckpt['epochs']
        end_epoch = args.epochs

        print(f"{start_epoch} epoch 까지 학습된 model 재시작, {end_epoch-start_epoch} epoch 를 추가 학습!")
        print(f"{ckpt_path}에서 불러온 model weight file...")

    ###################################################
    
    #Step6. train
    _max = -1
    for epoch in range(start_epoch, end_epoch):
        for idx, (images, targets) in enumerate(train_loader):
            images = images.to(args.device)
            targets = targets.to(args.device)

            #6-1. model output
            output = model(images)
            #6-2. loss 계산
            loss = loss_fn(output, targets)
            #6-3. parameters update
            loss.backward()
            optim.step()
            optim.zero_grad()

            if idx%100==0:
                print(loss)
                
                #Step7. evaluate
                acc = evaluate(model, test_loader, args.device)
                acc_class = evaluate_by_class(model, test_loader, args.device, args.num_classes)

                #7-1. 이전 평가 결과와 비교해서 더 좋다면, save model parameters
                if _max < acc:
                    _max = acc
                    print('새로운 acc 등장, 모델 weight udpate ', acc)
                    torch.save({
                            "model": model.state_dict(),
                            "optim": optim.state_dict(),
                            "epochs": epoch,
                            "loss": loss
                        },
                        os.path.join(save_path, "model_best.ckpt")
                    )
if __name__ == "__main__":
    main()