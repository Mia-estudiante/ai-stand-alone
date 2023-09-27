import os
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.parser import parse_args
from utils.save_folder import save_path_name
from networks.lstm import LSTM
from utils.get_loader import get_loaders
from utils.eval import evaluate#, evaluate_class

def main():
    #Step1. 파라미터 로드
    args = parse_args()

    #1-1. json 파일 형태로 하이퍼파라미터 저장 및 weight 파일 저장할 폴더 받아오기
    save_path = save_path_name(args)

    #Step2. dataloader - LSTM의 경우, 데이터를 먼저 호출해야 vocab을 알 수 있음
    train_loader, test_loader = get_loaders(args)

    #Step3. 텍스트 처리 모델 인스턴스 생성
    model = LSTM(vocab=train_loader.dataset.vocab, num_classes=args.num_classes, device=args.device).to(args.device)

    #Step4. Loss function & Optimizer 
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)
    # optim = Adam([
    #                 {'params': [weight for name, weight in model.named_parameters() \
    #                             if 'fc' not in name], 'lr': args.lr*0.01},
    #                 {'params': model.fc.parameters()}
    #             ], lr=args.lr)

    #Step5. train
    _max = -1

    for _ in range(args.epochs):
        for idx, (texts, targets, lengths) in enumerate(train_loader):
            texts = texts.to(args.device)
            targets = targets.to(args.device)
            lengths = lengths.to('cpu') #cpu로 맞춰줘야 pack_padded_seq 함수 사용 가능

            #5-1. model output
            output = model(texts, lengths)
            #5-2. loss 계산
            loss = loss_fn(output, targets)
            #5-3. parameters update
            loss.backward()
            optim.step()
            optim.zero_grad()

            if idx%100==0:
                print(loss)
                
                #Step6. evaluate
                acc = evaluate(model, test_loader, args.device)

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


