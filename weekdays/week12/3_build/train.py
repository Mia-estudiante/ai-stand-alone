import os
import torch
import torch.nn as nn
from torch.optim import Adam

from networks.transformer import IMDBTransformer
from utils.parser import parse_args
from utils.save_folder import save_path_name
from utils.get_loader import get_loaders
from utils.eval import evaluate

def main():
    #Step1. 하이퍼파라미터 로드
    args = parse_args()
    #Step1-1. 하이퍼파라미터 저장
    save_path = save_path_name(args)
    #Step2. dataloader
    train_loader, test_loader = get_loaders(args)
    #Step3. 모델 인스턴스 생성
    model = IMDBTransformer(train_loader.dataset.vocab, args.N, args.num_feature, args.num_head, args.num_classes).to(args.device)
    #Step4. loss function & optimizer 생성
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)
    
    #Step5. train
    _max = -1
    for _ in range(args.epochs):
        for idx, (texts, targets, lengths) in enumerate(train_loader):
            texts = texts.to(args.device)
            targets = targets.to(args.device)
            lengths = lengths.to('cpu')

            #5-1. input -> output
            output = model(texts, lengths)
            #5-2. loss 계산
            loss = loss_fn(output, targets)
            #5-3. parameters update
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            if idx%100==0:
                #Step6. evaluate
                acc = evaluate(model, test_loader, args.device)
                #Step7. model parameters 저장
                if _max<acc:
                    _max = acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(save_path, "best_model.ckpt")
                    )

if __name__ == '__main__':
    main()