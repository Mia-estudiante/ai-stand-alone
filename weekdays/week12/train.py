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
    args = parse_args()

    save_path = save_path_name(args)

    #1. data load
    train_loader, test_loader = get_loaders(args)
    
    #2. 모델 인스턴스 생성(by using Hyperparameters)
    model = IMDBTransformer(args.N, args.num_feature, args.num_head, train_loader.dataset.vocab, args.num_classes).to(args.device)

    #3. loss function 설정
    loss_fn = nn.CrossEntropyLoss()
    #4. Optimizer 설정
    optim = Adam(params=model.parameters(), lr=args.lr)

    #5. train
    _max = -1
    for _ in range(args.epochs):
        for idx, (texts, targets, lengths) in enumerate(train_loader):
            texts = texts.to(args.device)
            targets = targets.to(args.device)
            lengths = lengths.to('cpu')

            #5-1) input -> output
            output = model(texts, lengths)
            #5-2) 모델 output과 정답 비교
            loss = loss_fn(output, targets)
            #5-3) loss => parameters update
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