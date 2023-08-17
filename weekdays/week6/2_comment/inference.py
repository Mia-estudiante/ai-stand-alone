import os
import torch
import torch.nn as nn
from PIL import Image

from utils.parse import infer_parse_args
from utils.load_hparam import load_hparams
from utils.get_loader import get_transform
from networks.LeNet_network import LeNet

def main():
    args = infer_parse_args()

    #Step1. 타겟하는 학습 세팅
    assert os.path.exists(args.trained_folder), "target folder doesn't exist!"
    assert os.path.exists(args.target_image), "target image doesn't exist!"

    #Step2. 하이퍼파라미터 로드
    args = load_hparams(args)

    #Step3. 모델 인스턴스 생성
    model = LeNet(args.img_size, args.num_classes).to(args.device)

    #Step4. 모델 weight 로드
    ckpt = torch.load(
        os.path.join(args.trained_folder, "model_best.ckpt")
    )
    model.load_state_dict(ckpt)

    #Step5. 추론할 이미지 가져오기 & 전처리
    test_image = Image.open(args.target_image)
    trans = get_transform(args)
    image = trans(test_image).to(args.device)
    image = image.unsqueeze(0)

    #5-1. model output
    output = model(image)

    #5-2. result 출력
    output = torch.argmax(output, dim=1).item()
    print(f'Model says, the image is {output}')
    

if __name__ == '__main__':
    main()