#패키지 임포트
import os
import torch
from PIL import Image

from networks.MLP_network import MLP
from networks.LeNet_network import LeNet
from utils.parser import infer_parse_args
from utils.load_hparam import load_hparams
from utils.get_loader import get_transform

def main():
    args = infer_parse_args()

    #Step1. 타겟 폴더 학습 세팅
    #1-1. 타겟 폴더 존재 확인
    assert os.path.exists(args.trained_folder), "target folder doesn't exist"
    #1-2. 데이터 존재 확인
    assert os.path.exists(args.target_image), "target image doesn't exist"

    #Step2. 하이퍼파라미터 로드
    args = load_hparams(args)
    
    #Step3. 모델 객체 생성
    # myMLP = MLP(args.img_size, args.hidden_size, args.num_classes).to(args.device)
    model = LeNet(args.img_size, args.num_classes).to(args.device)

    #Step4. 모델 weight load
    ckpt = torch.load(
        os.path.join(args.trained_folder, "model_best.ckpt")
        )
    model.load_state_dict(ckpt)

    #Step5. inference 실행
    #5-1. rgb 이미지를 그대로 사용하기에 convert('L') 사용 x
    input_image = Image.open(args.target_image)

    #5-2. 학습 과정에서 사용했던 전처리 과정 그대로 실행
    #크기 맞추기 + tensor 형태 변환
    trans = get_transform(args)
    image = trans(input_image).to(args.device)
    image = image.unsqueeze(0) #batch 사이즈 추가

    #5-3. model output
    output = model(image)

    #5-4. 추론 결과를 우리가 이해할 수 있는 형태로 변환
    output = torch.argmax(output).item()
    print(f'Model says, the image is {output}')

if __name__ == '__main__':
    main()