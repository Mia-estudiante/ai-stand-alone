import os
import torch
from PIL import Image

from utils.parser import infer_parse_args
from utils.load_hparam import load_hparams
from networks.ResNet_network import ResNet
from utils.get_loader import get_transform

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    #파라미터 로드
    args = infer_parse_args()

    #Step1. 타겟하는 학습 세팅
    assert os.path.exists(args.trained_folder), "target folder doesn't exist!"
    assert os.path.exists(args.target_image), "target image doesn't exist!"
    
    ckpt = torch.load(
        os.path.join(args.trained_folder, 'model_best.ckpt')
    )

    #Step2. 하이퍼파라미터 로드
    args = load_hparams(args)

    #Step3. 모델 인스턴스 생성
    model = ResNet(args.num_classes, args.resnet_config)

    #Step4. 모델 weight 로드
    model.load_state_dict(ckpt)

    #Step5. 추론할 이미지 가져오기 & 전처리
    test_image = Image.open(args.target_image)
    trans = get_transform(args)
    image = trans(test_image).unsqueeze(0)

    #5-1. model output
    model.eval() #batch norm이 들어가 있기에 넣어준다.
    output = model(image)
    idx = torch.argmax(output, dim=1).item()

    #5-2. result 출력
    print(f'model says, target image class is {CIFAR10_CLASSES[idx]}!')

if __name__ == '__main__':
    main()