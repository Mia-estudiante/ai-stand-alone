#패키지 임포트
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

#Step1. 타겟 폴더 학습 세팅
#1-1. 타겟 폴더 존재 확인
target_folder = '../../weekdays/week5/results/4'
assert os.path.exists(target_folder), "target folder doesn't exist!"
#1-2. load hyperparameters(with text file)
with open(os.path.join(target_folder, "hparam.txt"), "r") as f:
    hparams = f.readlines()

img_size = int(hparams[0].strip())  
hidden_size = int(hparams[1].strip())
num_classes = int(hparams[2].strip())
batch_size = int(hparams[3].strip())
lr = float(hparams[4].strip())
epochs = int(hparams[5].strip())
results_folder = hparams[6].strip()

#Step2. weight 파일을 로드할 모델 class 선언
class MLP(nn.Module):
    def __init__(self, img_size, hidden_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.mlp1 = nn.Linear(img_size*img_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)
        self.mlp4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x): #x: [batch_size, 1, 28, 28]
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, self.img_size*self.img_size)) #[batch_size, 1*28*28]
        x = self.mlp1(x) #[batch_size, 500]
        x = self.mlp2(x) #[batch_size, 500]
        x = self.mlp3(x) #[batch_size, 500]
        x = self.mlp4(x) #[batch_size, 10]

        return x

#Step3. 모델 객체 생성
myMLP = MLP(img_size, hidden_size, num_classes)

#Step4. 모델 weight load
weight_file = torch.load(os.path.join(target_folder, "myMLP_best.ckpt"))
myMLP.load_state_dict(weight_file)

#Step5. inference 실행
#5-1. 추론할 데이터 가져오기
image_path = './test_image.jpg'
#5-2. 데이터 존재 확인
assert os.path.exists(image_path), "test image doesn't exist!"
#5-3. mnist 데이터는 grayscale 이미지이므로, 다운받은 이미지를 grayscale로 변환해야 함
image = Image.open(image_path).convert('L')

#5-4. 학습 과정에서 사용했던 전처리 과정 그대로 실행
#크기 맞추기 + tensor 형태 변환
resize = Resize(img_size)
totensor = ToTensor()
image = totensor(resize(image))

#5-5. model output
output = myMLP(image)

#5-6. 추론 결과를 우리가 이해할 수 있는 형태로 변환
predict = torch.argmax(output, dim=1).item()
print(f"Best model predicts class {predict}!")