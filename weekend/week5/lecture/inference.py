#패키지 임포트
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

#타겟하는 학습 세팅을 설정
target_folder = '../../weekdays/week5/results/4'
assert os.path.exists(target_folder), "target folder doesn't exist"

#하이퍼파라미터 로드
with open(os.path.join(target_folder, "hparam.txt"), "r") as f:
    data = f.readlines()

img_size = int(data[0].strip())
hidden_size = int(data[1].strip())
num_classes = int(data[2].strip())
batch_size = int(data[3].strip())
lr = float(data[4].strip())
epochs = int(data[5].strip())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_folder = data[6].strip()

#모델 class 만들기
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

#모델 객체 만들기
myMLP = MLP(img_size, hidden_size, num_classes).to(device)

#모델 weight를 업데이트
ckpt = torch.load(
    os.path.join(target_folder, 'myMLP_best.ckpt')
    )
myMLP.load_state_dict(ckpt)

#추론 데이터를 가져오기
#mnist 데이터는 기본적으로 grayscale 이미지이므로, 다운받은 이미지를 grayscale로 변환해야 함
image_path = './test_image.jpg'
assert os.path.exists(image_path), "target image doesn't exist"
input_image = Image.open(image_path).convert('L')

#학습 과정에서 사용했던 전처리 과정을 그대로 실행
#크기 맞추기 -> tensor로 만들기 
resizer = Resize(img_size)
totensor = ToTensor()
image = totensor(resizer(input_image))

#모델 추론 진행
output = myMLP(image)

#추론 결과를 우리가 이해할 수 있는 형태로 변환
output = torch.argmax(output).item()
print(f'Model says, the image is {output}')