import os
import tqdm #시각적 진행 확인 모듈
from PIL import Image
from torch.utils.data import Dataset

class Dog(Dataset):
    def __init__(self, path, transform, class2dix=None):
        super().__init__()
        self.image_names = []
        self.image_paths = []
        self.images = []
        self.image_classes = []
        self.transform = transform
        self.path = path
        self.class_name = [d for d in os.listdir(self.path) if not d.startswith('.DS')]
        
        if class2dix is None:
            self.class2idx = {c_name: idx for idx, c_name in enumerate(self.class_name)}
            self.idx2class = {value: key for key, value in self.class2idx.items()}
            # self.idx2class = {idx: c_name for idx, c_name in enumerate(self.class_name)}
        else:
            self.class2idx = class2dix
            self.idx2class = {value: key for key, value in self.class2idx.items()}
        
        for _class in self.class_name: #이미지에 해당하는 정보들 가져오기
            _images = os.listdir(os.path.join(self.path, _class))
            for _image in tqdm.tqdm(_images):
                if _image.startswith('.DS'): continue
                self.image_names.append(_image)
                self.image_paths.append(os.path.join(self.path, _class, _image))
                self.image_classes.append(_class)
                # self.images.append(Image.open(os.path.join(self.path, _class, _image)))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        
        if image.mode != "RGB":
            image = image.convert('RGB')
        image = self.transform(image)

        label = self.image_classes[idx]
        label = self.class2idx[label]

        return image, label