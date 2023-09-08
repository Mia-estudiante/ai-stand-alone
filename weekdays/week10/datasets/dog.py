#1. imagefolder 사용
#2. custom1 - 전체 데이터 사용
#3. custom2 - train, test 쪼개기
import os
import tqdm
from PIL import Image
from torch.utils.data import Dataset

class Dog(Dataset):
    def __init__(self, root, transform, class2idx=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.class_names = [cname for cname in os.listdir(root) if not cname.startswith('.DS')]
        self.image_names = []
        self.image_paths = []
        self.image_classes = []
        # self.images = []

        if class2idx is None:
            self.class2idx = {cname: idx for idx, cname in enumerate(self.class_names)}
        else:
            self.class2idx = class2idx
        self.idx2class = {value: key for key, value in self.class2idx.items()}

        for _class in self.class_names:
            class_path = os.path.join(self.root, _class)
            _images = os.listdir(class_path)
            for _image in tqdm.tqdm(_images):
                if _image.startswith('.DS'): continue
                self.image_names.append(_image)
                self.image_paths.append(os.path.join(class_path, _image))
                self.image_classes.append(_class)
                # self.images.append(Image.open(os.path.join(class_path, _image)))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)

        if image.mode!='RGB':
            image = image.convert('RGB')
        image = self.transform(image)

        label = self.image_classes[idx]
        label = self.class2idx[label]

        return image, label