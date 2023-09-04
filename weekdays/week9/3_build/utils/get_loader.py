import os

from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from dataset.dog import Dog
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split

CIFAR10_MEAN = [0.491, 0.482, 0.447]
CIFAR10_STD = [0.247, 0.244, 0.262]
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

def get_transform(args):
    if args.data == 'mnist':
        trans = Compose([
            Resize(args.img_size),
            ToTensor()
        ])
    elif args.data == 'cifar':
        trans = Compose([
            Resize((args.img_size, args.img_size)),
            ToTensor(),
            Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])  
    elif args.data == 'dog':
        trans = Compose([
            Resize((args.img_size, args.img_size)),
            ToTensor(),
            Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])      
    return trans

def get_loaders(args):
    root = os.path.join(args.data_path, args.data)
    transform = get_transform(args)

    #1) Dataset 설정
    if args.data == 'mnist':
        train_dataset = MNIST(root=root, train=True, transform=transform, download=True)
        test_dataset = MNIST(root=root, train=False, transform=transform, download=True)
    elif args.data == 'cifar':
        train_dataset = CIFAR10(root=root, train=True, transform=transform, download=True)
        test_dataset = CIFAR10(root=root, train=False, transform=transform, download=True)
    elif args.data == 'dog':
        root = os.path.join(args.data_path, 'dog_v1_imagefolder')
        train_path = os.path.join(root, 'train')
        test_path = os.path.join(root, 'val')
        #1. imagefolder 사용
        if args.dataset_type == 'imagefolder':
            train_dataset = ImageFolder(train_path, transform=transform) 
            test_dataset = ImageFolder(test_path, transform=transform) 
        #2. custom1 - Train, Test가 나뉘어 있는 경우(ImageFolder 구현)
        elif args.dataset_type == 'custom1':
            train_dataset = Dog(train_path, transform=transform)
            test_dataset = Dog(test_path, transform=transform, class2idx=train_dataset.class2idx)
        #3. custom2 - Train, Test가 나뉘어 있지 않은 경우(전체 데이터 이용 -> train_test_split)
        elif args.dataset_type == 'custom2':
            root = os.path.join(args.data_path, 'dog_v1')
            total_dataset = Dog(root, transform=transform)
            train_dataset, test_dataset = train_test_split(total_dataset, test_size=0.2, random_state=123)
    #2) Dataloader 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader
