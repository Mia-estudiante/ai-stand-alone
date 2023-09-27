import os

from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from datasets.dog import Dog

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
        #pretrained model이 데이터 전처리시 사용했던 transform을 그대로 사용하기 위함
        if args.pretrained:
            from torchvision.transforms._presets import ImageClassification
            trans = Compose([
                ImageClassification(crop_size=224),
                Resize(args.img_size) #학습 과정의 편의성을 위해 resize 진행
            ])
        else:
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
        train_path = os.path.join(root, "train")
        test_path = os.path.join(root, "val")    

        if args.dataset_type == 'imagefolder':
            from torchvision.datasets import ImageFolder
            train_dataset = ImageFolder(root=train_path, transform=transform)
            print('train dataset uploaded')
            test_dataset = ImageFolder(root=test_path, transform=transform)
            print('validation dataset uploaded')
        elif args.dataset_type == 'custom1': #Train, Test가 나뉘어 있는 경우(ImageFolder 구현)
            train_dataset = Dog(root=train_path, transform=transform)
            print('train dataset uploaded')
            test_dataset = Dog(root=test_path, transform=transform, class2idx=train_dataset.class2idx)
            print('validation dataset uploaded')            
        elif args.dataset_type == 'custom2': #Train, Test가 나뉘어 있지 않은 경우
            root = os.path.join(args.data_path, 'dog_v1')
            total_dataset = Dog(root=root, transform=transform)
            train_dataset, test_dataset = train_test_split(total_dataset, test_size=0.2, random_state=123)
            print('train&validation dataset uploaded')
            
    #2) Dataloader 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader