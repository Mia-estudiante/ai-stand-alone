import os

from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from custom_datasets.dog import Dog
from custom_datasets.imdb import IMDB, imdb_collate_fn

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
        #pretrained model이 사용한 데이터 전처리 과정을 동일하게 사용하기 위함
        if args.pretrained:
            from torchvision.transforms._presets import ImageClassification
            # from functools import partial
            trans = Compose([
                ImageClassification(crop_size=224),
                #학습 과정의 편의를 위해 이미지 크기 resize
                Resize((args.img_size, args.img_size))
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
    if args.data!='imdb': 
        transform = get_transform(args)
    collate_fn = None

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
    elif args.data == 'imdb':
        train_dataset = IMDB(split='train')
        test_dataset = IMDB(split='test', vocab=train_dataset.vocab)
        collate_fn = imdb_collate_fn(args.batch_size)

    #2) Dataloader 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, test_loader
