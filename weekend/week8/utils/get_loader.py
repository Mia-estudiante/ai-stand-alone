import os

from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torch.utils.data import DataLoader #, Dataset
from torchvision.transforms import Compose
from torchvision.transforms import Normalize

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR100

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
    #2. data load
    #2-1) Dataset 설정
    if args.data == 'mnist':
        root = os.path.join(args.data_path, 'mnist')
        train_dataset = MNIST(root=root, train=True, transform=get_transform(args), download=True)
        test_dataset = MNIST(root=root, train=False, transform=get_transform(args), download=True)
    elif args.data == 'cifar':
        root = os.path.join(args.data_path, 'cifar')
        train_dataset = CIFAR100(root=root, train=True, transform=get_transform(args), download=True)
        test_dataset = CIFAR100(root=root, train=False, transform=get_transform(args), download=True)
    elif args.data == 'dog':
        if args.dataset_type == 'imagefolder':
            #ImageFolder는 모든 이미지를 전처리 후, PIL로 open해서 갖고 있음
            from torchvision.datasets import ImageFolder
            root = os.path.join(args.data_path, 'dog_v1_imagefolder')
            train_path = os.path.join(root, "train")
            test_path = os.path.join(root, "val")
            train_dataset = ImageFolder(
                root=train_path, 
                transform=get_transform(args)
            )
            print("train dataset uploaded")
            test_dataset = ImageFolder(
                root=test_path,
                transform=get_transform(args)
            )
            print("validation dataset uploaded")
        elif args.dataset_type == 'custom1':
            pass
        elif args.dataset_type == 'custom1':
            pass
    #2-2) Dataloader 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader