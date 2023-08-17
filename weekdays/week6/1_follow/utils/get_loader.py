from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision.transforms import Compose

#normalize를 위한 값들
CIFAR10_MEAN = [0.491, 0.482, 0.447]
CIFAR10_STD = [0.247, 0.244, 0.262]
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

#transform 설정(전처리)
def get_transform(args):
    if args.data=='mnist':
        trans = Compose([
            Resize(args.img_size),
            ToTensor()
        ])
    elif args.data=='cifar':
        trans = Compose([
            Resize((args.img_size, args.img_size)),
            ToTensor(),
            Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])

    return trans

def get_loaders(args):
    #1-1. Dataset 설정
    if args.data == 'mnist':
        train_dataset = MNIST(root="../../data/mnist", train=True, transform=get_transform(args), download=True)
        test_dataset = MNIST(root="../../data/mnist", train=False, transform=get_transform(args), download=True)
    elif args.data == 'cifar':
        train_dataset = CIFAR10(root="../../data/cifar", train=True, transform=get_transform(args), download=True)
        test_dataset = CIFAR10(root="../../data/cifar", train=False, transform=get_transform(args), download=True)

    #1-2. Dataloader 설정
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader