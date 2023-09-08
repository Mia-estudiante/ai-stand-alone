from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

CIFAR10_MEAN = [0.491, 0.482, 0.447]
CIFAR10_STD = [0.247, 0.244, 0.262]
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

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
            Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return trans

def get_loaders(args):
    if args.data=='mnist':
        train_dataset = MNIST('../../data/mnist', train=True, transform=get_transform(args), download=True)  
        test_dataset = MNIST('../../data/mnist', train=False, transform=get_transform(args), download=True) 
    elif args.data=='cifar':
        train_dataset = CIFAR10("../../data/cifar", train=True, transform=get_transform(args), download=True)
        test_dataset = CIFAR10("../../data/cifar", train=False, transform=get_transform(args), download=True)
    
    train_loader= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader