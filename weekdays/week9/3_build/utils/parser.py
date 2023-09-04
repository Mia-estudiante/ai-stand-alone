import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--do_save', action='store_true', help='if given, save results')
    
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar', 'dog'])
    parser.add_argument('--dataset-type', type=str, default='imagefolder', choices=['imagefolder', 'custom1', 'custom2'])
    parser.add_argument('--data-path', type=str, default='../../data')
    
    parser.add_argument('--vgg-config', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'])
    parser.add_argument('--resnet-config', type=int, default=18, choices=[18,34,50,101,152])
    
    args = parser.parse_args()
    return args

def infer_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trained-folder', type=str, default='results')
    parser.add_argument('--target-image', type=str)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    args = parser.parse_args()
    return args