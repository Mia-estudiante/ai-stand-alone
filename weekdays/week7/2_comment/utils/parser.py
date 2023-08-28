import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--do_save', action='store_true')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar'])
    parser.add_argument('--vgg-config', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'])
    
    args = parser.parse_args()
    return args

def infer_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trained-folder', type=str, default='results')
    parser.add_argument('--target-image', type=str)

    args = parser.parse_args()
    return args