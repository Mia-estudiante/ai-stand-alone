import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--results_folder', type=str, default='results')
    
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar', 'dog', 'imdb'])
    parser.add_argument('--data-path', type=str, default='../../data')
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--N', type=int, default=3, )
    parser.add_argument('--num_feature', type=int, default=300, help='the number of transformers')
    parser.add_argument('--num_head', type=int, default=6)
    args = parser.parse_args()

    return args