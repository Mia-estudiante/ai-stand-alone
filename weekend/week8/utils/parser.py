import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--do_save', action='store_true')

    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar', 'dog'])
    parser.add_argument('--data_path', type=str, default='../../data')
    parser.add_argument('--dataset_type', type=str, default='imagefolder', choices=['imagefolder', 'custom1', 'custom2'])
    
    args = parser.parse_args()
    return args

def infer_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--trained_folder', type=str)
    parser.add_argument('--target_image', type=str)
    
    # parser.add_argument('--img_size', type=int, default=28)
    # parser.add_argument('--hidden_size', type=int, default=500)
    # parser.add_argument('--num_classes', type=int, default=10)
    # parser.add_argument('--batch_size', type=int, default=100)            #학습과 관련된 내용들
    # parser.add_argument('--lr', type=float, default=0.001)                #학습과 관련된 내용들
    # parser.add_argument('--epochs', type=int, default=3)                  #학습과 관련된 내용들
    # parser.add_argument('--results_folder', type=str, default='results')  #학습과 관련된 내용들
    # parser.add_argument('--do_save', action='store_true')                 #학습과 관련된 내용들
    
    # parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar'])

    args = parser.parse_args()
    return args
