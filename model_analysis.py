from conf import settings
from utils import get_network, get_test_dataloader

import argparse

from matplotlib import pyplot as plt

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    model_path = 'checkpoint'
    net = get_network(args)
    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    new_model = list(net.features.children())