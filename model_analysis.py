import torch
from torch import nn

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

from conf import settings
from utils import get_network, get_test_dataloader
from models.vgg import vgg16_bn
from models.refactor import NewVGG16
from tests.similarity import *

import argparse

from matplotlib import pyplot as plt


def get_origin_model():
    net = vgg16_bn()
    net.load_state_dict(torch.load('/home/lifabing/projects/pytorch-cifar100/checkpoint/vgg16'
                                   '/Thursday_17_March_2022_14h_01m_43s/vgg16-180-best.pth'))
    # new_model = list(net.features.children())[:]
    # return nn.Sequential(*new_model)
    return net


def get_eval_model(model_path):
    net = vgg16_bn()
    net = NewVGG16(net)
    net.load_state_dict(torch.load(model_path))
    # new_model = list(net.model.features.children())[:]
    # return nn.Sequential(*new_model)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    eval_model_path = '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Thursday_17_March_2022_21h_29m_06s/newvgg16-191-best.pth'
    origin_net = get_origin_model()
    eval_net = get_eval_model(eval_model_path)
    if args.gpu: #use_gpu
        origin_net = origin_net.cuda()
        eval_net = eval_net.cuda()
    origin_net.eval()
    eval_net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    kl_total = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output_origin = origin_net(image)
            output_eval = eval_net(image)

            _, pred = output_eval.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            correct_5 += correct[:, :5].sum()
            correct_1 += correct[:, :1].sum()
            kl_total += kl_divergence(output_eval.cpu().numpy(), output_origin.cpu().numpy())
    print("Top 1 acc: ", correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(cifar100_test_loader.dataset))
    print("Average KL Divergence: ", kl_total / len(cifar100_test_loader.dataset))