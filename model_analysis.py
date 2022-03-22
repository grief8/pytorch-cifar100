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
    path_map = {
        '17-16': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Thursday_17_March_2022_16h_09m_47s/newvgg16-190-best.pth',
        '17-21': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Thursday_17_March_2022_21h_29m_06s/newvgg16-191-best.pth',
        '17-18': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Thursday_17_March_2022_18h_35m_38s/newvgg16-199-best.pth',
        '18-11': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Friday_18_March_2022_11h_32m_50s/newvgg16-200-best.pth',
        '18-16': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Friday_18_March_2022_16h_30m_02s/newvgg16-177-best.pth',
        '18-14': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Friday_18_March_2022_14h_41m_19s/newvgg16-183-best.pth',
        '18-18': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Friday_18_March_2022_18h_01m_12s/newvgg16-174-best.pth',
        '19-13': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Saturday_19_March_2022_13h_41m_53s/newvgg16-192-best.pth',
        '19-16': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Saturday_19_March_2022_16h_29m_39s/newvgg16-185-best.pth',
        '19-21': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Saturday_19_March_2022_21h_46m_13s/newvgg16-170-best.pth',
        '20-13': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Sunday_20_March_2022_13h_21m_04s/newvgg16-172-best.pth',
        '20-17': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Sunday_20_March_2022_17h_35m_09s/newvgg16-181-best.pth',
        '20-21': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Sunday_20_March_2022_21h_31m_09s/newvgg16-184-best.pth',
        '21-16': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Monday_21_March_2022_16h_56m_44s/newvgg16-174-best.pth',
        '21-20': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Monday_21_March_2022_20h_34m_06s/newvgg16-187-best.pth',
        '21-18': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Monday_21_March_2022_18h_48m_51s/newvgg16-196-best.pth',
        '21-22': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Monday_21_March_2022_22h_18m_00s/newvgg16-196-best.pth',
        '22-10': '/home/lifabing/projects/pytorch-cifar100/checkpoint/newvgg16/Tuesday_22_March_2022_10h_33m_51s/newvgg16-182-best.pth'
}
    eval_model_path = path_map['17-18']
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
            output_eval = origin_net(image)

            _, pred = output_eval.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            correct_5 += correct[:, :5].sum()
            correct_1 += correct[:, :1].sum()
            kl_total += kl_divergence(output_eval.cpu().numpy(), output_origin.cpu().numpy())
    print("Top 1 acc: ", correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(cifar100_test_loader.dataset))
    print("Average KL Divergence: ", kl_total / len(cifar100_test_loader.dataset))