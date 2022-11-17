import argparse
import os.path

import scipy.stats
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from conf import settings
from utils import get_network, get_test_dataloader


def kl_divergence(x, y):
    x = x.reshape(-1).numpy()
    y = y.reshape(-1).numpy()
    std = max(abs(x.min()), abs(y.min()))
    x = x + std + 1
    y = y + std + 1
    KL = scipy.stats.entropy(x, y)
    return KL


def graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-path', type=str, required=True, help='the weights file you want to test')
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

    writer = SummaryWriter(log_dir=os.path.join('runs/featuremap/', args.net), comment='feature map')
    for i, data in enumerate(cifar100_test_loader, 0):
        # 获取训练数据
        inputs, labels = data
        if args.gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        x = inputs[0].unsqueeze(0)
        break

    # img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)

    # from models.resnet import resnet150 as base
    # # from models.linear_resnet import resnet152 as lr18
    # origin_model = base()
    origin_model = get_network(args)

    origin_model.load_state_dict(torch.load(args.path))
    origin_model.eval()

    for name, layer in origin_model._modules.items():
        # 为fc层预处理x
        x = x.view(x.size(0), -1) if 'fc' in name else x
        print(x.size())

        x = layer(x)
        print(f'{name}')

        # 查看卷积层的特征图
        if 'layer' in name or 'conv' in name:
            x1 = x.transpose(0, 1)  # C，B, H, W ---> B，C, H, W
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)
            writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)


def cal_kl():
    parser = argparse.ArgumentParser()
    parser.add_argument('-origin', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-approx', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    from models.resnet import resnet18 as base
    origin_model = base()
    from models.linear_resnet import resnet18 as lr18
    appro_model = lr18()
    # appro_model = base()
    origin_model.load_state_dict(torch.load(args.origin))
    appro_model.load_state_dict(torch.load(args.approx))
    origin_model.eval()
    appro_model.eval()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        # settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    total_KL = []
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')

            origin_output = origin_model(image)
            appro_output = appro_model(image)
            total_KL.append(kl_divergence(origin_output, appro_output))
    print(sum(total_KL)/len(total_KL))


if __name__ == '__main__':
    # graph()
    cal_kl()