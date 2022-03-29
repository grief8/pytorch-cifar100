import argparse
import os

import torch

from nas.tools import *
from utils import get_nas_network


def eval_oneshot(args, base_checkpoints_path):
    summary = get_clean_summary(reconstruct_model(get_nas_network(args, class_flag=True), base_checkpoints_path),
                                (3, 32, 32))
    print(sum(summary))
    # print('{} nonlinear ops with {} MB intermediate data'.format(len(summary), sum(summary)))


def generate_onnx():
    base_checkpoints_path = '/home/lifabing/projects/pytorch-cifar100/checkpoints/oneshot/mobilenet'
    for fpath in os.listdir(base_checkpoints_path):
        if fpath.endswith('.json'):
            model_path = os.path.join(base_checkpoints_path, fpath.strip('.json') + '.onnx')
            print(model_path)
            dummy_input1 = torch.randn(1, 3, 32, 32)
            input_names = ["input_1"]
            output_names = ["output_1"]
            torch.onnx.export(reconstruct_model(mobilenet, os.path.join(base_checkpoints_path, fpath)), dummy_input1,
                              model_path, verbose=True,
                              input_names=input_names,
                              output_names=output_names)


def travel_and_collect(args, base_path='/home/lifabing/projects/pytorch-cifar100/checkpoints/oneshot/'):
    data = dict()
    base_path = os.path.join(base_path, args.net)
    print('enter ', base_path)
    if os.path.exists(base_path):
        for sdir in os.listdir(base_path):
            file_path = os.path.join(base_path, sdir)
            for file in os.listdir(file_path):
                if file.endswith('.json'):
                    print(os.path.join(file_path, file))
                    try:
                        model = reconstruct_model(get_nas_network(args, class_flag=True),
                                                  os.path.join(file_path, file),
                                                  'cpu' if not args.gpu else 'cuda')
                        summary = get_clean_summary(model, (3, 32, 32))
                        print('{} nonlinear ops with {} MB intermediate data'.format(len(summary), sum(summary)))
                        data['{}-{}-{}'.format(args.net, sdir, file.strip('.json'))] = sum(summary)
                    except:
                        pass
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18', help='net type')
    parser.add_argument("--arc-checkpoint", default="./checkpoints/oneshot/mobilenet/checkpoint.json")

    args = parser.parse_args()
    eval_oneshot(args, args.arc_checkpoint)
    # generate_onnx()

    # data = travel_and_collect(args)
    # data = [(key, data[key]) for key in sorted(data)]
    # for i in range(len(data)):
    #     print('{}   {}'.format(data[i][0], data[i][1]))
