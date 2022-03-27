import os

import torch

from nas.tools import *
from nas.mobilenet import mobilenet


def eval_oneshot():
    base_checkpoints_path = '/home/lifabing/projects/pytorch-cifar100/checkpoints/oneshot/mobilenet'
    for fpath in os.listdir(base_checkpoints_path):
        if fpath.endswith('.json'):
            summary = get_clean_summary(reconstruct_model(mobilenet, os.path.join(base_checkpoints_path, fpath)),
                                        (3, 32, 32))
            print(fpath)
            print('{} nonlinear ops with {} MB intermediate data'.format(len(summary), sum(summary)))


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


if __name__ == '__main__':
    # eval_oneshot()
    generate_onnx()
