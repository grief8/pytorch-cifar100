# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from nni.retiarii import fixed_arch

from conf import settings
from nas.mobilenet import mobilenet
from nas.tools import get_clean_summary
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy, get_training_dataloader, get_test_dataloader

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--constraints", default=1.0, type=float)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=True, action="store_true")
    parser.add_argument("--v1", default=False, action="store_true")
    parser.add_argument("--checkpoints", default='./checkpoints/oneshot/mobilenet/contraints-0.5.json', type=str)
    parser.add_argument("--model-path", default="./checkpoints/oneshot/mobilenet/contraints-0.5.onnx", type=str)

    args = parser.parse_args()

    dataset_train = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        wrap=False
    )
    dataset_valid = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        wrap=False
    )
    model = mobilenet()

    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    if args.v1:
        from nni.algorithms.nas.pytorch.darts import DartsTrainer
        trainer = DartsTrainer(model,
                               loss=criterion,
                               metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                               optimizer=optim,
                               num_epochs=args.epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               batch_size=args.batch_size,
                               log_frequency=args.log_frequency,
                               unrolled=args.unrolled,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
        if args.visualization:
            trainer.enable_visualization()

        trainer.train()
    else:
        from nas.new_darts import DartsTrainer
        from models.mobilenet import mobilenet as mb
        summary = get_clean_summary(mb(), (3, 32, 32))

        trainer = DartsTrainer(
            model=model,
            loss=criterion,
            metrics=lambda output, target: accuracy(output, target, topk=(1,)),
            optimizer=optim,
            num_epochs=args.epochs,
            dataset=dataset_train,
            batch_size=args.batch_size,
            constraints=args.constraints,
            log_frequency=args.log_frequency,
            unrolled=args.unrolled,
            nonlinear_summary=summary
        )
        trainer.fit()
        final_architecture = trainer.export()
        print('Final architecture:', trainer.export())
        json.dump(trainer.export(), open(args.checkpoints, 'w'))
        with fixed_arch(args.checkpoints):
            model = mobilenet()
            dummy_input1 = torch.randn(1, 3, 32, 32)
            input_names = ["input_1"]
            output_names = ["output1"]
            torch.onnx.export(model, dummy_input1, "./checkpoints/oneshot/mobilenet/mobilenet.onnx", verbose=True,
                              input_names=input_names,
                              output_names=output_names)