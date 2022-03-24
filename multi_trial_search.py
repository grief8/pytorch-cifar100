import os
import random
from pathlib import Path
import pickle

import nni
import torch
import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment, debug_mutated_model
from conf import settings
from nas.mobilenet import mobilenet
from utils import get_training_dataloader, get_test_dataloader

from models.mobilenet import MobileNet


def stat_nonlinear_ops(model, operators):
    """
    return sum of nonlinear operators
    """
    total = 0
    for module in model.children():
        total = total + stat_nonlinear_ops(module, operators)
        name = module.__class__.__name__
        for op in operators:
            if name.find(op) != -1:
                total = total + 1
    return total


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    # export model for visualization
    if 'NNI_OUTPUT_DIR' in os.environ:
        torch.onnx.export(model, (torch.randn(1, 3, 32, 32), ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=64,
        shuffle=True
    )
    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=64,
        shuffle=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    nonlinear_ops = ['ReLU', 'MaxPool']
    stat = MobileNet()
    total_nonlinear_ops = stat_nonlinear_ops(stat, nonlinear_ops)
    num_nonlinear_ops = stat_nonlinear_ops(stat, nonlinear_ops)
    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        accuracy = accuracy + (1-num_nonlinear_ops/total_nonlinear_ops)*0.02
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)


if __name__ == '__main__':
    base_model = mobilenet()

    search_strategy = strategy.Random()
    model_evaluator = FunctionalEvaluator(evaluate_model)

    exp = RetiariiExperiment(base_model, model_evaluator, [], search_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'cifar100_search'
    exp_config.trial_concurrency = 6
    exp_config.max_trial_number = 32
    exp_config.training_service.use_active_gpu = True
    export_formatter = 'dict'

    # uncomment this for graph-based execution engine
    exp_config.execution_engine = 'base'
    export_formatter = 'code'

    exp.run(exp_config, 8081 + random.randint(0, 100))
    print('Final model:')
    for model_code in exp.export_top_models(formatter=export_formatter):
        print(model_code)
    # with open('checkpoints/exp-mobilenet.o', 'wb') as f:
    #     pickle.dump(exp, f)