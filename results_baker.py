import os
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator


def readEvent(event_path, scalarName):
    """
        读tensorboard生成的event文件中指定的标量值
            event_path:event文件路径
            scalarName：要操作的标量名称
    """
    event = event_accumulator.EventAccumulator(event_path)
    event.Reload()
    # print("\033[1;34m数据标签：\033[0m")
    # print(event.Tags())
    # print("\033[1;34m标量数据关键词：\033[0m")
    # print(event.scalars.Keys())
    value = event.scalars.Items(scalarName)
    # print("你要操作的scalar是：", scalarName)
    return value


def get_accuracy(base_path):
    data = {}
    for dirs in os.listdir(base_path):
        print(dirs)
        for file in os.listdir(os.path.join(base_path, dirs)):
            val = readEvent(os.path.join(base_path, dirs, file), 'Test/Accuracy')
            val = pd.DataFrame(val)
            data[dirs] = val['value'].max()
    return data


if __name__ == '__main__':
    base = '/home/lifabing/projects/pytorch-cifar100/runs/mobilenet'
    data = get_accuracy(base)
    data = [(key, data[key]) for key in sorted(data)]
    for i in range(len(data)):
        print('{}   {}'.format((i + 1) / 10., data[i][1]))


