import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import torch


class layerchoice_model_1_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=32)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_1_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=32)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__stem__0__block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_1_1 = layerchoice_model_1_1()

    def forward(self, *_inputs):
        layerchoice_model_1_1 = self.layerchoice_model_1_1(_inputs[0])
        return layerchoice_model_1_1


class _model__stem__0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__block = _model__stem__0__block()

    def forward(self, x__1):
        __block = self.__block(x__1)
        return __block


class layerchoice_model_2_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=32)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_2_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=32)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__stem__1__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_2_1 = layerchoice_model_2_1()

    def forward(self, *_inputs):
        layerchoice_model_2_1 = self.layerchoice_model_2_1(_inputs[0])
        return layerchoice_model_2_1


class layerchoice_model_3_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=64)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_3_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=64)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__stem__1__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_3_0 = layerchoice_model_3_0()

    def forward(self, *_inputs):
        layerchoice_model_3_0 = self.layerchoice_model_3_0(_inputs[0])
        return layerchoice_model_3_0


class _model__stem__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__stem__1__depthwise()
        self.__pointwise = _model__stem__1__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class _model__stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = _model__stem__0()
        self.__1 = _model__stem__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class layerchoice_model_4_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=64)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_4_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=64)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv1__0__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_4_0 = layerchoice_model_4_0()

    def forward(self, *_inputs):
        layerchoice_model_4_0 = self.layerchoice_model_4_0(_inputs[0])
        return layerchoice_model_4_0


class layerchoice_model_5_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_5_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv1__0__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_5_1 = layerchoice_model_5_1()

    def forward(self, *_inputs):
        layerchoice_model_5_1 = self.layerchoice_model_5_1(_inputs[0])
        return layerchoice_model_5_1


class _model__conv1__0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv1__0__depthwise()
        self.__pointwise = _model__conv1__0__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_6_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_6_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv1__1__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_6_1 = layerchoice_model_6_1()

    def forward(self, *_inputs):
        layerchoice_model_6_1 = self.layerchoice_model_6_1(_inputs[0])
        return layerchoice_model_6_1


class layerchoice_model_7_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_7_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv1__1__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_7_0 = layerchoice_model_7_0()

    def forward(self, *_inputs):
        layerchoice_model_7_0 = self.layerchoice_model_7_0(_inputs[0])
        return layerchoice_model_7_0


class _model__conv1__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv1__1__depthwise()
        self.__pointwise = _model__conv1__1__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class _model__conv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = _model__conv1__0()
        self.__1 = _model__conv1__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class layerchoice_model_8_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_8_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=128)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv2__0__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_8_0 = layerchoice_model_8_0()

    def forward(self, *_inputs):
        layerchoice_model_8_0 = self.layerchoice_model_8_0(_inputs[0])
        return layerchoice_model_8_0


class layerchoice_model_9_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_9_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv2__0__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_9_1 = layerchoice_model_9_1()

    def forward(self, *_inputs):
        layerchoice_model_9_1 = self.layerchoice_model_9_1(_inputs[0])
        return layerchoice_model_9_1


class _model__conv2__0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv2__0__depthwise()
        self.__pointwise = _model__conv2__0__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_10_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=256, kernel_size=3, groups=256, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_10_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=256, kernel_size=3, groups=256, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv2__1__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_10_1 = layerchoice_model_10_1()

    def forward(self, *_inputs):
        layerchoice_model_10_1 = self.layerchoice_model_10_1(_inputs[0])
        return layerchoice_model_10_1


class layerchoice_model_11_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_11_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv2__1__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_11_0 = layerchoice_model_11_0()

    def forward(self, *_inputs):
        layerchoice_model_11_0 = self.layerchoice_model_11_0(_inputs[0])
        return layerchoice_model_11_0


class _model__conv2__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv2__1__depthwise()
        self.__pointwise = _model__conv2__1__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class _model__conv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = _model__conv2__0()
        self.__1 = _model__conv2__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class layerchoice_model_12_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=256, kernel_size=3, groups=256, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_12_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=256, kernel_size=3, groups=256, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=256)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__0__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_12_1 = layerchoice_model_12_1()

    def forward(self, *_inputs):
        layerchoice_model_12_1 = self.layerchoice_model_12_1(_inputs[0])
        return layerchoice_model_12_1


class layerchoice_model_13_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_13_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__0__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_13_1 = layerchoice_model_13_1()

    def forward(self, *_inputs):
        layerchoice_model_13_1 = self.layerchoice_model_13_1(_inputs[0])
        return layerchoice_model_13_1


class _model__conv3__0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv3__0__depthwise()
        self.__pointwise = _model__conv3__0__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_14_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_14_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__1__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_14_0 = layerchoice_model_14_0()

    def forward(self, *_inputs):
        layerchoice_model_14_0 = self.layerchoice_model_14_0(_inputs[0])
        return layerchoice_model_14_0


class layerchoice_model_15_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_15_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__1__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_15_0 = layerchoice_model_15_0()

    def forward(self, *_inputs):
        layerchoice_model_15_0 = self.layerchoice_model_15_0(_inputs[0])
        return layerchoice_model_15_0


class _model__conv3__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv3__1__depthwise()
        self.__pointwise = _model__conv3__1__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_16_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_16_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__2__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_16_1 = layerchoice_model_16_1()

    def forward(self, *_inputs):
        layerchoice_model_16_1 = self.layerchoice_model_16_1(_inputs[0])
        return layerchoice_model_16_1


class layerchoice_model_17_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_17_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__2__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_17_1 = layerchoice_model_17_1()

    def forward(self, *_inputs):
        layerchoice_model_17_1 = self.layerchoice_model_17_1(_inputs[0])
        return layerchoice_model_17_1


class _model__conv3__2(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv3__2__depthwise()
        self.__pointwise = _model__conv3__2__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_18_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_18_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__3__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_18_0 = layerchoice_model_18_0()

    def forward(self, *_inputs):
        layerchoice_model_18_0 = self.layerchoice_model_18_0(_inputs[0])
        return layerchoice_model_18_0


class layerchoice_model_19_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_19_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__3__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_19_1 = layerchoice_model_19_1()

    def forward(self, *_inputs):
        layerchoice_model_19_1 = self.layerchoice_model_19_1(_inputs[0])
        return layerchoice_model_19_1


class _model__conv3__3(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv3__3__depthwise()
        self.__pointwise = _model__conv3__3__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_20_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_20_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__4__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_20_1 = layerchoice_model_20_1()

    def forward(self, *_inputs):
        layerchoice_model_20_1 = self.layerchoice_model_20_1(_inputs[0])
        return layerchoice_model_20_1


class layerchoice_model_21_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_21_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__4__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_21_1 = layerchoice_model_21_1()

    def forward(self, *_inputs):
        layerchoice_model_21_1 = self.layerchoice_model_21_1(_inputs[0])
        return layerchoice_model_21_1


class _model__conv3__4(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv3__4__depthwise()
        self.__pointwise = _model__conv3__4__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_22_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_22_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, padding=1,
                                                bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__5__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_22_0 = layerchoice_model_22_0()

    def forward(self, *_inputs):
        layerchoice_model_22_0 = self.layerchoice_model_22_0(_inputs[0])
        return layerchoice_model_22_0


class layerchoice_model_23_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_23_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv3__5__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_23_1 = layerchoice_model_23_1()

    def forward(self, *_inputs):
        layerchoice_model_23_1 = self.layerchoice_model_23_1(_inputs[0])
        return layerchoice_model_23_1


class _model__conv3__5(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv3__5__depthwise()
        self.__pointwise = _model__conv3__5__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class _model__conv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = _model__conv3__0()
        self.__1 = _model__conv3__1()
        self.__2 = _model__conv3__2()
        self.__3 = _model__conv3__3()
        self.__4 = _model__conv3__4()
        self.__5 = _model__conv3__5()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        __3 = self.__3(__2)
        __4 = self.__4(__3)
        __5 = self.__5(__4)
        return __5


class layerchoice_model_24_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_24_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512, stride=2,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=512)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv4__0__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_24_0 = layerchoice_model_24_0()

    def forward(self, *_inputs):
        layerchoice_model_24_0 = self.layerchoice_model_24_0(_inputs[0])
        return layerchoice_model_24_0


class layerchoice_model_25_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=1024)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_25_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=1024)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv4__0__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_25_0 = layerchoice_model_25_0()

    def forward(self, *_inputs):
        layerchoice_model_25_0 = self.layerchoice_model_25_0(_inputs[0])
        return layerchoice_model_25_0


class _model__conv4__0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv4__0__depthwise()
        self.__pointwise = _model__conv4__0__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class layerchoice_model_26_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, groups=1024,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=1024)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_26_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, groups=1024,
                                                padding=1, bias=False)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=1024)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv4__1__depthwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_26_1 = layerchoice_model_26_1()

    def forward(self, *_inputs):
        layerchoice_model_26_1 = self.layerchoice_model_26_1(_inputs[0])
        return layerchoice_model_26_1


class layerchoice_model_27_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=1024)
        self.__2 = torch.nn.modules.activation.ReLU(inplace=True)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        __2 = self.__2(__1)
        return __2


class layerchoice_model_27_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.conv.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.__1 = torch.nn.modules.batchnorm.BatchNorm2d(num_features=1024)

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model__conv4__1__pointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice_model_27_1 = layerchoice_model_27_1()

    def forward(self, *_inputs):
        layerchoice_model_27_1 = self.layerchoice_model_27_1(_inputs[0])
        return layerchoice_model_27_1


class _model__conv4__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__depthwise = _model__conv4__1__depthwise()
        self.__pointwise = _model__conv4__1__pointwise()

    def forward(self, x__1):
        __depthwise = self.__depthwise(x__1)
        __pointwise = self.__pointwise(__depthwise)
        return __pointwise


class _model__conv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = _model__conv4__0()
        self.__1 = _model__conv4__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1


class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self.__stem = _model__stem()
        self.__conv1 = _model__conv1()
        self.__conv2 = _model__conv2()
        self.__conv3 = _model__conv3()
        self.__conv4 = _model__conv4()
        self.__avg = torch.nn.modules.pooling.AdaptiveAvgPool2d(output_size=1)
        self.__fc = torch.nn.modules.linear.Linear(in_features=1024, out_features=100)

    def forward(self, x__1):
        __Constant1 = -1
        __Constant2 = 0
        __stem = self.__stem(x__1)
        __conv1 = self.__conv1(__stem)
        __conv2 = self.__conv2(__conv1)
        __conv3 = self.__conv3(__conv2)
        __conv4 = self.__conv4(__conv3)
        __avg = self.__avg(__conv4)
        __aten__size185 = __avg.size(dim=__Constant2)
        __ListConstruct186 = [__aten__size185, __Constant1]
        __aten__view187 = __avg.view(size=__ListConstruct186)
        __fc = self.__fc(__aten__view187)
        return __fc
