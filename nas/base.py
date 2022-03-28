from torch import Tensor
import torch.nn as nn


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x