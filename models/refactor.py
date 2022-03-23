import torch.nn as nn


def classifier(config, num_class=100):
    strategy = {'rm-r1': [1, 1, 3], 'rm-r1-r2': [1, 1, 3, 3], 'rm-r2': [1, 4, 4],
                'rp-r1': [2, 4], 'rp-r1-r2': [2, 5], 'rp-r2': [1, 5]}
    modules = [
        nn.Linear(512, 4096),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, num_class)
    ]
    for i in strategy[config]:
        modules.pop(i)
    return nn.Sequential(*modules)


class NewVGG16(nn.Module):
    def __init__(self, model, num_class=100):
        super(NewVGG16, self).__init__()
        self.model = model
        relu_conv = nn.Conv2d(64, 64, kernel_size=(1, 1))
        relu_conv2 = nn.Conv2d(64, 64, kernel_size=(2, 2), dilation=(1, 1), stride=(2, 2))
        relu_conv3 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        relu_conv4 = nn.Conv2d(128, 128, kernel_size=(2, 2), dilation=(1, 1), stride=(2, 2))
        mods = list(model.features.children())
        # mods[2] = relu_conv
        # mods.pop(2)
        # mods[5] = relu_conv
        mods[6] = relu_conv2
        # mods[9] = relu_conv3
        # mods[12] = relu_conv4
        # mods.pop(6)
        # mods.pop(5)
        # mods.pop(5)
        # mods.pop(12)
        self.model.features = nn.Sequential(*mods)
        # self.model.classifier = classifier('rp-r2', num_class)

    def forward(self, x):
        x = self.model(x)
        return x
