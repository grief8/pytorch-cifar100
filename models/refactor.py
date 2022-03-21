import torch.nn as nn


class NewVGG16(nn.Module):
    def __init__(self, model, num_class=100):
        super(NewVGG16, self).__init__()
        self.model = model
        # relu_conv = nn.Conv2d(64, 64, kernel_size=(1, 1))
        # relu_conv2 = nn.Conv2d(64, 64, kernel_size=(2, 2), dilation=(1, 1), stride=(2, 2))
        # relu_conv3 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        # relu_conv4 = nn.Conv2d(128, 128, kernel_size=(2, 2), dilation=(1, 1), stride=(2, 2))
        # mods = list(model.features.children())
        # mods[2] = relu_conv
        # mods[5] = relu_conv2
        # mods[9] = relu_conv3
        # mods[12] = relu_conv4
        # mods.pop(6)
        # mods.pop(12)
        # self.model.features = nn.Sequential(*mods) # 替换原来的conv1
        self.model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            # nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            # nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.model(x)
        return x
