import torch.nn as nn


class NewVGG16(nn.Module):
    def __init__(self, model, num_class=100):
        super(NewVGG16, self).__init__()
        self.model = model
        classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        self.model.classifier = classifier  # 替换原来的conv1

    def forward(self, x):
        x = self.model(x)
        return x
