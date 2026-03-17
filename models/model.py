import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, num_classes)
        )

    def forward(self, x):
        return self.net(x)