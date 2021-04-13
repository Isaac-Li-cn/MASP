import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


net = Net()

print(net)
