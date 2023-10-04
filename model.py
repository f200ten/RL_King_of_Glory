import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCNN(nn.Module):
    def __init__(self, input_channels, train):
        super(SharedCNN, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.flt = nn.Flatten()
        
        self.fc = nn.Linear(256 * 34 * 60, 128)
        self.train = train

    def forward(self, x):
        if not self.train:
            x = torch.from_numpy(x).float()
        else:
            # x = x.float()
            pass
        if not isinstance(x, torch.Tensor):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x)
            else:
                raise ValueError("x is neither a torch.Tensor nor a numpy.ndarray.")
        x = self.avgpool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flt(x)
        if not self.train:
            x = x.view(1, -1)
        x = F.relu(self.fc(x))
        x = x.squeeze(0)
        return x

class Actor(nn.Module):
    def __init__(self, input_channels, output_dim, train):
        super(Actor, self).__init__()
        self.shared_cnn = SharedCNN(input_channels, train)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, state):
        x = self.shared_cnn(state)
        action_probs = torch.softmax(self.fc(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, input_channels, train):
        super(Critic, self).__init__()
        self.shared_cnn = SharedCNN(input_channels, train)
        self.fc = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_cnn(state)
        value = self.fc(x)
        return value
