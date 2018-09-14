import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.nn import init

from torch.distributions.categorical import Categorical


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_size)
        self.critic = nn.Linear(128, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(CnnActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(7 * 7 * 64, 512),
        )
        self.actor = nn.Linear(512, output_size)
        self.critic = nn.Linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()


    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value
