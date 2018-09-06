import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

from torch.distributions.categorical import Categorical


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
        super(CnnActorCriticNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(9 * 9 * 32, 256),
        )
        self.actor = nn.Linear(256, output_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        x = self.feature(state)
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value
