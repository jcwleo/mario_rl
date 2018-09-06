import ray
import gym
import os
import random

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

from model import *

import torch.optim as optim

from collections import deque
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray


@ray.remote
class Environment(object):
    def __init__(self, env_id, is_render):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make(env_id)
        self.is_render = is_render

    def step(self, action):
        if self.is_render:
            self.env.render()
        obs, reward, done, info = self.env.step(action)
        
        if done:
            obs = self.reset()
            
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class ActorAgent(object):
    def __init__(self, input_size, output_size):
        self.model = BaseActorCriticNetwork(input_size, output_size)

        self.output_size = output_size
        self.input_size = input_size

    def get_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        state = state.float()
        policy, value = self.model(state)
        policy = policy.detach().numpy()
        action = np.random.choice(np.arange(self.output_size), p=policy[0])

        return action

    def train_model(self, s_batch, target_batch, y_batch, adv_batch):
        s_batch = torch.FloatTensor(s_batch)
        target_batch = torch.FloatTensor(target_batch)
        y_batch = torch.LongTensor(y_batch)
        adv_batch = torch.FloatTensor(adv_batch)
        # for multiply advantage
        policy, value = self.model(s_batch)
        m = Categorical(policy)

        # mse = nn.SmoothL1Loss()
        mse = nn.MSELoss()

        # Actor loss
        actor_loss = -m.log_prob(y_batch) * adv_batch.sum(1)

        # Entropy(for more exploration)
        entropy = m.entropy()
        # Critic loss
        critic_loss = mse(value, target_batch)

        # Total loss
        loss = actor_loss.mean() + 0.5 * critic_loss - 0.01 * entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 3)
        self.optimizer.step()


if __name__ == '__main__':
    env_id = 'CartPole-v0'
    env = gym.make(env_id)
    input_size = env.observation_space.shape[0]  # 4
    output_size = env.action_space.n  # 2
    env.close()

    num_env = 2
    is_render = False
    ray.init()
    envs = [Environment.remote(env_id, is_render) for _ in range(num_env)]
    
    # when Envs only run first, call reset().
    result = ray.get([env.reset.remote() for env in envs])

    for _ in range(1000):
        actions = [1] * num_env
        actions[0] = np.random.randint(output_size)

        result = ray.get([env.step.remote(action) for env, action in zip(envs, actions)])
        # print(result)
