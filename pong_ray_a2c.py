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
class PongEnvironment(object):
    def __init__(self, env_id, is_render, env_idx, history_size=4, h=84, w=84):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make(env_id)
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rlist.append(0)
        self.history_size = history_size
        self.history = np.zeros([4, 84, 84])
        self.h = h
        self.w = w

    def step(self, action):
        if self.is_render:
            self.env.render()
        obs, reward, done, info = self.env.step(action + 1)

        self.history[:3, :, :] = self.history[1:, :, :]
        self.history[3, :, :] = self.pre_proc(obs)

        self.rall += reward
        self.steps += 1

        if reward != 0:
            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(self.episode, self.env_idx,
                                                                                        self.steps, self.rall,
                                                                                        np.mean(self.recent_rlist)))
                self.history = self.reset()
            done = True

        return self.history[:, :, :], reward, done, info

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]

    def pre_proc(self, X):
        x = resize(rgb2gray(X), (self.h, self.w), mode='reflect')
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


class ActorAgent(object):
    def __init__(self, input_size, output_size, num_env, num_step, gamma, lam=0.95, use_gae=True, use_cuda=False):
        self.model = CnnActorCriticNetwork(input_size, output_size)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.model.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(policy)

        return action

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def train_model(self, s_batch, target_batch, y_batch, adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch).to(self.device)
            target_batch = torch.FloatTensor(target_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        # for multiply advantage
        policy, value = self.model(s_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        mse = nn.SmoothL1Loss()
        # mse = nn.MSELoss()

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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

    @staticmethod
    def reform_batch(batchs):
        # (step, env, result) -> (env, step, result)
        batchs = np.stack(batchs).transpose([1, 0, 2])
        return batchs

    def make_train_date(self, batches):
        states = []
        targets = []
        actions = []
        advantages = []

        for idx in range(len(batches)):
            sample = np.stack(batches[idx])
            discounted_return = np.empty([self.num_step, 1])

            s = np.stack(sample[:, 4])
            s1 = np.stack(sample[:, 0])
            y = sample[:, 5]
            r = np.reshape(np.stack(sample[:, 1]), [self.num_step, 1])
            d = np.reshape(np.stack(sample[:, 2]), [self.num_step, 1]).astype(int)

            state = torch.from_numpy(s).to(self.device)
            state = state.float()
            _, value = agent.model(state)

            next_state = torch.from_numpy(s1).to(self.device)
            next_state = next_state.float()
            _, next_value = agent.model(next_state)

            value = value.data.cpu().numpy()
            next_value = next_value.data.cpu().numpy()

            # Discounted Return
            if self.use_gae:
                gae = 0
                for t in range(self.num_step - 1, -1, -1):
                    delta = r[t] + self.gamma * next_value[t, 0] * (1 - d[t]) - value[t, 0]
                    gae = delta + self.gamma * self.lam * (1 - d[t]) * gae

                    discounted_return[t, 0] = gae + value[t]

                # For critic
                target = r + self.gamma * (1 - d) * next_value

                # For Actor
                adv = discounted_return - value

            else:
                running_add = next_value[self.num_step - 1, 0] * (1 - d[self.num_step - 1, 0])
                for t in range(self.num_step - 1, -1, -1):
                    if d[t]:
                        running_add = 0
                    running_add = r[t] + self.gamma * running_add
                    discounted_return[t, 0] = running_add

                # For critic
                target = r + self.gamma * (1 - d) * next_value

                # For Actor
                adv = discounted_return - value

            states.extend(s)
            targets.extend(target)
            actions.extend(y)
            advantages.extend(adv)

        return states, targets, actions, advantages


if __name__ == '__main__':
    env_id = 'PongDeterministic-v4'
    env = gym.make(env_id)
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2
    output_size = 3

    env.close()
    use_cuda = False
    num_env = 1
    num_step = 5
    num_worker = 4
    gamma = 0.99
    agent = ActorAgent(input_size, output_size, num_env, num_step, gamma, use_cuda)
    is_render = False
    ray.init(num_cpus=num_worker)
    envs = [PongEnvironment.remote(env_id, is_render, idx) for idx in range(num_env)]

    # when Envs only run first, call reset().
    obs = ray.get([env.reset.remote() for env in envs])

    while True:
        batchs = []
        for _ in range(num_step):
            actions = agent.get_action(obs)

            result = ray.get([env.step.remote(action) for env, action in zip(envs, actions)])

            for i in range(num_env):
                result[i] = result[i] + (obs[i],) + (actions[i],)
                obs[i] = result[i][0]

            batchs.append(result)

        batchs = agent.reform_batch(batchs)

        train_data = agent.make_train_date(batchs)
        agent.train_model(*train_data)
        # print(train_data)
