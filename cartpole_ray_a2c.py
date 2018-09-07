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
from itertools import chain


@ray.remote
class CartPoleEnvironment(object):
    def __init__(self, env_id, is_render, env_idx):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make(env_id)
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rlist.append(0)

    def step(self, action):
        if self.is_render:
            self.env.render()
        obs, reward, done, info = self.env.step(action)
        self.rall += reward
        self.steps += 1

        if done:
            if self.steps < self.env.spec.timestep_limit:
                reward = -1

            self.recent_rlist.append(self.rall)
            print("[Episode {}({})] Reward: {}  Recent Reward: {}".format(self.episode, self.env_idx, self.rall,
                                                                          np.mean(self.recent_rlist)))
            obs = self.reset()

        return obs, reward, done, info

    def reset(self):
        self.step = 0
        self.episode += 1
        self.rall = 0

        return np.array(self.env.reset())


class ActorAgent(object):
    def __init__(self, input_size, output_size, num_env, num_step, gamma, lam=0.95, use_gae=True):
        self.model = BaseActorCriticNetwork(input_size, output_size)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_action(self, state):
        state = torch.Tensor(state)
        state = state.float()
        policy, value = self.model(state)
        policy = F.softmax(policy, dim=-1).detach().numpy()

        action = self.random_choice_prob_index(policy)

        return action

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def train_model(self, s_batch, target_batch, y_batch, adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch)
            target_batch = torch.FloatTensor(target_batch)
            y_batch = torch.LongTensor(y_batch)
            adv_batch = torch.FloatTensor(adv_batch)

        # for multiply advantage
        policy, value = self.model(s_batch)
        m = Categorical(F.softmax(policy, dim=-1))

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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

    @staticmethod
    def reform_batch(batchs):
        # (step, env, result) -> (env, step, result)
        batchs = np.stack(batchs).transpose([1, 0, 2])
        return batchs

    def foward_transition(self, batches):
        batches = np.concatenate(batches)

        s = np.reshape(np.stack(batches[:, 4]), [self.num_step * self.num_env, agent.input_size])
        s1 = np.reshape(np.stack(batches[:, 0]), [self.num_step * self.num_env, agent.input_size])

        state = torch.from_numpy(s)
        state = state.float()
        _, value = agent.model(state)

        next_state = torch.from_numpy(s1)
        next_state = next_state.float()
        _, next_value = agent.model(next_state)

        value = value.detach().numpy()
        next_value = next_value.detach().numpy()

        return value, next_value


@ray.remote
def make_train_data(batches, value, next_value):
    sample = np.stack(batches)
    discounted_return = np.empty([num_step, 1])

    s = np.reshape(np.stack(sample[:, 4]), [num_step, agent.input_size])
    y = sample[:, 5]
    r = np.reshape(np.stack(sample[:, 1]), [num_step, 1])
    d = np.reshape(np.stack(sample[:, 2]), [num_step, 1]).astype(int)

    # Discounted Return
    if use_gae:
        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = r[t] + gamma * next_value[t, 0] * (1 - d[t]) - value[t, 0]
            gae = delta + gamma * lam * (1 - d[t]) * gae

            discounted_return[t, 0] = gae + value[t]

        # For critic
        target = r + gamma * (1 - d) * next_value

        # For Actor
        adv = discounted_return - value

    else:
        running_add = next_value[num_step - 1, 0] * (1 - d[num_step - 1, 0])
        for t in range(num_step - 1, -1, -1):
            if d[t]:
                running_add = 0
            running_add = r[t] + gamma * running_add
            discounted_return[t, 0] = running_add

        # For critic
        target = r + gamma * (1 - d) * next_value

        # For Actor
        adv = discounted_return - value

    return s, target, y, adv

# @ray.remote
# def change_state(result, obs, action):
#     return (result + (obs,) + (action,))[0], (result + (obs,) + (action,))
#

if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    input_size = env.observation_space.shape[0]  # 4
    output_size = env.action_space.n  # 2
    env.close()

    num_env = 4
    num_step = 5
    num_worker = 4
    gamma = 0.99
    lam = 0.95
    use_gae = True

    agent = ActorAgent(input_size, output_size, num_env, num_step, gamma)
    is_render = False
    ray.init(num_cpus=num_worker)
    envs = [CartPoleEnvironment.remote(env_id, is_render, idx) for idx in range(num_env)]

    # when Envs only run first, call reset().
    obs = ray.get([env.reset.remote() for env in envs])

    while True:
        batchs = []
        for _ in range(num_step):
            actions = agent.get_action(obs)
            result = ray.get([env.step.remote(action) for env, action in zip(envs, actions)])

            # obs, result = ray.get([change_state.remote(result[i], obs[i], actions[i]) for i in range(num_env)])
            for i in range(num_env):
                result[i] = result[i] + (obs[i],) + (actions[i],)
                obs[i] = result[i][0]

            batchs.append(result)

        batchs = agent.reform_batch(batchs)
        value, next_value = agent.foward_transition(batchs)

        train_data = ray.get([make_train_data.remote(batchs[idx], value[idx * num_step:(idx + 1) * num_step],
                                                     next_value[idx * num_step:(idx + 1) * num_step]) for idx in range(num_env)])

        train_data = list(chain.from_iterable(train_data))
        states = np.vstack([train_data[4 * idx] for idx in range(num_env)])
        targets = np.concatenate([train_data[4 * idx + 1] for idx in range(num_env)])
        actions = np.concatenate([train_data[4 * idx + 2] for idx in range(num_env)]).astype(int)
        advantages = np.concatenate([train_data[4 * idx + 3] for idx in range(num_env)])

        agent.train_model(states, targets, actions, advantages)
        # print(train_data)
