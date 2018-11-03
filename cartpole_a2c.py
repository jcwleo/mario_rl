import gym
import os
import random

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

from model import *

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray
from itertools import chain


class CartPoleEnvironment(Process):
    def __init__(self, env_id, env_idx, is_render, child_conn):
        super(CartPoleEnvironment, self).__init__()
        self.daemon = True
        self.env = gym.make(env_id)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rlist.append(0)
        self.child_conn = child_conn

        self.reset()

    def run(self):
        super(CartPoleEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if self.is_render:
                self.env.render()
            obs, reward, done, info = self.env.step(action)
            self.rall += reward
            self.steps += 1

            if done:
                if self.steps < self.env.spec.timestep_limit:
                    reward = -1

                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Reward: {}  Recent Reward: {}".format(
                    self.episode, self.env_idx, self.rall, np.mean(self.recent_rlist)))
                obs = self.reset()

            self.child_conn.send([obs, reward, done, info])

    def reset(self):
        self.step = 0
        self.episode += 1
        self.rall = 0

        return np.array(self.env.reset())


class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False):
        self.model = BaseActorCriticNetwork(
            input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=0.0224, eps=0.1, alpha=0.99)
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

        policy, value = self.model(s_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        # mse = nn.SmoothL1Loss()
        mse = nn.MSELoss()

        # Actor loss
        actor_loss = -m.log_prob(y_batch) * adv_batch

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        critic_loss = mse(value.sum(1), target_batch)

        # Total loss
        loss = actor_loss.mean() + 0.5 * critic_loss - 0.01 * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        self.optimizer.step()

    def forward_transition(self, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        _, value = agent.model(state)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value = agent.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value


def make_train_data(reward, done, value, next_value):
    discounted_return = np.empty([num_step])

    # Discounted Return
    if use_gae:
        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For critic
        target = reward + gamma * (1 - done) * next_value

        # For Actor
        adv = discounted_return - value

    else:
        running_add = next_value[num_step - 1, 0] * (1 - done[num_step - 1, 0])
        for t in range(num_step - 1, -1, -1):
            if d[t]:
                running_add = 0
            running_add = reward[t] + gamma * running_add
            discounted_return[t] = running_add

        # For critic
        target = r + gamma * (1 - done) * next_value

        # For Actor
        adv = discounted_return - value

    adv = (adv - adv.mean()) / (adv.std() + 1e-5)

    return target, adv


if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    input_size = env.observation_space.shape[0]  # 4
    output_size = env.action_space.n  # 2
    env.close()

    use_cuda = False
    num_worker_per_env = 1
    num_step = 5
    num_worker = 16
    use_noisy_net = True

    gamma = 0.99
    lam = 0.95
    use_gae = True

    agent = ActorAgent(
        input_size,
        output_size,
        num_worker_per_env *
        num_worker,
        num_step,
        gamma,
        use_gae=use_gae,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net)
    is_render = False

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = CartPoleEnvironment(env_id, idx, is_render, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker * num_worker_per_env, input_size])
    while True:
        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []

        for _ in range(num_step):
            actions = agent.get_action(states)
            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            rewards, dones, next_states = [], [], []
            for parent_conn in parent_conns:
                s, r, d, _ = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)

            next_states = np.vstack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)

            total_next_state.append(next_states)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            states = next_states[:, :]

        total_state = np.stack(total_state).transpose(
            [1, 0, 2]).reshape([-1, input_size])
        total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2]).reshape([-1, input_size])
        total_reward = np.stack(total_reward).transpose().reshape([-1])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose().reshape([-1])

        value, next_value = agent.forward_transition(
            total_state, total_next_state)

        total_target = []
        total_adv = []
        for idx in range(num_worker):
            target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                          total_done[idx *
                                                     num_step:(idx + 1) * num_step],
                                          value[idx *
                                                num_step:(idx + 1) * num_step],
                                          next_value[idx * num_step:(idx + 1) * num_step])
            # print(target.shape)
            total_target.append(target)
            total_adv.append(adv)

        agent.train_model(
            total_state,
            np.hstack(total_target),
            total_action,
            np.hstack(total_adv))
