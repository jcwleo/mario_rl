import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CuriosityModel, CnnActorCriticNetwork, RNDModel


class A2CAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            beta=0.2,
            icm_scale=10.0,
            eta=0.01,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            train_method='A2C'):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.icm, self.rnd = None, None
        self.beta = beta
        self.eta = eta
        self.icm_scale = icm_scale
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        if train_method == 'ICM':
            self.icm = CuriosityModel(input_size, output_size)
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm.parameters()), lr=learning_rate)
            self.icm = self.icm.to(self.device)
        elif train_method == 'RND':
            self.rnd = RNDModel(input_size, output_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=learning_rate)
            self.rnd = self.rnd.to(self.device)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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

    def forward_transition(self, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value = self.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch).to(self.device)
            target_batch = torch.FloatTensor(target_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        # for multiply advantage
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
        loss = actor_loss.mean() + 0.5 * critic_loss - self.ent_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()


class PPOAgent(A2CAgent):
    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        sample_range = np.arange(len(s_batch))

        with torch.no_grad():
            # ------------------------------------------------------------
            # Calculate old policy
            policy_old_list = []
            for i in range(int(len(s_batch) / self.batch_size)):
                policy_old, _ = self.model(s_batch[self.batch_size * i: self.batch_size * (i + 1)])
                policy_old_list.extend(policy_old)

            policy_old_list = torch.stack(policy_old_list)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                policy, value = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value.sum(1), target_batch[sample_idx])

                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()


class ICMAgent(PPOAgent):
    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * (real_next_state_feature - pred_next_state_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()
        self.model.train()
        self.icm.train()

        with torch.no_grad():
            # ------------------------------------------------------------
            # Calculate old policy
            policy_old_list = []
            for i in range(int(len(s_batch) / self.batch_size)):
                policy_old, _ = self.model(s_batch[self.batch_size * i: self.batch_size * (i + 1)])
                policy_old_list.extend(policy_old)

            policy_old_list = torch.stack(policy_old_list)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                action_onehot = torch.FloatTensor(len(s_batch[sample_idx]), self.output_size).to(self.device)
                action_onehot.zero_()
                action_onehot.scatter_(1, y_batch[sample_idx], 1)

                real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                    [s_batch[sample_idx], next_s_batch[sample_idx], action_onehot])

                inverse_loss = ce(
                    pred_action, y_batch[sample_idx].detach())

                forward_loss = forward_mse(
                    pred_next_state_feature, real_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                policy, value = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])
                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = (actor_loss + 0.5 * critic_loss) + self.icm_scale * \
                       ((1 - self.beta) * inverse_loss + self.beta * forward_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.icm.parameters()),
                    self.clip_grad_norm)
                self.optimizer.step()


class RNDAgent(PPOAgent):
    def compute_intrinsic_reward(self, next_state):
        next_state = torch.FloatTensor(next_state).to(self.device)

        target_next_feature = self.rnd.target(next_state)
        predict_next_feature = self.rnd.predictor(next_state)
        intrinsic_reward = self.eta * (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()
        self.model.train()
        self.rnd.train()

        with torch.no_grad():
            # ------------------------------------------------------------
            # Calculate old policy
            policy_old_list = []
            for i in range(int(len(s_batch) / self.batch_size)):
                policy_old, _ = self.model(s_batch[self.batch_size * i: self.batch_size * (i + 1)])
                policy_old_list.extend(policy_old)

            policy_old_list = torch.stack(policy_old_list)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_s_batch[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                policy, value = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])
                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                self.rnd_optimizer.zero_grad()
                rnd_loss = forward_loss
                rnd_loss.backward()
                self.rnd_optimizer.step()
