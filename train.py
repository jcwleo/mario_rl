import gym

from model import *
from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe, Process

from collections import deque

from tensorboardX import SummaryWriter

import numpy as np


def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env = gym.make(env_id)
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    icm_path = 'models/{}.icm'.format(env_id)

    writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_standardization = default_config.getboolean('UseNorm')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])
    if train_method in ['PPO', 'ICM']:
        num_step = int(ppo_config['NumStep'])
    else:
        num_step = int(default_config['NumStep'])

    ppo_eps = float(ppo_config['PPOEps'])
    epoch = int(ppo_config['Epoch'])
    batch_size = int(ppo_config['BatchSize'])
    learning_rate = float(default_config['LearningRate'])
    stable_eps = float(default_config['StableEps'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    beta = float(icm_config['beta'])
    icm_scale = float(icm_config['ICMScale'])
    eta = float(icm_config['ETA'])

    if train_method == 'PPO':
        agent = PPOAgent
    elif train_method == 'ICM':
        agent = ICMAgent
    else:
        agent = A2CAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        beta=beta,
        icm_scale=icm_scale,
        eta=eta,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        train_method=train_method
    )

    if is_load_model:
        agent.model.load_state_dict(torch.load(model_path))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    while True:
        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            actions = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones = [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            # total reward = int reward + ext Resard
            if train_method == 'ICM':
                intrinsic_reward = agent.compute_intrinsic_reward(
                    states, next_states, actions)
                rewards += intrinsic_reward
                sample_i_rall += intrinsic_reward[sample_env_idx]

            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            states = next_states[:, :, :, :]

            sample_rall += rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                if train_method == 'ICM':
                    writer.add_scalar('data/int_reward', sample_i_rall, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        total_state = np.stack(total_state).transpose(
            [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().reshape([-1])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose().reshape([-1])

        value, next_value, policy = agent.forward_transition(
            total_state, total_next_state)

        policy = policy.detach()
        m = F.softmax(policy, dim=-1)
        recent_prob.append(m.max(1)[0].mean().cpu().numpy())
        writer.add_scalar('data/max_prob', np.mean(recent_prob), sample_episode)

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

        if use_standardization:
            total_adv = (total_adv - np.mean(total_adv)) / (np.std(total_adv) + stable_eps)

        agent.train_model(total_state, total_next_state, np.hstack(total_target), total_action, np.hstack(total_adv))

        if global_step % (num_worker * num_step * 100) == 0:
            torch.save(agent.model.state_dict(), model_path)
            if train_method == 'ICM':
                torch.save(agent.icm.state_dict(), icm_path)


if __name__ == '__main__':
    main()
