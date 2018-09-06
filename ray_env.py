import ray
import gym
import os
import random
import numpy as np
import copy

import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

@ray.remote
class Environment(object):
    def __init__(self, env_id):
        os.environ["MKL_NUM_THREADS"] = "1"
        env = gym_super_mario_bros.make('SuperMarioBros-v2')
        env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
        self.env = env

    def step(self, action):
        self.env.render()
        return copy.deepcopy(self.env.step(action))

    def reset(self):
        return copy.deepcopy(self.env.reset())


if __name__ == '__main__':
    env_id = 'Pong-v0'
    num_env = 2
    ray.init()
    envs = [Environment.remote(env_id) for _ in range(num_env)]

    result = ray.get([env.reset.remote() for env in envs])

    for _ in range(1000):
        actions = [2] * num_env
        actions[0] = np.random.randint(6)

        result = ray.get([env.step.remote(action) for env, action in zip(envs, actions)])
        # print(result)
