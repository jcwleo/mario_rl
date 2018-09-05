import ray
import gym
import os
import random
import numpy as np


@ray.remote
class Environment(object):
    def __init__(self, env_id):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = gym.make(env_id)

    def step(self, action):
        self.env.render()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


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
