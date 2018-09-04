from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from multiprocessing import Process, Pool
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
from multiprocessing import Process
import time

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
print(env.action_space.n)


class ParallelEnvironment(object):
    def __init__(self, num_env, num_worker, env_id):
        self.num_env = num_env
        self.num_worker = num_worker
        self.envs = [self._make_env(env_id) for _ in range(num_env)]

    def _make_env(self, env_id):
        env = gym_super_mario_bros.make(env_id)
        env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
        return env

    def step(self, actions):
        obss = []
        rewards = []
        dones = []
        infos = []

        with Pool(processes=self.num_worker) as pool:
            pool_results = [pool.apply_async(env.step, action) for env, action in zip(self.envs, actions)]
            results = [res.get() for res in pool_results]

    def _step(self):

    def _reset(self):
        pass



def run(env, idx):
    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        # if step % 2 ==0:
        #     state, reward, done, info = env.step(1)
        # else:
        #     state, reward, done, info = env.step(0)

        state, reward, done, info = env.step(env.action_space.sample())

        # print(reward)
        # if step % 100 == 0:
        #     env.reset()
        if reward == -15 or done:
            print('die')

        env.render()

    env.close()


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    return env

def main():
    envs = [make_env() for _ in range(2)]

    procs = []
    for idx,env in enumerate(envs):
        proc = Process(target=run, args=(env,idx))
        time.sleep(1)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()

