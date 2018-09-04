from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from multiprocessing import Process
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
import time

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
print(env.action_space.n)


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

