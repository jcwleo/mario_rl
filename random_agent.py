from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
print(env.action_space.n)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    if reward == -15 or done:
        print('die')


    env.render()

env.close()