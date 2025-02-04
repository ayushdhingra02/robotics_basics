import gymnasium as gym
import torch
import numpy as np
from gym.wrappers import RecordVideo
import mujoco
# import argparse
from parameters import *
from PPO import Ppo
from collections import deque
# parser = argparse.ArgumentParser()
# parser.add_argument('--env_name', type=str, default="Humanoid-v2",
#                     help='name of Mujoco environement')
# args = parser.parse_args()

# rgb_array
env = gym.make('Hopper-v4', render_mode='human')
# envs = RecordVideo(envs, 'video', episode_trigger=lambda x: x%1000==0,disable_logger=True)
# envs.reset(seed=1)
# envs.start_video_recorder()
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
# print (parameters.)
# envs.seed(500)
env.reset(seed=500)
torch.manual_seed(500)
np.random.seed(500)

class Nomalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S, ))
        self.stdd = np.zeros((N_S, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:

            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)

        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean

        x = x / (self.std + 1e-8)

        x = np.clip(x, -5, +5)


        return x



ppo = Ppo(N_S,N_A)
nomalize = Nomalize(N_S)
directory="models"
iter=2800
episodes=38881
ppo.load_model(iter,episodes,directory)
s, info = env.reset(seed=500)
s=nomalize(s)
while True:
    a = ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
    s_, r, done, truncated, info = env.step(a)
    s_ = nomalize(s_)
    s=s_
    if done:
       s,_= env.reset()
       s=nomalize(s)