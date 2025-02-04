import gymnasium as gym
import torch
import numpy as np
import mujoco
# import argparse
from parameters import *
from PPO import Ppo
from collections import deque
# parser = argparse.ArgumentParser()
# parser.add_argument('--env_name', type=str, default="Humanoid-v2",
#                     help='name of Mujoco environement')
# args = parser.parse_args()


env = gym.make('Hopper-v4',render_mode='human')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# envs.seed(500)
seed=500
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
episodes = 0
eva_episodes = 0
for iter in range(Iter):
    memory = deque()
    ppo_s=[]
    ppo_s = []
    ppo_a = []
    ppo_r = []
    ppo_m = []
    scores = []
    steps = 0
    while steps <2048:
        episodes += 1
        s, info = env.reset(seed=seed)
        s=nomalize(s)
        score = 0
        for _ in range(MAX_STEP):
            steps += 1

            a=ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
            s_ , r ,done,truncated,info = env.step(a)
            s_ = nomalize(s_)

            mask = (1-done)*1
            memory.append([s,a,r,mask])
            ppo_s.append(s)
            ppo_a .append(a)
            ppo_r .append(r)
            ppo_m .append(mask)

            score += r
            s = s_
            if done:
                break
        with open('log_' + "Humaonoid" + '.txt', 'a') as outfile:
            outfile.write('\t' + str(episodes)  + '\t' + str(score) + '\n')
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(episodes, score_avg))

    ppo.train(ppo_s,ppo_a,ppo_r,ppo_m)
