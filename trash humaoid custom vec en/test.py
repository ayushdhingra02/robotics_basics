import gymnasium as gym
import numpy as np
from humanoid_env import HumanoidEnv
from PPO import Ppo  # Assuming your PPO class is in a file named ppo.py
import torch
from parameters import *

def make_env():
    return HumanoidEnv()


if __name__ == '__main__':
    N_S = 57  # Number of states, set according to your environment
    N_A = 6  # Number of actions, set according to your environment

    envs = gym.vector.SyncVectorEnv([make_env] * 3)  # Vectorized environments
    obs = envs.reset()

    # PPO agent initialization
    ppo_agent = Ppo(N_S, N_A)

    max_steps = MAX_STEP  # Steps per episode (or total number of steps, depending on setup)
    num_episodes = Iter  # Number of training iterations/episodes
    batch_size = 64  # Batch size for PPO training

    for episode in range(num_episodes):
        states = []
        actions = []
        rewards = []
        dones = []
        masks = []

        obs = envs.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done and step < max_steps:
            step += 1

            # Sample actions from the PPO agent
            action = [ppo_agent.actor_net.choose_action(torch.tensor(obs[i], dtype=torch.float32)) for i in
                      range(3)]

            # Take a step in the environment
            next_obs, reward, done_envs, _, infos = envs.step(action)

            # Store experience
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done_envs)
            masks.append([0 if d else 1 for d in done_envs])  # Mask for terminal states

            obs = next_obs
            episode_reward += np.mean(reward)

            # Render environments if needed
            for env in envs.envs:
                env.render()

            # If any environment is done, exit the loop
            done = any(done_envs)

        # Training step after collecting transitions
        ppo_agent.train(states, actions, rewards, masks)

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

        # Save the model every N episodes or periodically
        if (episode + 1) % 100 == 0:
            ppo_agent.save_model(episode + 1, episode_reward, ppo_agent.directory)

    envs.close()