import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Define the PPO Agent
class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2, epochs=10, batch_size=64):
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size

    def select_action(self, obs):
        # Ensure obs is a numpy array
        if isinstance(obs, list):
            obs = np.array(obs)
        print("Observation shape:", obs.shape)  # Debugging statement
        # Convert to torch tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy(obs)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_returns(self, rewards, dones, next_value, gamma):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, obs, actions, log_probs_old, returns):
        for _ in range(self.epochs):
            new_log_probs = []
            for ob, act in zip(obs, actions):
                ob = torch.tensor(ob, dtype=torch.float32)
                m = Categorical(self.policy(ob))
                new_log_probs.append(m.log_prob(torch.tensor(act)))

            new_log_probs = torch.stack(new_log_probs)
            ratios = torch.exp(new_log_probs - log_probs_old)
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * returns
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Logger Class for Displaying Metrics
class Logger:
    def __init__(self):
        self.data = {}

    def log(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def get_mean(self, key):
        return np.mean(self.data[key])

    def display(self, iteration, total_timesteps, fps):
        print(f"------------------------------------------")
        print(f"| rollout/                |              |")
        print(f"|    ep_len_mean          | {self.get_mean('ep_len'):.0f}         |")
        print(f"|    ep_rew_mean          | {self.get_mean('ep_rew'):.0f}         |")
        print(f"| time/                   |              |")
        print(f"|    fps                  | {fps:.0f}          |")
        print(f"|    iterations           | {iteration}           |")
        print(f"|    time_elapsed         | {int(time.time() - self.start_time)}          |")
        print(f"|    total_timesteps      | {total_timesteps}       |")
        print(f"| train/                  |              |")
        print(f"|    approx_kl            | {self.get_mean('approx_kl'):.7f} |")
        print(f"|    clip_fraction        | {self.get_mean('clip_fraction'):.4f}       |")
        print(f"|    clip_range           | 0.2          |")
        print(f"|    entropy_loss         | {self.get_mean('entropy_loss'):.3f}       |")
        print(f"|    explained_variance   | {self.get_mean('explained_variance'):.4f}       |")
        print(f"|    learning_rate        | 0.0003       |")
        print(f"|    loss                 | {self.get_mean('loss'):.5f}      |")
        print(f"|    n_updates            | {iteration * agent.epochs}          |")
        print(f"|    policy_gradient_loss | {self.get_mean('policy_gradient_loss'):.4f}      |")
        print(f"|    value_loss           | {self.get_mean('value_loss'):.6f}     |")
        print(f"------------------------------------------")

    def start_timer(self):
        self.start_time = time.time()

# Create environment and initialize the PPO agent and logger
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = PPOAgent(obs_dim, act_dim)
logger = Logger()
logger.start_timer()

max_timesteps = 250000
timesteps = 0
obs = env.reset()
episode_rewards = []
episode_lengths = []
iteration = 0

# Training Loop
while timesteps < max_timesteps:
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    while len(observations) < agent.batch_size:
        action, log_prob = agent.select_action(obs)
        new_obs, reward, done, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        obs = new_obs
        timesteps += 1

        if done:
            obs = env.reset()
            episode_rewards.append(sum(rewards))
            episode_lengths.append(len(rewards))

    next_value = 0 if done else agent.select_action(obs)[0]
    returns = agent.compute_returns(rewards, dones, next_value, agent.gamma)

    # Update the policy network
    agent.update(observations, actions, torch.stack(log_probs), returns)

    # Log training statistics
    logger.log('ep_len', np.mean(episode_lengths))
    logger.log('ep_rew', np.mean(episode_rewards))
    logger.log('approx_kl', 0.0049190093)  # Replace with actual computed value
    logger.log('clip_fraction', 0.0491)  # Replace with actual computed value
    logger.log('entropy_loss', -0.475)  # Replace with actual computed value
    logger.log('explained_variance', 0.0125)  # Replace with actual computed value
    logger.log('loss', 0.00227)  # Replace with actual computed value
    logger.log('policy_gradient_loss', -0.0029)  # Replace with actual computed value
    logger.log('value_loss', 0.000317)  # Replace with actual computed value

    if timesteps >= 8192 * (iteration + 1):  # Log every 8192 timesteps
        fps = timesteps / (time.time() - logger.start_time)
        logger.display(iteration + 1, timesteps, fps)
        iteration += 1
        episode_rewards = []
        episode_lengths = []

env.close()