import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as dist

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.action_low = -10
        self.action_high = -self.action_low


    def forward(self, obs):
        action = self.actor(obs)
        clipped_action = torch.clamp(action, self.action_low, self.action_high)
        value = self.critic(obs)
        return clipped_action, value
    
    def get_distribution(self, obs):
        action = self.actor(obs)

        # Clip action before creating the distribution (if needed)
        clipped_action = torch.clamp(action, self.action_low, self.action_high)

        dist = torch.distributions.Normal(clipped_action, 0.1)  # Adjust std as needed
        return dist


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-3, gamma=0.99, clip_ratio=0.2, epochs=10, batch_size=64):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.actor_critic = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
    
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        dist = self.actor_critic.get_distribution(obs)
        action = dist.sample()

        # Clip the action before returning
        clipped_action = torch.clamp(action, self.actor_critic.action_low, self.actor_critic.action_high)

        value = self.actor_critic.critic(obs)
        return clipped_action.numpy(), value.item()

    def compute_advantages(self, rewards, values, next_values, dones):
        returns = []
        advs = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[i])
            advs.insert(0, gae)
            returns.insert(0, gae + values[i])
        return returns, advs

    def train(self, obs, actions, advantages, returns):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        returns = torch.as_tensor(returns, dtype=torch.float32)

        for _ in range(self.epochs):
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            for i in range(0, len(obs), self.batch_size):
                idx = indices[i:i+self.batch_size]
                obs_batch = obs[idx]
                actions_batch = actions[idx]
                advantages_batch = advantages[idx]
                returns_batch = returns[idx]
                
                # Update actor
                dist = self.actor_critic.get_distribution(obs_batch)
                values = self.actor_critic.critic(obs_batch).squeeze()

                old_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                ratio = torch.exp(new_log_probs - old_log_probs)

                min_adv = torch.where(advantages_batch > 0, (1 + self.clip_ratio) * advantages_batch, (1 - self.clip_ratio) * advantages_batch)
                actor_loss = -torch.mean(torch.min(ratio * advantages_batch, min_adv))

                # Update critic
                critic_loss = torch.mean((returns_batch - values) ** 2)

                # Combine losses
                loss = actor_loss + critic_loss

                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
    def save(self, filepath):
        torch.save(self.actor_critic.state_dict(), filepath)
    
    def load(self, filepath):
        self.actor_critic.load_state_dict(torch.load(filepath))

