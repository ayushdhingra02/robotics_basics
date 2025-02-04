import numpy as np
from humanoid import HumanoidEnv
import gymnasium as gym
from ppo import PPOAgent
import os

def train_ppo(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        obs = env.reset()
        # env.render()
        done = False
        ep_rewards = []
        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        adv_buf = []
        ret_buf = []
        next_val = 0

        while not done:
            action, value = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            _, next_val = agent.act(next_obs)

            # Check for NaN values in observations, actions, rewards, and values
            if np.isnan(obs).any() or np.isnan(action).any() or np.isnan(reward) or np.isnan(value) or np.isnan(next_val):
                print(f"NaN detected in episode {episode}, skipping this episode")
                break  # Use break instead of continue to ensure the episode ends properly

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)

            obs = next_obs
            ep_rewards.append(reward)

            # if episode % 10 == 0 and not done:
            #     env.save_image(episode)

            if done:
                next_val = 0

        if done:
            returns, advantages = agent.compute_advantages(rew_buf, val_buf, [next_val] * len(rew_buf), [done] * len(rew_buf))
            obs_buf = np.array(obs_buf)
            act_buf = np.array(act_buf)
            adv_buf = np.array(advantages)
            ret_buf = np.array(returns)

            agent.train(obs_buf, act_buf, adv_buf, ret_buf)
            total_reward = np.sum(ep_rewards)
            print(f'Episode {episode}, Total Reward: {total_reward}')
            # env.close()

            # Save the model every 100 episodes
            if (episode + 1) % 100 == 0:
                agent.save(f'weights/ppo_agent_{episode+1}.pth')

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists('weights'):
        os.makedirs('weights')

    env = HumanoidEnv()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    agent = PPOAgent(obs_dim, act_dim)
    train_ppo(env, agent)
    # Save the final model
    agent.save('weights/ppo_agent_final.pth')
