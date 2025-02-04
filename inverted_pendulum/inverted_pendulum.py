from stable_baselines3 import PPO
import gymnasium as gym
import mujoco
import mediapy as media
# envs = gym.make('InvertedPendulum-v4',mode="human")
# envs.reset(seed=300)

# Create the environment
env = gym.make('InvertedPendulum-v4', render_mode='human')

# Reset the environment
env.reset()

# Number of steps to run
num_steps = 1000

for step in range(num_steps):
    # Render the environment
    if step %100==0 :
        env.render()  # Use 'human' mode for visualization

    # Sample a random action (or use the policy to get an action)
    action = env.action_space.sample()

    # Perform the action
    obs, reward, done,truncated, info = env.step(action)

    # If done, reset the environment
    if done:
        obs = env.reset()

# Close the environment when done
env.close()