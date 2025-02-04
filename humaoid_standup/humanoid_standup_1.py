from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("HumanoidStandup-v4", n_envs=4)

# Load the existing model if it exists
try:
    model = PPO.load("ppo_humanoid_standup", env=vec_env)  # Load the model and specify the environment
    print("Model loaded. Continuing training...")
except FileNotFoundError:
    # If the model doesn't exist, create a new one
    model = PPO("MlpPolicy", vec_env, verbose=1)
    print("No existing model found. Starting new training...")

# Continue training the model
model.learn(total_timesteps=5000000)
model.save("ppo_humanoid_standup")

# Demonstrate saving and loading
del model  # Remove to demonstrate saving and loading

model = PPO.load("ppo_humanoid_standup", env=vec_env)  # Make sure to load the environment as well

# Run the model
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")