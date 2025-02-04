from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("Hopper-v4", n_envs=4)



model = PPO.load("ppo_hopper", env=vec_env)  # Make sure to load the environment as well

# Run the model
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")