import gymnasium as gym
import numpy as np
from humaoid_env import HumanoidEnv


def make_env():
    return HumanoidEnv()


if __name__ == '__main__':
    # envs = gym.vector.AsyncVectorEnv(
    #     [make_env] *3
    # )
    # # print(envs[0])
    # obs=envs.reset()
    # # print (obs)
    #
    # done = False
    # iteration=0
    # while not done:
    #     iteration+=1
    #     print(iteration)
    #     actions = envs.action_space.sample()  # Sample random actions
    #     obs, rewards, dones,_, infos = envs.step(actions)
    #     done = any(dones)  # If any environment is done, exit the loop
    #
    # envs.close()

    envs = gym.vector.SyncVectorEnv([make_env] * 1)  # Use SyncVectorEnv for rendering support
    obs = envs.reset()
    # print(obs)

    done = False
    while not done:
        actions = [env.action_space.sample() for env in envs.envs]  # Sample random actions
        obs, rewards, dones, _,infos = envs.step(actions)
        print("Observation shape: ", obs.shape)  # Fixed line
        done = any(dones)  # If any environment is done, exit the loop

        for env in envs.envs:
            env.render()  # Render each environment

    envs.close()