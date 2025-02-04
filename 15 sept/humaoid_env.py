import sys

import gymnasium as gym
import mujoco
from stable_baselines3 import PPO
import mujoco.viewer
import numpy as np
from gymnasium import spaces
import os
import tracemalloc
from PIL import Image

# Start tracing memory allocations




class HumanoidEnv(gym.Env):
    def __init__(self):
        super().__init__()
        xml_path = os.path.join(os.path.dirname(__file__), 'robot_standing.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_count = 0
        self.initial_pos=self.data.qpos
        self.viewer = None
        n_actions = self.model.nu
        self.action_space = spaces.Box(low=-2.356, high=2.356, shape=(n_actions,), dtype=np.float32)

        n_obs = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n_obs,), dtype=np.float32)
        self._state = ""
        self.timestep = 0
        self.feet_geom_ids = [self.model.body('LeftFoot'), self.model.body('RightFoot')]

    # def save_image(self, episode):
    #     self.viewer.sync()
    #     # img = self.viewer.read_pixels(width=1400, height=1000, depth=False)
    #     # img = np.flipud(img)
    #     # img = Image.fromarray(img)
    #     # if self.timestep % 1000 == 0:
    #     #     img.save(f"images/episode_{episode}_frame_{self.frame_count}.png")
    #     #     self.frame_count += 1

    def reset(self, seed=None, option=None):
        mujoco.mj_resetData(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        self.frame_count = 0
        self.data.qpos[:] = self.initial_pos
        self.timestep = 0
        obs = self._get_obs()
        self._state=obs
        info={}
        return self._state,info

    def step(self, action):
        action= np.clip(action, -2.356,2.356)
        torques = self._compute_torques(action)
        self.data.ctrl[:] = torques
        mujoco.mj_step(self.model, self.data)
        self.timestep += 1
        if self.viewer:
            self.viewer.sync()
        self._state = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()

        return self._state, reward, done,False, {}

    def _compute_torques(self,action):
        Kp = 2.0  # Proportional gain
        Kd = 1.0  # Derivative gain

        # Indices for qpos and qvel as given in your description
        qpos_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        qvel_indices = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

        # Compute torques for each actuator
        torques = []
        for i in range(len(qpos_indices)):
            # Get the current position (qpos) and velocity (qvel) of the joint
            current_pos = self.data.qpos[qpos_indices[i]]
            current_vel = self.data.qvel[qvel_indices[i]]

            # Desired position from the action vector
            desired_pos = action[i]

            # Previous position from prev_action
            # prev_pos = prev_action[i]

            # PD control: compute position error and apply proportional-derivative control
            position_error = desired_pos - current_pos
            torque = Kp * position_error - Kd * current_vel

            torques.append(torque)

        torques= np.clip(torques, -.5, .5)
        return torques

    def get_geom_name(self,geom_id):
        """Retrieve the name of a geometry from its ID."""
        return self.model.geom_names[geom_id]

    def _get_obs(self):
        # print ("qpos number=" ,len(self.data.qpos))
        # print ("qvel number=" ,len(self.data.qvel))
        collision_info = self.data.contact

        # for contact in collision_info:
        #     geom1_id = contact.geom1
        #     geom2_id = contact.geom2
        #
        #     # Get the names of the colliding geometries
        #     geom1_name = self.model.geom(geom1_id).name
        #     geom2_name = self.model.geom(geom2_id).name
        #
        #     # Print the colliding parts
        #     if not (geom2_name) or  not (geom2_name):
        #         # return True
        #         print(f"Collision detected between: {geom1_name} and {geom2_name}")
        return np.concatenate([self.data.qpos, self.data.qvel])

    # @staticmethod
    def reward_func(self, x, x_hat, c):
        return np.exp(c * (x_hat - x) ** 2)

    def _get_reward(self):
        # Weights
        w_i = 1 / 17

        # Terms for the reward function
        phi_base = np.array([0, 0, -1])
        h_base = np.array([0, 0, -1])
        v_base = np.array([0, 0, 0])
        tau = self.data.qfrc_actuator
        tau_hat = np.zeros_like(tau)
        q_i = self.data.qvel
        q_dot_hat = np.zeros_like(q_i)
        weights = [1/17, 4/17, 4/17, 1/17, 4/17, 1/17, 4/17, 1/17, 4/17, 4/17]
        normalization = [-1.02, -12.5, -2, -0.031, -0.109, 1, -1.02, -5.556, -16.33, -16.33]
        torso_pose = np.array([0, 0, -1])
        head_height = np.array([0, 0, 0.36])  # Assuming head height should be around 1 when standing
        body_ground_contact = 0 if any(contact.geom1 == 'ground' for contact in self.data.contact) else 1

        # Compute the reward terms
        # base_pose_reward = -w_i * np.linalg.norm(phi_base - self.data.qpos[0:3])
        # base_height_reward = -w_i * np.linalg.norm(h_base - self.data.qpos[2])
        # base_velocity_reward = -w_i * np.linalg.norm(v_base - self.data.qvel[0:3])
        # joint_torque_regularization = -w_i * np.linalg.norm(tau)
        #
        # joint_velocity_regularization = -w_i * np.linalg.norm(q_i)
        #
        # body_ground_contact_reward = -w_i * body_ground_contact
        #
        # upper_torso_pose_reward = -w_i * np.linalg.norm(torso_pose - self.data.qpos[3:6])
        # head_height_reward = -w_i * np.linalg.norm(head_height - self.data.qpos[2])

        left_foot_placement_reward = -w_i * np.linalg.norm(
            self.data.site_xpos[self.model.site('left_foot_site').id] - self.data.qpos[0:3])
        right_foot_placement_reward = -w_i * np.linalg.norm(
            self.data.site_xpos[self.model.site('right_foot_site').id] - self.data.qpos[0:3])


        base_pose_reward=weights[0]*np.sum(self.reward_func(self.data.qpos[0:3], phi_base, normalization[0]))
        base_height_reward=weights[1]*np.sum(self.reward_func(self.data.qpos[2], h_base, normalization[1]))
        base_velocity_reward=weights[2]*np.sum(self.reward_func(self.data.qvel[0:3], v_base, normalization[2]))
        joint_torque_regularization=weights[3]*np.sum(self.reward_func(tau, tau_hat, normalization[3]))
        joint_velocity_regularization=weights[4]*np.sum(self.reward_func(q_i, q_dot_hat, normalization[4]))
        body_ground_contact_reward = weights[5] * body_ground_contact
        upper_torso_pose_reward=weights[6]*np.sum(self.reward_func(self.data.qpos[3:6], torso_pose, normalization[6]))
        head_height_reward=weights[7]*np.sum(self.reward_func(self.data.qpos[2], head_height, normalization[7]))

        # Sum of all rewards
        reward = (base_pose_reward + base_height_reward + base_velocity_reward +
                  joint_torque_regularization + joint_velocity_regularization +
                  body_ground_contact_reward + upper_torso_pose_reward +
                  head_height_reward + left_foot_placement_reward +
                  right_foot_placement_reward)

        return reward

    def _is_done(self):
        if self.timestep > 10000:
            return True
        return False

    def render(self, mode='human'):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()

    def __call__(self):
        return self


total_timesteps=""
env= HumanoidEnv()


def get_total_size(obj, seen=None):
    """Recursively find the memory footprint of a Python object and all its attributes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Recursively add size of attributes, items (if iterable), and objects referenced by obj
    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])

    return size


# Example usage
# total_size_before_reset = get_total_size(env)
# print(f"Total size of env: {total_size_before_reset} bytes")
# env.reset()
# total_size_after_reset = get_total_size(env)
# print(f"Total size of env: {total_size_after_reset} bytes")

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=total_timesteps)
# model.save("humanoid_sbPPO {total_timesteps} ")
# del model

model = PPO.load(f"humanoid_sbPPO{total_timesteps}")
tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()
print(f"Memory after reset: {snapshot1.statistics('filename')[0].size / 1024} KB")
env=HumanoidEnv()
obs,_ = env.reset()
snapshot1 = tracemalloc.take_snapshot()
print(f"Memory after reset: {snapshot1.statistics('filename')[0].size / 1024} KB")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones,_, info = env.step(action)
    snapshot1 = tracemalloc.take_snapshot()
    print(f"Memory after reset: {snapshot1.statistics('filename')[0].size / 1024} KB")
    # env.render("human")