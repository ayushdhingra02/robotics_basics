import gym
import mujoco
import mujoco.viewer
import numpy as np
from gym import spaces
from PIL import Image


class HumanoidEnv(gym.Env):
    def __init__(self):
        super(HumanoidEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path('./robot.xml')
        self.data = mujoco.MjData(self.model)
        self.frame_count = 0

        # self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        n_actions = self.model.nu
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

        n_obs = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.timestep = 0
        self.feet_geom_ids = [self.model.body('LeftFoot'), self.model.body('RightFoot')]

    # def save_image(self, episode):
        # self.viewer.sync()
        # img = self.viewer.read_pixels(width=1400, height=1000, depth=False)
        # img = np.flipud(img)
        # img = Image.fromarray(img)
        # if self.timestep % 1000 == 0:
        #     img.save(f"images/episode_{episode}_frame_{self.frame_count}.png")
        #     self.frame_count += 1

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # self.viewer.sync()
        self.frame_count = 0
        data = [-3.06489210e-02, -2.73319387e-02, 1.01589137e-01, 4.20836321e-01,
                1.91209867e-01, 8.61099642e-01, 2.11761619e-01, -6.38631491e-03,
                -1.63541619e-03, -7.23161031e-01, 1.49410925e+00, -1.62802741e-03,
                -7.76672179e-06, 1.53459394e-02, -1.76933054e-01, -2.11790545e-02,
                -3.65922210e-02, -7.10615967e-02, -1.09966790e-01, -6.91756405e-03,
                -8.37033556e-04, -6.26437853e-01, -4.57133018e-02, -3.72682173e-01,
                -1.67040438e+00, -6.83312642e-01, -4.08508517e-02, 1.02621833e+00,
                -1.46028789e+00]
        self.data.qpos[:] = data
        self.timestep = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.timestep += 1

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()

        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    # @staticmethod
    def reward_func(self, x, x_hat, c):
        return -np.exp(c * ((x_hat - x) ** 2))

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
        reward = (0.2*base_pose_reward + 0.2*base_height_reward + 0.2*base_velocity_reward +
                  0.05*joint_torque_regularization + 0.05*joint_velocity_regularization +
                  0.2*body_ground_contact_reward + 0.05*upper_torso_pose_reward +0.2-
                  0.2*head_height_reward + 0.001*left_foot_placement_reward +
                  0.001*right_foot_placement_reward)

        return reward

    def _is_done(self):
        if self.timestep > 10000:
            return True
        return False

    def render(self, mode='human'):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        self.viewer.close()
