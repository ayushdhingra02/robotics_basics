import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class DoubleInvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(DoubleInvertedPendulumEnv, self).__init__()

        # Constants
        self.gravity = -9.8
        self.mass_cart = 1.0
        self.mass_pole1 = 0.1
        self.mass_pole2 = 0.1
        self.length_pole1 = 0.5  # Half the length of pole 1
        self.length_pole2 = 0.5  # Half the length of pole 2

        self.tau = 0.02  # Time step size

        # State: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        self.state = None

        # Action space: continuous force applied to the cart
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # Observation space: positions and velocities of the cart and two pendulums
        high = np.array([np.inf, np.inf, np.pi, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.fig, self.ax = None, None
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.pole1_length = 2 * self.length_pole1
        self.pole2_length = 2 * self.length_pole2
        self.cart_y = self.cart_height / 2  # Y position of the cart

    def reset(self):
        # Initialize state: small random angles and zero velocities
        self.state = np.array([0, 0, 3.14+np.random.uniform(-0.05, 0.05), 0, 3.14+np.random.uniform(-0.05, 0.05), 0])
        return self.state

    def step(self, action):
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        force = np.clip(action, -10.0, 10.0)[0]  # Clip the force to the allowed action space

        # Dynamics equations (simplified, Euler integration):
        # Constants
        total_mass = self.mass_cart + self.mass_pole1 + self.mass_pole2
        pole1_inertia = self.mass_pole1 * self.length_pole1 ** 2
        pole2_inertia = self.mass_pole2 * self.length_pole2 ** 2

        # Update theta1, theta2 (pendulum angles) using simplified dynamics
        # In a real system, you'd solve the full nonlinear dynamics (e.g., with Lagrangian mechanics)
        theta1_acc = (self.gravity / self.length_pole1) * np.sin(theta1)
        theta2_acc = (self.gravity / self.length_pole2) * np.sin(theta2)

        # Update the cart position and velocities using a simple cart model
        x_acc = (force + theta1_acc + theta2_acc) / total_mass

        # Euler integration for updating the state
        x_dot += x_acc * self.tau
        x += x_dot * self.tau
        theta1_dot += theta1_acc * self.tau
        theta1 += theta1_dot * self.tau
        theta2_dot += theta2_acc * self.tau
        theta2 += theta2_dot * self.tau

        self.state = np.array([x, x_dot, theta1, theta1_dot, theta2, theta2_dot])

        # Calculate reward: Upright bonus and penalties for deviation
        reward = self._get_reward(self.state, action)

        x_threshold = 2.4  # Cart can only move within [-2.4, 2.4]
        theta_threshold_radians = np.pi / 4  # 45 degrees in both directions from upright (π radians)

        # Conditions for ending the episode
        done = bool(
            x < -x_threshold or x > x_threshold or  # Cart out of bounds
            abs(theta1 - np.pi) > theta_threshold_radians or  # Pendulum 1 too far from upright
            abs(theta2 - np.pi) > theta_threshold_radians  # Pendulum 2 too far from upright
        )

        return self.state, reward, done, {}

    # def calculate_reward(self, state, action):
    #     x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
    #
    #     # Define thresholds
    #     theta_threshold_radians = np.pi / 4  # 45 degrees from upright
    #
    #     # Reward for staying upright
    #     upright_penalty = - (abs(theta1 - np.pi) + abs(theta2 - np.pi))  # closer to π is better
    #
    #     # Reward for staying close to the center
    #     center_penalty = - (abs(x) / 2.4)  # penalizing distance from the center position
    #
    #     # Combined reward
    #     reward = upright_penalty + center_penalty
    #
    #     # Check done conditions to apply negative rewards for falling
    #     if abs(theta1 - np.pi) > theta_threshold_radians or abs(theta2 - np.pi) > theta_threshold_radians:
    #         reward -= 10  # Large penalty for falling
    #     if x < -2.4 or x > 2.4:
    #         reward -= 10  # Large penalty for cart out of bounds
    #
    #     # Optional: small negative reward for each step to encourage efficiency
    #     reward -= 0.1  # Small penalty for each time step
    #
    #     return reward
    def _get_reward(self, state, action):
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state

        # Penalty for angle deviation (both pendulums)
        angle_penalty = -((theta1 - np.pi) ** 2 + (theta2 - np.pi) ** 2)

        # Penalty for velocity (to avoid erratic movements)
        velocity_penalty = -(theta1_dot ** 2 + theta2_dot ** 2 + x_dot ** 2)

        # Penalty for large actions (control effort)
        action_penalty = -action ** 2

        # Bonus for staying upright
        upright_bonus = +1 if abs((theta1 - np.pi)) < 0.1 or abs((theta2 - np.pi)) < 0.1 else 0

        reward = upright_bonus + angle_penalty + velocity_penalty + action_penalty
        return reward

    def render(self, mode='human'):
        # Render the environment (visualization logic, skipped for simplicity)
        if self.fig is None:
            # Initialize the plot
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-3, 3)
            self.ax.set_ylim(-2, 2)
            self.ax.set_aspect('equal')
            self.ax.set_title('Double Inverted Pendulum')

            # Clear the axis to redraw
        self.ax.clear()

        # Extract the state
        x = self.state[0]
        theta1 = self.state[2]
        theta2 = self.state[4]

        # Cart coordinates
        cart = plt.Rectangle((x - self.cart_width / 2, self.cart_y - self.cart_height / 2),
                             self.cart_width, self.cart_height, color='black')

        # Pendulum 1 (from the cart's center)
        pole1_x = x + self.pole1_length * np.sin(theta1)
        pole1_y = self.cart_y - self.pole1_length * np.cos(theta1)

        # Pendulum 2 (from the end of pendulum 1)
        pole2_x = pole1_x + self.pole2_length * np.sin(theta2)
        pole2_y = pole1_y - self.pole2_length * np.cos(theta2)

        # Draw the cart
        self.ax.add_patch(cart)

        # Draw pendulum 1 (line from the cart to pole 1 end)
        self.ax.plot([x, pole1_x], [self.cart_y, pole1_y], color='blue', linewidth=2)

        # Draw pendulum 2 (line from the pole 1 end to pole 2 end)
        self.ax.plot([pole1_x, pole2_x], [pole1_y, pole2_y], color='red', linewidth=2)

        # Draw the pivot point (cart's center)
        self.ax.plot(x, self.cart_y, 'ko')  # Draw the cart pivot as a black circle

        # Draw pole end points for visualization
        self.ax.plot(pole1_x, pole1_y, 'bo')  # Pole 1 end as blue circle
        self.ax.plot(pole2_x, pole2_y, 'ro')  # Pole 2 end as red circle

        # Update the plot
        plt.pause(0.01)  # Small pause to make the rendering real-time
        # pass/

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        # pass

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


if __name__=="__main__":
    # env = gym.make("InvertedDoublePendulum-v4")
    # # obs = env.reset()
    #
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=250000)
    # model.save("ppo_cartpole")
    #
    # del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")
    env = gym.make("InvertedDoublePendulum-v4")
    obs,_ = env.reset()
    for _ in range(1000):
        action = model.predict(obs)  # For testing, sample random actions
        obs, reward, done, _ ,_= env.step(action)
        env.render()  # This will show the real-time animation
        if done:
            obs,_ = env.reset()

    env.close()