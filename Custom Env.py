
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


class ShowerEnv(Env):
    def __init__(self, render_mode=None):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=float)
        # Set start temp
        self.state = 38 + random.randint(-3, 3)
        # Set shower length
        self.shower_length = 60

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.state

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        # Reset shower temperature
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        # Reset shower time
        self.shower_length = 60

        observation = self.state
        info = {}

        return observation, info

    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action - 1
        # Reduce shower length by 1 second
        self.shower_length -= 1

        # Calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

            # Check if shower is done
        if self.shower_length <= 0:
            terminated = True
        else:
            terminated = False

        observation = self.state
        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return observation, reward, terminated, False, info

    def render(self):
        # Implement viz
        pass





env=ShowerEnv()

env.reset()

check_env(env, warn=True)


episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, trunc, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()




log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=400000)
model.save('PPO')

print(evaluate_policy(model, env, n_eval_episodes=10))