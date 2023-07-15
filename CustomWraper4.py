from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import numpy as np
import random
import os
from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from epyt import epanet

import matplotlib.pyplot as plt


class ShowerEnv(Env):
    def __init__(self, render_mode=None):
        # Actions we can take, down, stay, up
        # self.action_space = Discrete(3)
        self.action_space = Box(low=np.array([0]), high=np.array([5]), dtype=float)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100000]), dtype=float)
        # Set start temp
        #self.state = 38 + random.randint(-3, 3)

        self.pumpMod = 0

        # Set day length
        self.Day_time = 24

        self.net = epanet('7.inp')
        self.net.setReport("FILE energy")
        self.net.setReport("ENERGY YES")

        self.initlev = self.net.getNodeTankInitialLevel()

        self.press = self.initlev


        self.net.setLinkInitialSetting(2,self.pumpMod)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.enArx = []
        self.enAr6 = []
        self.enAry = []
        self.tankAr = []

    def _get_obs(self):
        file = open("energy").readlines()
        totalCost = ""

        for line in file:
            if 'Total Cost:' in line:
                totalCost = line.split(":")[-1].strip()

        intcost = float(totalCost)
        return intcost

    def _get_log(self):
        file = open("6.txt").readlines()
        WarningAvail = False

        for line in file:
            if 'WARNING:' in line:
                print(line)
                WarningAvail = True

        return WarningAvail

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        # Reset shower temperature
        #self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        # Reset shower time

        self.press = self.initlev

        self.pumpMod = 0
        self.net.setLinkInitialSetting(2,self.pumpMod)

        self.Day_time = 24

        #observation = self._get_obs()
        observation = np.array([self._get_obs()]).astype(float)
        info = {}

        self.enArx.clear()
        self.enAry.clear()
        self.enAr6.clear()
        self.tankAr.clear()


        return observation, info

    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        #self.state += action - 1

        # self.pumpMod += action - 1
        # if self.pumpMod < 0:
        #     self.pumpMod = 0

        self.pumpMod = action[0]

        # print(self.pumpMod)
        self.net.setLinkInitialSetting(2,self.pumpMod)

        self.net.setTimePatternStart(3600 * (24-self.Day_time))
        self.net.setNodeTankInitialLevel(3, self.press)
        #    self.net.setNodeElevations(elev)
        self.net.runsCompleteSimulation()

        self.press = self.net.getNodePressure(3)

        self.enArx.append(self.Day_time)
        self.enAry.append(self._get_obs())
        self.enAr6.append(self._get_obs())
        self.tankAr.append(self.press)

        #reduce time by 1
        self.Day_time -= 1

        #print(self.net.getStatistic().disp())

        # Calculate reward
        #if self.state >= 37 and self.state <= 39:
        # if self.press >= 3 and self.press <= 7.5 and not self._get_log():
        #
        #
        # else:
        #     reward = -50
        if self.press <= 5.15:
            reward = (80/2.15)*(self.press-3)
        else:
            reward = (-80 / 2.15) * (self.press - 7.3)

        if self._get_log():
            reward -= 100

        if len(self.enAr6) == 6:
            reward += (-20/120)*sum(self.enAr6)+10
            self.enAr6.clear()

            # Check if one day is over
        if self.Day_time <= 0:
            terminated = True
            # print(sum(self.enAry))
            if sum(self.enAry) >= 480:
                reward += -200
            else:
                reward += 100

            plt.plot(self.enAry)
            plt.xlabel('Hour')
            plt.ylabel('Energy')
            plt.show()
            plt.plot(self.tankAr)
            plt.xlabel('Hour')
            plt.ylabel('Fill')
            plt.show()

            # print(self.enAry)
            # print(self.tankAr)
            # print(sum(self.enAry))
        else:
            terminated = False

        #observation = self._get_obs()
        observation = np.array([self._get_obs()]).astype(float)
        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return observation, reward, terminated, False, info

    def render(self):
        pass


env=ShowerEnv()
#env = gym.wrappers.Monitor(env)

env.reset()

#check_env(env, warn=True)


# episodes = 5
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, trunc, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()


log_path = os.path.join('Training', 'Logs5')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
# model = PPO.load('Water5', print_system_info=False)
# model.set_env(env)
# model.learn(total_timesteps=1000000)
# model.save('Water5')

model = PPO.load('Water5', print_system_info=False)

print(evaluate_policy(model, env, n_eval_episodes=1))