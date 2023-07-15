
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v1"

env = gym.make(environment_name, render_mode="rgb_array")

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps=20000)



PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)
del model