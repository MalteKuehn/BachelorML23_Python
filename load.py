import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


env = gym.make("CartPole-v1", render_mode="human")

env = Monitor(env)

PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')

model = PPO.load(PPO_path, env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=True)

vec_env = model.get_env()



obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if dones:
        print('info', info)
        break

env.close()

