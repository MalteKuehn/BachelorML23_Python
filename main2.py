
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v0"

env = gym.make(environment_name)

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()



# 0-push cart to left, 1-push cart to the right
env.action_space.sample()

# [cart position, cart velocity, pole angle, pole angular velocity]
env.observation_space.sample()



env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps=20000)



PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)
del model