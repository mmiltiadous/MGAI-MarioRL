# import os 
# from matplotlib import pyplot as plt

# from nes_py.wrappers import JoypadSpace

# import gym_super_mario_bros

# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from gym.wrappers import GrayScaleObservation, TimeLimit
# from gym.wrappers import FrameStack

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor

from nes_py.wrappers import JoypadSpace
import gym 
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY 

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import time
import matplotlib.pyplot as plt

def preprocess():
    env = gym_super_mario_bros.make('SuperMarioBros-v3' )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    # env = DummyVecEnv([lambda: env])
    # env = VecFrameStack(env, 4, channels_order='last')
    env = FrameStack(env, num_stack=4)
    # env = TimeLimit(env, max_episode_steps=10000)
    env = Monitor(env, allow_early_resets=True)

    return env

# def train(env):
#     callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
#     model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
#     model.learn(total_timesteps=1000000, callback=callback)

def play(path: str, env):

    model = PPO.load(path)
    state = env.reset()
    score = 0
    for _ in range(1000): 
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        # env.render()
        score += reward
    print(score)

def eval(environment):
    model = PPO.load('train/best_model_10000.zip')
    # environment = preprocess()
    mean_reward, std_reward = evaluate_policy(model, environment, n_eval_episodes=1, deterministic=True, render=False)
    print(mean_reward, std_reward)

def main():
    environment = preprocess()
    # eval(environment)
    play('train/best_model_10000.zip', environment)

main() 