

import os
import time
from typing import Callable
from itertools import product

import numpy as np

from nes_py.wrappers import JoypadSpace
import gym 
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY 

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure



from gym_utils import SMBRamWrapper
from arguments import get_args

args = get_args()
print(args)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

x0 = 0
x1 = 16
y0 = 0
y1 = 13
n_stack = 4
n_skip = 4

env_wrap = SMBRamWrapper(env, [x0, x1, y0, y1], n_stack=n_stack, n_skip=n_skip)

env_wrap = Monitor(env_wrap)  # for tensorboard log
env_wrap = DummyVecEnv([lambda: env_wrap])

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            episode_rewards = np.array(self.locals["rewards"])
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                self.logger.record('rollout/ep_rew_mean', mean_reward)
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


lr = [0.0003, 0.00003]
gae_lambda = [0.9, 0.95, 0.99]
clip_range = [0.2, 0.1, 0.5]
lin_schedule = [True, False]

# Create the parameter grid
param_grid = {
    'lr': lr,
    'gae_lambda': gae_lambda,
    'clip_range': clip_range,
    'linear_schedule': lin_schedule
}


product_list = list(product(*param_grid.values()))

t_start = time.time()

for l, gae, clip, lin in product_list:
    # print(l, gae, clip, lin)
    if lin:
        log_dir = f'./gridsearch/ppo_linear_{l}_{gae}_{clip}'
        model_dir = f'./gridsearch_model/ppo_linear_{l}_{gae}_{clip}'
    else:
        log_dir = f'./gridsearch/ppo_{l}_{gae}_{clip}'
        model_dir = f'./gridsearch_model/ppo_{l}_{gae}_{clip}'
    print(log_dir)
    # print(type(l), type(gae), type(clip), type(lin))
    if lin:
        model = PPO('MlpPolicy', env_wrap, verbose=0, learning_rate=linear_schedule(float(l)),
                    gae_lambda=gae,clip_range=clip,
                    tensorboard_log=log_dir, device='mps') 


    else:

        model = PPO('MlpPolicy', env_wrap, verbose=0, learning_rate=l,
            gae_lambda=gae,clip_range=clip,
            tensorboard_log=log_dir, device='mps') 


    callback = TrainAndLoggingCallback(check_freq=100, save_path=model_dir)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model.set_logger(new_logger)

    model.learn(total_timesteps=100_000, callback=callback)

t_elapsed = time.time() - t_start
print(t_elapsed)