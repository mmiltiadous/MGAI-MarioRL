import os
import time
from typing import Callable

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

MODEL_DIR = f'./models/{args.algo}_{args.log_dir}'
LOG_DIR = f'./TESTING/{args.algo}_{args.log_dir}'

if args.lr_schedule:
    lr = linear_schedule(args.lr)
else:
    lr = args.lr
if args.algo == 'ppo':

    model = PPO('MlpPolicy', env_wrap, verbose=1, learning_rate=lr, tensorboard_log=LOG_DIR, device='mps')
elif args.algo == 'sac':
    model = SAC('MlpPolicy', env_wrap, verbose=1, learning_rate=lr, tensorboard_log=LOG_DIR, device='mps') 

new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

callback = TrainAndLoggingCallback(check_freq=1000, save_path=MODEL_DIR)

t_start = time.time()
model.set_logger(new_logger)
model.learn(total_timesteps=5000, callback=callback)

t_elapsed = time.time() - t_start