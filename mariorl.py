import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import os 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

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


def preprocess():
    env = gym_super_mario_bros.make('SuperMarioBros-v3' )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env

def train(env):
    callback = TrainAndLoggingCallback(check_freq=10_000, 
                                       save_path=CHECKPOINT_DIR)
    
    model = PPO('CnnPolicy', env, 
                verbose=1, tensorboard_log=LOG_DIR, 
                learning_rate=0.000001, n_steps=10_000, device='mps')
    
    model.learn(total_timesteps=1e6, callback=callback)

def play(path: str, env):
    model = PPO.load(path)
    state = env.reset()
    while True: 
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()


def main():
    environment = preprocess()
    train(environment)


main() 