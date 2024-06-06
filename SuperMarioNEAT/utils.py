import os
import pickle
import time
from neat.genome import Genome
import gym
import cv2
import numpy as np

class DownsampleFrame(gym.ObservationWrapper):
    def __init__(self, env, new_shape=(84, 84)):
        super(DownsampleFrame, self).__init__(env)
        self.new_shape = new_shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.new_shape[0], self.new_shape[1], 1), dtype=np.uint8)

    def observation(self, obs):
        return self.downsample(obs)

    def downsample(self, frame):
        # Resize the frame
        resized_frame = cv2.resize(frame, self.new_shape, interpolation=cv2.INTER_AREA)
        # Add channel dimension
        processed_frame = np.expand_dims(resized_frame, axis=-1)
        return processed_frame

def save_genome(genome: Genome):
    os.makedirs('models', exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join('models', f'winner_f{round(genome.fitness)}_t{timestamp}.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(genome, f)
