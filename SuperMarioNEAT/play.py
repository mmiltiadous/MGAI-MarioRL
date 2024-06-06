import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation

from neat.feed_forward import FeedForwardNetwork

import pickle

import argparse

from utils import DownsampleFrame

parser = argparse.ArgumentParser(description="Run a trained NEAT network")
parser.add_argument("-g", type=str, help="Genome file")
parser.add_argument("-c", type=str, help="Checkpoint file")
args = parser.parse_args()


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = DownsampleFrame(env, new_shape=(84, 84))
    return env


def preprocess(state):
    return state.flatten().tolist()


def main():
    if args.c:
        pop = pickle.load(open(args.c, "rb"))
        genome = pop.best_genome
    else:
        genome = pickle.load(open(args.g, "rb"))
    env = make_env()
    done = False
    state = state = env.reset()
    net = FeedForwardNetwork.create_from_genome(genome)
    while not done:
        env.render()
        state = preprocess(state)
        output = net.activate(state)
        action = np.argmax(output)
        next_state, _, done, _ = env.step(action)
        state = next_state
    env.close()


if __name__ == "__main__":
    main()
