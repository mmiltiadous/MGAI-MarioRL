from datetime import datetime
import os
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation

from neat.feed_forward import FeedForwardNetwork

import pickle
import argparse
import matplotlib.pyplot as plt

from utils import DownsampleFrame

parser = argparse.ArgumentParser(description="Run a trained NEAT network")
parser.add_argument("-g", type=str, help="Genome file")
parser.add_argument("-r", type=int, default=50, help="Number of runs")
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
    genome = pickle.load(open(args.g, "rb"))
    net = FeedForwardNetwork.create_from_genome(genome)
    env = make_env()

    rewards = []
    x_positions = []
    scores = []

    for run in range(args.r):
        state = env.reset()
        done = False
        total_reward = 0
        max_x_pos = 0
        max_score = 0

        while not done:
            env.render()
            state = preprocess(state)
            output = net.activate(state)
            action = np.argmax(output)
            next_state, reward, done, info = env.step(action)
            state = next_state

            total_reward += reward
            if info["x_pos"] > max_x_pos:
                max_x_pos = info["x_pos"]
            if info["score"] > max_score:
                max_score = info["score"]

        rewards.append(total_reward)
        x_positions.append(max_x_pos)
        scores.append(max_score)

        print("Run {} finished".format(run))

    env.close()

    print(set(rewards))
    print(set(x_positions))
    print(set(scores))

    # Create evaluation directory
    eval_dir = "evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    # Generate timestamp and file path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(os.path.join(eval_dir, f"t{timestamp}"), exist_ok=True)

    eval_dir = os.path.join(eval_dir, f"t{timestamp}")
    eval_file = os.path.join(eval_dir, f"evals_t{timestamp}_r{args.r}.pkl")

    # Save evaluation data
    eval_data = {"rewards": rewards, "x_positions": x_positions, "scores": scores}
    with open(eval_file, "wb") as f:
        pickle.dump(eval_data, f)

    plot_rewards(rewards, eval_dir)
    plot_x_position(x_positions, eval_dir)
    plot_score(scores, eval_dir)


def plot_rewards(rewards, eval_dir):
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(args.r), rewards, label="Total Reward")
    plt.xlabel("Run")
    plt.ylabel("Reward")
    plt.title("Total Reward Over Runs")
    plt.savefig(os.path.join(eval_dir, f"rewards.png"))


def plot_x_position(x_positions, eval_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(args.r), x_positions, label="Max X Position")
    plt.xlabel("Run")
    plt.ylabel("X Position")
    plt.title("Max X Position Over Runs")
    plt.savefig(os.path.join(eval_dir, f"x_position.png"))


def plot_score(scores, eval_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(args.r), scores, label="Max Score")
    plt.xlabel("Run")
    plt.ylabel("Score")
    plt.title("Max Score Over Runs")
    plt.savefig(os.path.join(eval_dir, f"score.png"))


if __name__ == "__main__":
    main()
