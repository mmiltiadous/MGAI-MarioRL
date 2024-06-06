import argparse
import logging
import multiprocessing
import os
import pickle
import sys
from typing import Dict

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
import numpy as np

from neat.feed_forward import FeedForwardNetwork
from neat.genome import Genome
from neat.neat_config import NEATConfig
from neat.population import Population
from utils import DownsampleFrame, save_genome

import random

parser = argparse.ArgumentParser(description="Train NEAT")
parser.add_argument("--size", type=int, default=150, help="Population size")
parser.add_argument("--gen", type=int, default=50, help="Number of generations")
parser.add_argument("--population", type=str, help="Population file")
args = parser.parse_args()

# Create a logger
logger = logging.getLogger(__name__)
# Set the logging level
logger.setLevel(logging.DEBUG)
# Create a file handler
file_handler = logging.FileHandler(os.path.join(os.getcwd(), "log.txt"))
file_handler.setLevel(logging.DEBUG)
# Create a stream handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# Add the formatter to the handlers
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
WORKERS = 10


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = DownsampleFrame(env, new_shape=(84, 84))
    return env


def preprocess(state):
    return state.flatten().tolist()


def _fitness_func(genome: Genome, o: multiprocessing.Queue):
    env = make_env()
    try:
        net = FeedForwardNetwork.create_from_genome(genome)
        state = env.reset()
        old_fitness, fitness, i, done = 0, 0, 0, False
        while not done:
            state = preprocess(state)
            output = net.activate(state)
            action = np.argmax(output)
            next_state, reward, done, _ = env.step(action)

            fitness += reward
            state = next_state
            i += 1

            if i % 50 == 0:
                if fitness <= old_fitness:
                    break
                else:
                    old_fitness = fitness
        o.put((genome.genome_id, fitness))
        env.close()
    except KeyboardInterrupt:
        env.close()
        exit()


def eval_genomes(genomes_dict: Dict[int, Genome]):
    genomes = list(genomes_dict.values())
    for i in range(0, len(genomes), WORKERS):
        output = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(target=_fitness_func, args=(genome, output))
            for genome in genomes[i : i + WORKERS]
        ]

        [p.start() for p in processes]
        logger.debug(f"Submitted {len(processes)} processes")
        [p.join() for p in processes]
        logger.debug(f"Finished {len(processes)} processes")

        results = [output.get() for _ in processes]

        for gid, fitness in results:
            genomes_dict[gid].fitness = fitness


def main():
    random.seed(42)

    env = make_env()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    env.close()

    logger.info(f"POPULATION SIZE: {args.size}")
    logger.info(f"GENERATIONS: {args.gen}")

    config = NEATConfig(
        # Setup
        population_size=args.size,
        num_inputs=state_size,
        num_outputs=action_size,
        num_hidden=0,
        # Checkpointing
        checkpoint_frequency=5,
        # Weights
        weight_min_value=-30.0,
        weight_max_value=30.0,
        weight_init_stdev=1.0,
        weight_init_mean=0.0,
        # Bias
        bias_min_value=-30.0,
        bias_max_value=30.0,
        bias_init_stdev=1.0,
        bias_init_mean=0.0,
        # Mutation
        weight_mutate_rate=0.8,
        weight_replace_rate=0.1,
        add_link_rate=0.85,
        remove_link_rate=0.2,
        add_neuron_rate=0.35,
        remove_neuron_rate=0.1,
        weight_mutate_power=0.5,
        # Speciation
        disjoint_coeficient=2.0,
        weight_coeficient=0.5,
        compatibility_threshold=4.5,
        min_species_size=2,
        # Adjustment
        desired_species_size=5,
        compatibility_threshold_adjustment_factor=0.1,
        adjust_compatibility_threshold=False,
        # Stagnation
        max_stagnation=10,
        # Reproduction
        survival_threshold=0.25,
        elitism=2,
    )

    if args.population:
        with open(args.population, "rb") as f:
            population = pickle.load(f)
    else:
        population = Population(config=config)

    winner = population.run(eval_genomes, args.gen)
    print("Winner genome fitness: ", winner.fitness)
    population.save()
    # Save the winner genome
    save_genome(winner)


if __name__ == "__main__":
    main()
