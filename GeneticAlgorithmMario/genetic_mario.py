import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import random
import os
import torch
import csv
import time
import  random 

# Constants
ELITISM_COUNT = 3
FRAME_COUNT = 4500
GENERATION_SIZE = 50 
POPULATION_SIZE = 50
TOURNAMENT_SIZE = 20
JUMP_WEIGHT = 0.95
MUTATION_RATE_0 = 0.05
MUTATION_RATE_1 = 0.15
MODELS_DIR = "models/" #Change for testing other levels except 1-1
LOGS_DIR = "logs/"
LOG_FILE = os.path.join(LOGS_DIR, "training_log.csv")
# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class Individual:
    def __init__(self):
        self.bytes = np.random.choice([0, 1], size=(FRAME_COUNT,), p=[1-JUMP_WEIGHT, JUMP_WEIGHT])
        self.fitness = 0
        self.frameNumber = 0
        self.score = 0
        self.x_pos = 0
        self.flag = False

def generateRandomPopulation():
    return [Individual() for _ in range(POPULATION_SIZE)]

def tournamentSelection(population):
    selected = random.sample(population, TOURNAMENT_SIZE)
    best = max(selected, key=lambda indiv: (indiv.fitness, -indiv.frameNumber))
    return best


def crossover(indiv1: Individual, indiv2: Individual) -> Individual: #two point
    point1, point2 = sorted(random.sample(range(FRAME_COUNT), 2))
    child = Individual()
    child.bytes = np.concatenate((indiv1.bytes[:point1], indiv2.bytes[point1:point2], indiv1.bytes[point2:]))
    return child


def mutate(indiv):
    new_bytes = []
    for byte in indiv.bytes:
        if byte == 1 and random.random() < MUTATION_RATE_1:
            new_bytes.append(0)
        elif byte == 0 and random.random() < MUTATION_RATE_0:
            new_bytes.append(1)
        else:
            new_bytes.append(byte)
    indiv.bytes = np.array(new_bytes)


def evolvePopulation(population):
    new_population = []

    # Keep the top individuals (elitism)
    sorted_population = sorted(population, key=lambda indiv: (indiv.fitness, -indiv.frameNumber), reverse=True)
    new_population.extend(sorted_population[:ELITISM_COUNT])

    # Create new individuals
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournamentSelection(population)
        parent2 = tournamentSelection(population)
        child = crossover(parent1, parent2)  # two-point crossover
        mutate(child)  
        new_population.append(child)

    return new_population


def play_game(indiv, env, render=False):
    obs = env.reset()
    indiv.fitness = 0
    indiv.frameNumber = 0
    indiv.score = 0
    indiv.x_pos = 0
    indiv.flag = False
    moveIndex = 0
    total_reward = 0

    for frame in range(FRAME_COUNT):
        if moveIndex >= FRAME_COUNT:
            break

        # Decode the action from binary representation
        action_index = indiv.bytes[moveIndex]

        # Map action_index to action
        if action_index == 0:
            action = 0  # Do nothing
        elif action_index == 1:
            action = 2 # Jump

        obs, reward, done, info = env.step(action)
        total_reward += reward
        indiv.frameNumber += 1

        if render:
            env.render()

        moveIndex += 1
        
        if done:
            break

    indiv.fitness = total_reward 
    indiv.score = info['score']
    indiv.x_pos = info['x_pos']
    indiv.flag = info['flag_get']
    return indiv


 
def save_model(indiv, generation, index):
    model_path = os.path.join(MODELS_DIR, f"best_model_gen_{generation}_{index}.npy")
    np.save(model_path, indiv.bytes)
    print(f"Saved best model of generation {generation} with fitness {indiv.fitness}")


def log_metrics(generation, mean_fitness, std_fitness, best_fitness, worst_fitness, median_fitness, fitness_improvement, diversity, avg_frame_number, time_taken):
    with open(LOG_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([generation, mean_fitness, std_fitness, best_fitness, worst_fitness, median_fitness, fitness_improvement, diversity, avg_frame_number, time_taken])
    print(f"Logged metrics for generation {generation}")




def get_specific_model():
    latest_model_file = 'best_model_gen_best.npy' #Change for testing othe models
    return os.path.join(MODELS_DIR, latest_model_file)

def load_model(model_path):
    bytes = np.load(model_path)
    indiv = Individual()
    indiv.bytes = bytes
    return indiv



def test_specific_model():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3') #Change for other levels
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    model_path = get_specific_model() # for specifying the model you want to test speciffically
    if model_path:
        _individual = load_model(model_path)
        print(f"Testing the individual from {model_path}")
        indiv = play_game(_individual, env, render=True)
        print('Score:', indiv.score,'X position:', indiv.x_pos,'Fitness:',indiv.fitness, 'Caprure the flug?', indiv.flag )
    else:
        print("No model found to load.")


def compute_diversity(population):
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distances.append(np.sum(population[i].bytes != population[j].bytes))
    return np.mean(distances)

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3') #Change for other levels
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    population = generateRandomPopulation()
    generation = 0
    best_fitness = 0
    best_individual = None


    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Generation', 'MeanFitness', 'StdFitness', 'BestFitness', 'WorstFitness', 'MedianFitness', 'FitnessImprovement', 'Diversity', 'AvgFrameNumber', 'TimeTaken'])

    previous_best_fitness = 0

    while True:
        start_time = time.time()
        generation += 1
        print(f'Generation: {generation}')
        
        fitness_scores = []
        frame_numbers = []

        for i, indiv in enumerate(population):
            time1 = time.time()
            current_indiv = play_game(indiv, env)
            fitness = current_indiv.fitness
            fitness_scores.append(fitness)
            frame_numbers.append(indiv.frameNumber)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = indiv
                save_model(best_individual, generation,i)


            time2=time.time()-time1
            print(f'Individual {i+1} Fitness: {fitness} Time: {time2}')


        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        worst_fitness = np.min(fitness_scores)
        median_fitness = np.median(fitness_scores)
        fitness_improvement = best_fitness - previous_best_fitness
        previous_best_fitness = best_fitness
        diversity = compute_diversity(population)
        avg_frame_number = np.mean(frame_numbers)
        time_taken = time.time() - start_time


        log_metrics(generation, mean_fitness, std_fitness, best_fitness, worst_fitness, median_fitness, fitness_improvement, diversity, avg_frame_number, time_taken)

        print(f'Best Fitness: {best_fitness}')
        population = evolvePopulation(population)

        # Stop condition 
        if generation >= GENERATION_SIZE: 
            break

    env.close()

if __name__ == "__main__":
    # main() #Trainig-->#Comment this for testing specific model

    test_specific_model() #Testing