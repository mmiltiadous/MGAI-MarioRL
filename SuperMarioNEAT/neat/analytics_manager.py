import csv
import json
import os
import time
from typing import Dict, List
import matplotlib.pyplot as plt
from neat.genome import Genome
from neat.species import Species


class AnalyticsManager:
    best_fitness_per_generation: List[float]
    average_fitness_per_generation: List[float]
    species_counts_per_generation: List[Dict[int, int]]
    average_neurons_per_generation: List[float]
    average_links_per_generation: List[float]

    def __init__(self):
        self.best_fitness_per_generation = []
        self.average_fitness_per_generation = []
        self.species_counts_per_generation = []
        self.average_neurons_per_generation = []
        self.average_links_per_generation = []

    def report(self, genomes_dict: Dict[int, Genome], species_dict: Dict[int, Species]):
        genomes = genomes_dict.values()
        fitnesses = [g.fitness for g in genomes]
        self.best_fitness_per_generation.append(max(fitnesses))
        self.average_fitness_per_generation.append(sum(fitnesses) / len(fitnesses))
        self.species_counts_per_generation.append({s.species_id: len(s.genomes_ids) for s in species_dict.values()})

        total_nodes = 0
        total_links = 0
        for g in genomes:
            total_nodes += len(g.neurons())
            total_links += len(g.links())
        
        self.average_neurons_per_generation.append(total_nodes / len(genomes))
        self.average_links_per_generation.append(total_links / len(genomes))
    
    def save_csv(self, filename):
         # Convert complex structures to JSON strings
        species_counts_per_generation_json = [json.dumps(species) for species in self.species_counts_per_generation]


        # Combine all data into rows
        data = zip(
            self.best_fitness_per_generation,
            self.average_fitness_per_generation,
            species_counts_per_generation_json,
            self.average_neurons_per_generation,
            self.average_links_per_generation
        )

        # Define the header
        header = [
            'Best Fitness Per Generation',
            'Average Fitness Per Generation',
            'Species Counts Per Generation',
            'Average Neurons Per Generation',
            'Average Links Per Generation'
        ]

        # Write data to CSV
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for row in data:
                writer.writerow(row)

    def plot_fitness_over_generations(self, filename):
        generations = range(len(self.best_fitness_per_generation))
        plt.figure(figsize=(10, 5))
        plt.plot(generations, self.best_fitness_per_generation, label='Best Fitness')
        plt.plot(generations, self.average_fitness_per_generation, label='Average Fitness')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()
        plt.savefig(filename)
    
    def plot_species_over_generations(self, filename):
        generations = range(len(self.species_counts_per_generation))
        plt.figure(figsize=(10, 5))
        for species_id in self.species_counts_per_generation[-1].keys():
            species_counts = [gen[species_id] if species_id in gen else 0 for gen in self.species_counts_per_generation]
            plt.plot(generations, species_counts, label=f'Species {species_id}')
        plt.xlabel('Generations')
        plt.ylabel('Number of Genomes')
        plt.title('Species Population Over Generations')
        plt.legend()
        plt.savefig(filename)
    
    def plot_complexity_over_generations(self, filename):
        generations = range(len(self.average_neurons_per_generation))

        plt.figure(figsize=(10, 5))
        plt.plot(generations, self.average_neurons_per_generation, label='Average Neurons')
        plt.plot(generations, self.average_links_per_generation, label='Average Links')
        plt.xlabel('Generations')
        plt.ylabel('Count')
        plt.title('Complexity Over Generations')
        plt.legend()
        plt.savefig(filename)

    def save(self):
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # create timestamp directory
        timestamp = int(time.time())
        timestamp_dir = os.path.join(reports_dir, f"{timestamp}")
        os.makedirs(timestamp_dir, exist_ok=True)

        # save csv
        filename = os.path.join(timestamp_dir, f"report.csv")
        self.save_csv(filename)

        # plot fitness over generations
        filename = os.path.join(timestamp_dir, f"fitness_over_generations.png")
        self.plot_fitness_over_generations(filename)

        # plot species over generations
        filename = os.path.join(timestamp_dir, f"species_over_generations.png")
        self.plot_species_over_generations(filename)

        # plot complexity over  generations
        filename = os.path.join(timestamp_dir, f"complexity_over_generations.png")
        self.plot_complexity_over_generations(filename)
