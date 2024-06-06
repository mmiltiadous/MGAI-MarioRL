from itertools import count
import logging
import os
import random
import sys
from typing import List

from neat.neat_config import NEATConfig

from .genome import Genome

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


class SpeciesIndexer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._counter = count(0)
        return cls._instance

    def generate_id(self):
        return next(self._counter)


class Species:
    species_id: int
    genomes_ids: List[int]
    best_fitness: float
    stagnation_counter: int
    representative: Genome
    normalized_fitness: float

    def __init__(self, species_id: int = None):
        self.genomes_ids = []
        self.stagnation_counter = 0
        self.best_fitness = None
        self.representative = None
        self.species_id = (
            SpeciesIndexer().generate_id() if species_id is None else species_id
        )
        self.normalized_fitness = None

    def add_genome(self, genome: Genome):
        self.genomes_ids.append(genome.genome_id)

    def remove_genome(self, genome: Genome):
        self.genomes_ids.remove(genome.genome_id)

    def random_genome(self):
        return self.genomes_ids[random.randrange(0, len(self.genomes_ids))]

    def update_best_fitness(self, genomes_dict):
        fitnesses = [genomes_dict[gid].fitness for gid in self.genomes_ids]
        max_fitness = max(fitnesses)
        if self.best_fitness is None or max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def is_stagnant(self, config: NEATConfig):
        return self.stagnation_counter > config.max_stagnation

    def clear_genomes(self):
        self.genomes_ids = []

    def calculate_disjoint_and_excess_genes(ids1, ids2):
        # Calculate maximum link_id in both genomes
        max_id1 = max(ids1, default=-1)
        max_id2 = max(ids2, default=-1)

        # Convert lists to sets for set operations
        set_ids1 = set(ids1)
        set_ids2 = set(ids2)

        # Disjoint genes (appear in one set but not the other, with ids <= max_id1 and max_id2)
        disjoint_genes = [
            id_ for id_ in ids1 if id_ not in set_ids2 and id_ <= max_id2
        ] + [id_ for id_ in ids2 if id_ not in set_ids1 and id_ <= max_id1]

        # Excess genes (appear in one set but not the other, with ids > max_id1 or max_id2)
        excess_genes = [
            id_ for id_ in ids1 if id_ not in set_ids2 and id_ > max_id2
        ] + [id_ for id_ in ids2 if id_ not in set_ids1 and id_ > max_id1]

        return len(disjoint_genes), len(excess_genes)

    def distance(
        genome1: Genome,
        representative: Genome,
        disjoint_coeficient: float,
        weight_coeficient: float,
    ):
        if (
            genome1 is None
            or representative is None
            or genome1.genome_id == representative.genome_id
        ):
            return 0.0

        # Compute node gene distance component.
        neuron_distance = 0.0
        if genome1.neurons_dict or representative.neurons_dict:
            disjoint_nodes = 0
            for k2 in representative.neurons_dict:
                if k2 not in genome1.neurons_dict:
                    disjoint_nodes += 1

            for k1, n1 in genome1.neurons_dict.items():
                n2 = representative.neurons_dict.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    neuron_distance += n1.distance(n2, weight_coeficient)

            max_neurons = max(
                len(genome1.neurons_dict), len(representative.neurons_dict)
            )
            neuron_distance = (
                neuron_distance + (disjoint_coeficient * disjoint_nodes)
            ) / max_neurons

        # Compute connection gene differences.
        link_distance = 0.0
        if genome1.links_dict or representative.links_dict:
            disjoint_links = 0
            for k2 in representative.links_dict:
                if k2 not in genome1.links_dict:
                    disjoint_links += 1

            for k1, c1 in genome1.links_dict.items():
                c2 = representative.links_dict.get(k1)
                if c2 is None:
                    disjoint_links += 1
                else:
                    # Homologous genes compute their own distance value.
                    link_distance += c1.distance(c2, weight_coeficient)

            max_conn = max(len(genome1.links_dict), len(representative.links_dict))
            link_distance = (
                link_distance + (disjoint_coeficient * disjoint_links)
            ) / max_conn

        distance = neuron_distance + link_distance
        logger.debug(
            f"Distance between {genome1.genome_id} and {representative.genome_id}: {distance}"
        )
        return distance
