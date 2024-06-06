from itertools import count
import math
import logging
import os
import pickle
import sys
import time
from typing import Dict, List

from numpy import mean
from .crossover import crossover
from .genome import Genome
from .link_gene import LinkGene
from .mutation import mutate
from .neat_config import NEATConfig
from .neuron_gene import NeuronGene
from .species import Species
from .utils import choose_random_element
from .analytics_manager import AnalyticsManager
import os

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


class Population:
    population_size: int
    config: NEATConfig
    best_genome: Genome
    genome_indexer: count
    species_dict: Dict[int, Species]
    genomes_dict: Dict[int, Genome]
    generations: int
    analytics_manager: AnalyticsManager
    compatibility_threshold: float

    def __init__(self, config: NEATConfig):
        self.species_dict = {}
        self.genomes_dict = {}
        self.generations = 0
        self.analytics_manager = AnalyticsManager()
        self.config = config
        self.compatibility_threshold = config.compatibility_threshold
        self.genome_indexer = count(1)
        self.best_genome = None
        for _ in range(config.population_size):
            g = self._new_genome()
            self.genomes_dict[g.genome_id] = g
        self.speciate()
        logger.info(f"Population initialized with {len(self.genomes())} genomes")

    def genomes(self) -> List[Genome]:
        return list(self.genomes_dict.values())

    def species(self) -> List[Species]:
        return list(self.species_dict.values())

    def run(self, eval_genomes, num_generations):
        for _ in range(num_generations):
            # Evaluate genomes
            logger.info(f"(GENERATION {self.generations}) - Evaluating...")
            eval_genomes(self.genomes_dict)
            logger.info(
                f"(GENERATION {self.generations}) - Evaluated ({len(self.genomes())} genomes)"
            )

            # Save checkpoint
            if (
                self.config.checkpoint_frequency > 0
                and self.generations % self.config.checkpoint_frequency == 0
            ):
                self.save()

            # Update analytics
            self.analytics_manager.report(self.genomes_dict, self.species_dict)

            # Update best
            logger.info(f"(GENERATION {self.generations}) - Updating best...")
            self._update_best()
            logger.info(
                f"(GENERATION {self.generations}) - best genome fitness: {self.best_genome.fitness}"
            )

            # check for extinction
            if not self.species():
                logger.info("No species left, population extinct")
                break

            # Reproduce
            logger.info(f"(GENERATION {self.generations}) - Reproducing...")
            self.reproduce()
            logger.info(f"(GENERATION {self.generations}) - Reproduced")

            # Create species
            logger.info(f"(GENERATION {self.generations}) - Speciating...")
            self.speciate()
            logger.info(
                f"(GENERATION {self.generations}) - Speciated ({len(self.species())} species)"
            )

            self.generations += 1

        self.analytics_manager.save()
        return self.best_genome

    def speciate(self):
        for genome in self.genomes():
            assigned = False
            for species in self.species():
                # Select random genome from species as representative
                representative = species.representative
                if representative is None:
                    continue

                # Calculate genetic distance
                distance = Species.distance(
                    genome,
                    representative,
                    self.config.disjoint_coeficient,
                    self.config.weight_coeficient,
                )
                if distance < self.compatibility_threshold:
                    # Add genome to species
                    assigned = True
                    species.add_genome(genome)
                    break

            if not assigned:
                # Add new Species
                new_species = Species()
                new_species.add_genome(genome)
                new_species.representative = genome
                self.species_dict[new_species.species_id] = new_species

    def adjust_compatibility_threshold(self):
        if not self.config.adjust_compatibility_threshold:
            return
        # Adjust compatibility threshold
        previous_threshold = self.compatibility_threshold
        if len(self.species()) > self.config.desired_species_size:
            self.compatibility_threshold -= (
                self.config.compatibility_threshold_adjustment_factor
            )
        elif len(self.species()) < self.config.desired_species_size:
            self.compatibility_threshold += (
                self.config.compatibility_threshold_adjustment_factor
            )
        self.compatibility_threshold = max(self.compatibility_threshold, 0.1)
        logger.info(
            f"Compatibility threshold adjusted from {previous_threshold} to {self.compatibility_threshold}"
        )

    def adjust_fitnesses(self):
        min_fitness = min([g.fitness for g in self.genomes()])
        max_fitness = max([g.fitness for g in self.genomes()])
        fit_range = max(max_fitness - min_fitness, 1.0)
        for s in self.species():
            mean_species_fitness = mean(
                [self.genomes_dict[gid].fitness for gid in s.genomes_ids]
            )
            normalized_fitness = (mean_species_fitness - min_fitness) / fit_range
            s.normalized_fitness = normalized_fitness

    def select_parents(self, genomes):
        p1 = choose_random_element(genomes)
        p2 = choose_random_element(genomes)
        return p1, p2

    def spawn_sizes(
        self, normalized_fitnesses, previous_sizes, pop_size, min_species_size
    ):
        nf_sum = sum(normalized_fitnesses)
        spawns = []
        for nf, previous_size in zip(normalized_fitnesses, previous_sizes):
            if nf_sum > 0:
                spawn = max(min_species_size, nf / nf_sum * pop_size)
            else:
                spawn = min_species_size
            d = (spawn - previous_size) / 2.0
            c = int(round(d))
            spawn = previous_size
            spawn += c
            spawns.append(spawn)

        normalized = pop_size / sum(spawns)
        spawns = [max(min_species_size, int(round(n * normalized))) for n in spawns]
        return spawns

    def reproduce(self):
        if len(self.species()) == 0:
            self.genomes_dict = {}

        # update stagnant state of species
        for s in self.species_dict.values():
            s.update_best_fitness(self.genomes_dict)

        my_species = [
            s for s in self.species_dict.values() if not s.is_stagnant(self.config)
        ]

        if not my_species:
            self.genomes_dict = {}

        self.adjust_fitnesses()

        new_genomes_dict = {}
        new_species_dict = {}

        previous_sizes = [len(s.genomes_ids) for s in my_species]
        min_species_size = max(self.config.min_species_size, self.config.elitism)
        normalized_fitnesses = [s.normalized_fitness for s in my_species]
        spawn_sizes = self.spawn_sizes(
            normalized_fitnesses,
            previous_sizes,
            self.config.population_size,
            min_species_size,
        )

        for s, spawn_size in zip(my_species, spawn_sizes):
            old_members = list(s.genomes_ids)
            s.genomes_ids = []
            new_species_dict[s.species_id] = s

            old_members.sort(
                reverse=True, key=lambda gid: self.genomes_dict[gid].fitness
            )

            if self.config.elitism > 0:
                for gid in old_members[: self.config.elitism]:
                    new_genomes_dict[gid] = self.genomes_dict[gid]
                    spawn_size -= 1
            if spawn_size <= 0:
                continue

            cutoff = int(math.ceil(self.config.survival_threshold * len(old_members)))
            cutoff = max(cutoff, 2)
            old_members = old_members[:cutoff]

            for _ in range(spawn_size):
                # Choose parents
                p1_id, p2_id = self.select_parents(old_members)
                p1 = self.genomes_dict[p1_id]
                p2 = self.genomes_dict[p2_id]
                # crossover
                child = crossover(p1, p2, self.genome_indexer)
                # mutate
                mutate(child, self.config)
                new_genomes_dict[child.genome_id] = child

        # Add new population
        self.genomes_dict = new_genomes_dict
        self.species_dict = new_species_dict

    def remove_genome(self, genome: Genome):
        assert genome.genome_id in self.genomes_dict
        del self.genomes_dict[genome.genome_id]

    def _update_best(self):
        current_champ = max(
            self.genomes(),
            key=lambda g: g.fitness,
        )
        if self.best_genome is None or current_champ.fitness > self.best_genome.fitness:
            self.best_genome = current_champ

    def _new_genome(self):
        genome = Genome(
            genome_id=next(self.genome_indexer),
            num_inputs=self.config.num_inputs,
            num_outputs=self.config.num_outputs,
        )

        for neuron_id in range(genome.num_outputs):
            genome.add_neuron(NeuronGene.new(neuron_id=neuron_id, config=self.config))

        if self.config.num_hidden > 0:
            # add hidden neurons
            hidden = [
                NeuronGene.new(neuron_id=genome.get_new_neuron_id(), config=self.config)
                for _ in range(self.config.num_hidden)
            ]
            [genome.add_neuron(n) for n in hidden]

            for i in range(genome.num_inputs):
                input_id = -i - 1
                for h in hidden:
                    genome.add_link(LinkGene.new(input_id, h.neuron_id, self.config))
            for h in hidden:
                for output_id in range(genome.num_outputs):
                    genome.add_link(LinkGene.new(h.neuron_id, output_id, self.config))
        else:
            for i in range(genome.num_inputs):
                input_id = -i - 1
                for output_id in range(genome.num_outputs):
                    genome.add_link(LinkGene.new(input_id, output_id, self.config))
        return genome

    def save(self):
        checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        full_path = os.path.join(
            checkpoints_dir,
            f"population_g{self.generations}_t{int(time.time())}.pickle",
        )
        with open(full_path, "wb") as f:
            pickle.dump(self, f)
