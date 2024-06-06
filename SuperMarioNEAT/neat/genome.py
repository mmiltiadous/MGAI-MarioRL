import os
import random
from typing import Dict, List
from .link_gene import LinkGene
from .neuron_gene import NeuronGene
from itertools import count
import matplotlib.pyplot as plt
import networkx as nx


class GenomeIndexer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._counter = count(0)
        return cls._instance

    def generate_id(self):
        return next(self._counter)


class Genome:
    genome_id: int
    num_inputs: int
    num_outputs: int
    neurons_dict: Dict[int, NeuronGene]
    links_dict: Dict[int, LinkGene]
    fitness: float
    # normalized_fitness: float
    neuron_indexer: count

    def __init__(self, genome_id, num_inputs, num_outputs):
        self.genome_id = genome_id
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons_dict = {}
        self.links_dict = {}
        self.fitness = None
        # self.normalized_fitness = None
        self.neuron_indexer = None

    def neurons(self) -> List[NeuronGene]:
        return list(self.neurons_dict.values())

    def links(self) -> List[LinkGene]:
        return list(self.links_dict.values())

    def add_neuron(self, neuron: NeuronGene):
        self.neurons_dict[neuron.neuron_id] = neuron

    def remove_neuron(self, neuron: NeuronGene):
        del self.neurons_dict[neuron.neuron_id]

    def add_link(self, link: LinkGene):
        self.links_dict[link.link_id] = link

    def remove_link(self, link: LinkGene):
        del self.links_dict[link.link_id]

    def find_link(self, link: LinkGene) -> LinkGene:
        if link.link_id in self.links_dict:
            return self.links_dict[link.link_id]
        return None

    def find_link_by_source_target(
        self, source_neuron_id: int, target_neuron_id: int
    ) -> LinkGene:
        for _link in self.links():
            if (
                _link.source_neuron_id == source_neuron_id
                and _link.target_neuron_id == target_neuron_id
            ):
                return _link
        return None

    def find_neuron(self, id: int) -> NeuronGene:
        if id in self.neurons_dict:
            return self.neurons_dict[id]
        return None

    def random_link(self) -> LinkGene:
        keys = list(self.links_dict.keys())
        r = random.randrange(0, len(keys))
        return self.links_dict[keys[r]]

    def random_neuron(self) -> NeuronGene:
        keys = list(self.neurons_dict.keys())
        r = random.randrange(0, len(keys))
        return self.neurons_dict[keys[r]]

    def make_input_keys(self):
        return [-i - 1 for i in range(self.num_inputs)]

    def make_output_keys(self):
        return [i for i in range(self.num_outputs)]

    def get_new_neuron_id(self):
        if self.neuron_indexer is None:
            self.neuron_indexer = count(max([n.neuron_id for n in self.neurons()]) + 1)

        new_id = next(self.neuron_indexer)

        assert new_id not in self.neurons_dict

        return new_id

    def __eq__(self, value: "Genome") -> bool:
        return self.genome_id == value.genome_id

    def __str__(self):
        return f"Genome {self.genome_id}"

    def visualize_network(self):
        G = nx.DiGraph()

        # Add neurons as nodes
        for neuron in self.neurons():
            G.add_node(neuron.neuron_id)

        # Add links as edges
        for link in self.links():
            G.add_edge(link.source_neuron_id, link.target_neuron_id, weight=link.weight)

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=1500,
            edge_color="black",
            linewidths=1,
            font_size=10,
            arrows=True,
        )

        # Add weights on edges
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        plt.savefig(os.path.join(os.getcwd(), f"genome_{self.genome_id}.png"))
        print("Saved genome to file")
        input("Press enter to continue")
