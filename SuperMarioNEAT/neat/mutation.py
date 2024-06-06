from typing import List
import random
from .genome import Genome
from .link_gene import LinkGene
from .neat_config import NEATConfig
from .neuron_gene import NeuronGene
from .utils import clamp_value


def _creates_cycle(connections: List[LinkGene], i: int, o: int) -> bool:
    """
    Returns true if the addition of the link i->o  would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for conn in connections:
            a, b = conn.source_neuron_id, conn.target_neuron_id
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def _mutate_add_link(genome: Genome, config: NEATConfig):
    """Adds a new link to the genome."""
    possible_outputs = [n.neuron_id for n in genome.neurons()]
    output_id = random.choice(possible_outputs)

    possible_inputs = possible_outputs + genome.make_input_keys()
    input_id = random.choice(possible_inputs)

    existing_link = genome.find_link_by_source_target(input_id, output_id)
    if existing_link is not None:
        existing_link.enabled = True
        return

    output_keys = genome.make_output_keys()
    if input_id in output_keys and output_id in output_keys:
        return

    if _creates_cycle(genome.links(), input_id, output_id):
        return

    link = LinkGene.new(input_id, output_id, config)
    genome.add_link(link)


def _mutate_remove_link(genome: Genome):
    """Removes a link from the genome."""
    if len(genome.links()) == 0:
        return
    to_remove_link = genome.random_link()
    genome.remove_link(to_remove_link)


def _mutate_add_neuron(genome: Genome, config: NEATConfig):
    if len(genome.links()) == 0:
        return

    link_to_split = genome.random_link()

    new_neuron_id = genome.get_new_neuron_id()
    new_neuron = NeuronGene.new(neuron_id=new_neuron_id, config=config)
    genome.add_neuron(new_neuron)

    link_to_split.enabled = False

    genome.add_link(LinkGene(link_to_split.source_neuron_id, new_neuron_id, 1.0, True))
    genome.add_link(
        LinkGene(
            new_neuron_id, link_to_split.target_neuron_id, link_to_split.weight, True
        )
    )


def _mutate_remove_neuron(genome: Genome):
    available_neurons = [
        k for k in genome.neurons() if k.neuron_id not in genome.make_output_keys()
    ]
    if not available_neurons or len(available_neurons) == 0:
        return -1

    neuron_to_remove = genome.random_neuron()

    links_to_remove = set()
    for link in genome.links():
        if neuron_to_remove.neuron_id in [link.source_neuron_id, link.target_neuron_id]:
            links_to_remove.add((link.source_neuron_id, link.target_neuron_id))

    for source, target in links_to_remove:
        link = genome.find_link_by_source_target(source, target)
        genome.remove_link(link)

    genome.remove_neuron(neuron_to_remove)
    return neuron_to_remove.neuron_id


def _mutate_weight(
    link: LinkGene,
    config: NEATConfig,
):
    r = random.random()
    if r < config.weight_mutate_rate:
        mutation = random.gauss(0.0, config.weight_mutate_power)
        link.weight = clamp_value(
            link.weight + mutation, config.weight_min_value, config.weight_max_value
        )

    if r < config.weight_replace_rate + config.weight_mutate_rate:
        link.weight = LinkGene.new_weight(
            config.weight_min_value,
            config.weight_max_value,
            config.weight_init_mean,
            config.weight_init_stdev,
        )


def mutate(genome: Genome, config: NEATConfig):
    if random.random() < config.add_neuron_rate:
        _mutate_add_neuron(genome, config)

    if random.random() < config.remove_neuron_rate:
        _mutate_remove_neuron(genome)

    if random.random() < config.add_link_rate:
        _mutate_add_link(genome, config)

    if random.random() < config.remove_link_rate:
        _mutate_remove_link(genome)

    for link in genome.links():
        _mutate_weight(link, config)
