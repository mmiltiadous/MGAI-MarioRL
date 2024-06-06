from .genome import Genome, GenomeIndexer
from .link_gene import LinkGene
from .neuron_gene import NeuronGene
from .utils import choose_random


def _crossover_neuron(a: NeuronGene, b: NeuronGene) -> NeuronGene:
    """Crossover two neurons to create a new one."""
    assert a.neuron_id == b.neuron_id
    neuron_id = a.neuron_id
    activation = choose_random(a.activation, b.activation, 0.5)
    bias = choose_random(a.bias, b.bias, 0.5)
    return NeuronGene(neuron_id, bias, activation)


def _crossover_link(a: LinkGene, b: LinkGene) -> LinkGene:
    """Crossover two links to create a new one."""
    assert a == b
    source_id, target_id = a.source_neuron_id, a.target_neuron_id
    weight = choose_random(a.weight, b.weight, 0.5)
    enabled = choose_random(a.enabled, b.enabled, 0.5)
    return LinkGene(source_id, target_id, weight, enabled)


def crossover(parent1: Genome, parent2: Genome, genome_indexer) -> Genome:
    """Crossover two genomes to create a new one."""
    # Set fittest genome as dominant
    dominant, recessive = (
        (parent1, parent2) if parent1.fitness < parent2.fitness else (parent2, parent1)
    )

    offspring = Genome(
        next(genome_indexer),
        dominant.num_inputs,
        dominant.num_outputs,
    )

    # Crossover neurons
    for dominant_neuron in dominant.neurons():
        neuron_id = dominant_neuron.neuron_id
        recessive_neuron = recessive.find_neuron(neuron_id)
        if recessive_neuron is None:
            offspring.add_neuron(dominant_neuron)
        else:
            offspring.add_neuron(_crossover_neuron(dominant_neuron, recessive_neuron))

    # Crossover links
    for dominant_link in dominant.links():
        recessive_link = recessive.find_link(dominant_link)
        if recessive_link is None:
            offspring.add_link(dominant_link)
        else:
            offspring.add_link(_crossover_link(dominant_link, recessive_link))

    return offspring
