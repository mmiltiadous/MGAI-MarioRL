from itertools import count
import random

from neat.neat_config import NEATConfig
from neat.utils import clamp_value

from .activations import sigmoid_activation


class NeuronGeneIndexer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._counter = count(0)
            return cls._instance

    def generate_id(self):
        return next(self._counter)


class NeuronGene:
    neuron_id: int
    bias: float
    activation: callable

    def __init__(self, neuron_id: int, bias: float, activation: callable):
        self.neuron_id = neuron_id
        self.bias = bias
        self.activation = activation

    def __eq__(self, value: "NeuronGene") -> bool:
        return self.neuron_id == value.neuron_id

    def new(neuron_id, config: NEATConfig):
        bias = NeuronGene.new_bias(
            config.bias_min_value,
            config.bias_max_value,
            config.bias_init_mean,
            config.bias_init_stdev,
        )
        activation = sigmoid_activation
        return NeuronGene(neuron_id, bias, activation)

    def new_bias(min_value, max_value, init_mean, init_stdev):
        bias = random.gauss(init_mean, init_stdev)
        return clamp_value(bias, min_value, max_value)

    def distance(self, other, compatibility_weight_coefficient):
        d = abs(self.bias - other.bias)
        if self.activation != other.activation:
            d += 1.0
        return d * compatibility_weight_coefficient
