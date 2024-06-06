from itertools import count
import random

from .neat_config import NEATConfig
from .utils import clamp_value


class LinkIndexer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._counter = count(0)
        return cls._instance

    def generate_id(self):
        return next(self._counter)


class LinkGene:
    link_id: int
    source_neuron_id: int
    target_neuron_id: int
    weight: float
    enabled: bool

    def __init__(self, source_neuron_id, target_neuron_id, weight, enabled):
        self.link_id = LinkIndexer().generate_id()
        self.source_neuron_id = source_neuron_id
        self.target_neuron_id = target_neuron_id
        self.weight = weight
        self.enabled = enabled

    def new(source_neuron_id, target_neuron_id, config: NEATConfig, enabled=True):
        return LinkGene(
            source_neuron_id=source_neuron_id,
            target_neuron_id=target_neuron_id,
            weight=LinkGene.new_weight(
                config.weight_min_value,
                config.weight_max_value,
                config.weight_init_mean,
                config.weight_init_stdev,
            ),
            enabled=enabled,
        )

    def new_weight(min_value, max_value, init_mean, init_stdev):
        weight = random.gauss(init_mean, init_stdev)
        return clamp_value(weight, min_value, max_value)

    def __eq__(self, value: object) -> bool:
        return (
            self.source_neuron_id == value.source_neuron_id
            and self.target_neuron_id == value.target_neuron_id
        )

    def __str__(self):
        return f"LinkGene {self.source_neuron_id} -> {self.target_neuron_id}"

    def distance(self, other, weight_coeficient):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * weight_coeficient
