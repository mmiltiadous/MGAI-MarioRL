from .genome import Genome
from .link_gene import LinkGene


class NeuronInput:
    id: int
    weight: float

    def __init__(self, id: int, weight: float):
        self.id = id
        self.weight = weight

    def __copy__(self):
        return NeuronInput(self.id, self.weight)


class Neuron:
    id: int
    activation: callable
    bias: float
    inputs: list[NeuronInput]

    def __init__(
        self,
        id: int,
        bias: float,
        activation: callable,
        inputs: list[NeuronInput],
    ):
        self.id = id
        self.bias = bias
        self.activation = activation
        self.inputs = inputs

    def __copy__(self):
        return Neuron(self.id, self.bias, self.activation, self.inputs.copy())


class FeedForwardNetwork:
    input_ids: list[int]
    output_ids: list[int]
    neurons: list[Neuron]
    values: dict[int, float]

    def __init__(
        self, input_ids: list[int], output_ids: list[int], neurons: list[Neuron]
    ):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.neurons = neurons
        self.values = dict((key, 0.0) for key in input_ids + output_ids)

    def activate(self, inputs: list[float]) -> list[float]:
        assert len(inputs) == len(self.input_ids)

        for k, v in zip(self.input_ids, inputs):
            self.values[k] = v

        for neuron in self.neurons:
            neuron_inputs = []
            for _input in neuron.inputs:
                neuron_inputs.append(self.values[_input.id] * _input.weight)
            s = sum(neuron_inputs)
            self.values[neuron.id] = neuron.activation(neuron.bias + s)

        outputs = [self.values[i] for i in self.output_ids]
        return outputs

    def create_from_genome(genome: Genome) -> "FeedForwardNetwork":
        inputs = genome.make_input_keys()
        outputs = genome.make_output_keys()
        layers = FeedForwardNetwork._feed_forward_layers(
            inputs, outputs, genome.links()
        )

        neurons = []
        for layer in layers:
            for neuron_id in layer:
                neuron_inputs = []
                for link in genome.links():
                    if neuron_id == link.target_neuron_id:
                        neuron_inputs.append(
                            NeuronInput(link.source_neuron_id, link.weight)
                        )
                neuron_gene_opt = genome.find_neuron(neuron_id)
                assert neuron_gene_opt is not None
                neurons.append(
                    Neuron(
                        neuron_gene_opt.neuron_id,
                        neuron_gene_opt.bias,
                        neuron_gene_opt.activation,
                        neuron_inputs.copy(),
                    )
                )
        return FeedForwardNetwork(
            input_ids=inputs.copy(), output_ids=outputs.copy(), neurons=neurons.copy()
        )

    def _feed_forward_layers(
        inputs: list[int], outputs: list[int], links: list[LinkGene]
    ) -> list[Neuron]:
        required = FeedForwardNetwork._required_for_output(inputs, outputs, links)

        layers = []
        s = set(inputs)
        while 1:
            c = set(
                link.target_neuron_id
                for link in links
                if link.source_neuron_id in s and link.target_neuron_id not in s
            )
            t = set()
            for n in c:
                if n in required and all(
                    link.source_neuron_id in s
                    for link in links
                    if link.target_neuron_id == n
                ):
                    t.add(n)
            if not t:
                break

            layers.append(t)
            s = s.union(t)
        return layers

    def _required_for_output(
        inputs: list[int], outputs: list[int], links: list[LinkGene]
    ) -> list[int]:
        assert not set(inputs).intersection(outputs)

        required = set(outputs)
        s = set(outputs)
        while 1:
            t = set(
                link.source_neuron_id
                for link in links
                if link.target_neuron_id in s and link.source_neuron_id not in s
            )

            if not t:
                break

            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break

            required = required.union(layer_nodes)
            s = s.union(t)
        return required
