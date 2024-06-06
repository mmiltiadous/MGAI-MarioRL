class NEATConfig:
    population_size: int
    num_inputs: int
    num_outputs: int
    survival_threshold: float
    disjoint_coeficient: float
    weight_coeficient: float
    compatibility_threshold: float
    compatibility_threshold_adjustment_factor: float
    adjust_compatibility_threshold: bool
    weight_min_value: float
    weight_max_value: float
    weight_init_stdev: float
    weight_init_mean: float
    weight_mutate_power: float
    add_link_rate: float
    remove_link_rate: float
    add_neuron_rate: float
    remove_neuron_rate: float
    weight_mutate_rate: float
    weight_replace_rate: float
    checkpoint_frequency: int
    desired_species_size: int
    min_species_size: int
    max_stagnation: int
    num_hidden: int
    bias_init_mean: float
    bias_init_stdev: float
    bias_max_value: float
    bias_min_value: float
    elitism: int

    def __init__(
        self,
        population_size: int,
        num_inputs: int,
        num_outputs: int,
        num_hidden: int,
        survival_threshold: float,
        add_neuron_rate: float,
        remove_neuron_rate: float,
        add_link_rate: float,
        remove_link_rate: float,
        disjoint_coeficient: float,
        weight_coeficient: float,
        compatibility_threshold: float,
        compatibility_threshold_adjustment_factor: float,
        adjust_compatibility_threshold: bool,
        weight_min_value: float,
        weight_max_value: float,
        weight_init_stdev: float,
        weight_init_mean: float,
        weight_mutate_power: float,
        weight_mutate_rate: float,
        weight_replace_rate: float,
        checkpoint_frequency: int,
        desired_species_size: int,
        min_species_size: int,
        max_stagnation: int,
        bias_init_mean: float,
        bias_init_stdev: float,
        bias_max_value: float,
        bias_min_value: float,
        elitism: int,
    ):
        self.population_size = population_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.survival_threshold = survival_threshold
        self.add_neuron_rate = add_neuron_rate
        self.remove_neuron_rate = remove_neuron_rate
        self.add_link_rate = add_link_rate
        self.remove_link_rate = remove_link_rate
        self.disjoint_coeficient = disjoint_coeficient
        self.weight_coeficient = weight_coeficient
        self.compatibility_threshold = compatibility_threshold
        self.weight_min_value = weight_min_value
        self.weight_max_value = weight_max_value
        self.weight_init_stdev = weight_init_stdev
        self.weight_init_mean = weight_init_mean
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.compatibility_threshold_adjustment_factor = (
            compatibility_threshold_adjustment_factor
        )
        self.desired_species_size = desired_species_size
        self.adjust_compatibility_threshold = adjust_compatibility_threshold
        self.min_species_size = min_species_size
        self.max_stagnation = max_stagnation
        self.bias_init_mean = bias_init_mean
        self.bias_init_stdev = bias_init_stdev
        self.bias_max_value = bias_max_value
        self.bias_min_value = bias_min_value
        self.elitism = elitism
