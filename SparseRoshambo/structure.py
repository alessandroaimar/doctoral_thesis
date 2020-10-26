import logging

log = logging.getLogger()


class Layer(object):

    def __init__(self, structure):
        self.type = structure["type"]
        self.name = structure["name"]

        try:
            self.fixed_point_weights_num_bits = structure["weights_num_bits"]
            self.fixed_point_activations_num_bits = structure["activations_num_bits"]
            self.fixed_point_biases_num_bits = structure["biases_num_bits"]
        except KeyError:
            if self.type == "conv" or self.type == "FC":
                log.info("Layer doesnt have quantization information")

        try:
            self.strides = structure["strides"]
            self.padding = structure["padding"]
        except KeyError:
            if self.type == "conv" or self.type == "pool":
                raise AttributeError("Conv/pool provided without strides/padding information")

        try:
            self.filters = structure["filters"]
            self.kernel_size = structure["kernel_size"]

        except KeyError:
            if self.type == "conv":
                raise AttributeError("Conv provided without kernel information")

        try:
            self.activation = structure["activation"]
        except KeyError:
            if self.type == "conv" or self.type == "FC":
                raise AttributeError("Conv/FC provided without activation information")

        try:
            self.window_shape = structure["window_shape"]
            self.pooling_type = structure["pooling_type"]
        except KeyError:
            if self.type == "pool":
                raise AttributeError("Pool provided without window information")


class Structure(object):

    def __init__(self, structure, use_fixed_point, data_format):
        self.use_fixed_point = use_fixed_point
        self.data_format = data_format
        self.layers = self.generate_layers(structure)

    def generate_layers(self, structure):
        layers = []
        for structure_entry in structure:
            new_layer = Layer(structure_entry)
            layers.append(new_layer)

        return layers
