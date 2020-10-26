from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.keras.models import Model
import numpy as np
import os

def save_for_debug(model, input_tensor, test_dataset, savedir):


    input_image = list(test_dataset.as_numpy_iterator())[0]
    model.predict(input_image, verbose=1)
    os.makedirs(savedir)

    for layer in model.layers:
        for variable in layer.variables:
            if "output_stored_tensor" in variable.name:
                full_path = savedir + "layer_{}_out".format(layer.name)
                np.save(file=full_path, arr=variable.numpy())

    input_image = list(test_dataset.as_numpy_iterator())[0][0][0]
    full_path = savedir + "image"
    np.save(file=full_path, arr=input_image)

