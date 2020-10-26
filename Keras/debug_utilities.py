from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

def get_weight_grad(model, data, labels):
    means = []
    stds = []

    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(data, labels)
    output_grad = f(x + y + sample_weight)

    for layer in range(len(model.layers)):
        if model.layers[layer].__class__.__name__ == 'Conv2D':
            means.append(output_grad[layer].mean())
            stds.append(output_grad[layer].std())
    return means, stds


def get_stats(model, data):
    means = []
    stds = []

    for layer in range(len(model.layers)):
        if model.layers[layer].__class__.__name__ == 'Conv2D':
            m = Model(model.input, model.layers[layer].output)
            pred = m.predict(data)
            means.append(pred.mean())
            stds.append(pred.std())
    return means, stds