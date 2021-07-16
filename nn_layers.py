import jax.numpy as np
from jax import random
from nn_activations import *


class Dense:
    def __init__(self, input_size, output_size, params,
                 activation_function=relu,
                 key=random.PRNGKey(0),
                 weight_initializer=random.normal, scale=1e-2):
        self.activation = activation_function
        w_key, b_key = random.split(key)
        w = scale * weight_initializer(w_key, (output_size, input_size))
        self.w_index = len(params)
        params.append(w)
        b = scale * weight_initializer(b_key, (output_size,))
        self.b_index = len(params)
        params.append(b)

    def predict(self, params, input_activations):
        w = params[self.w_index]
        b = params[self.b_index]
        return self.activation(np.dot(w, input_activations) + b)


class FeedbackUnit:
    def __init__(self, input_size, latent_vector_size, params, param_init_scale=1e-2):
        self.encoder = Dense(
            input_size, latent_vector_size, params,
            activation_function=relu, scale=param_init_scale)
        self.back_projector = Dense(
            latent_vector_size, input_size, params,
            activation_function=relu, scale=param_init_scale)
        self.reconstruction_error_predictor = Dense(
            latent_vector_size, 1, params,
            activation_function=sigmoid, scale=param_init_scale)
        self.stacked_unit = None

    def predict(self, params, input_activations):
        enc_activations = self.encoder.predict(params, input_activations)
        if self.stacked_unit:
            next_layer_back_projection, next_layer_error_prediction = self.stacked_unit.predict(enc_activations)
            latent_activation = (enc_activations * (np.float32(1) - next_layer_error_prediction) +
                                 next_layer_back_projection * next_layer_error_prediction)
        else:
            latent_activation = enc_activations
        back_projection = self.back_projector.predict(params, latent_activation)
        error_prediction = self.reconstruction_error_predictor.predict(params, latent_activation)
        return back_projection, error_prediction
