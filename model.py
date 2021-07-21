import jax.numpy as np
from nn_layers import FeedbackUnit, Dense
from nn_activations import relu, apr1


def build_model(input_size, latent_vector_sizes=(32,), param_init_scale=1e-2):
    if len(latent_vector_sizes) < 1:
        raise Exception("There needs to mbe at least one latent vector. Please supply a size like [32].")
    params = []
    global model
    model = Dense(input_size, input_size, params, scale=param_init_scale, activation_function=relu)
    #model = FeedbackUnit(
    #    input_size, latent_vector_sizes[0], params,
    #    param_init_scale=param_init_scale)
    last_layer = model
    for latent_vector_size_base, latent_vector_size_next in zip(latent_vector_sizes[:-1], latent_vector_sizes[1:]):
        last_layer.stacked_unit = FeedbackUnit(
            latent_vector_size_base, latent_vector_size_next, params,
            param_init_scale=param_init_scale)
        last_layer = last_layer.stacked_unit
    return params


def print_params(params):
    print("params: ", end="")
    for a in params:
        if len(a.shape) == 1:
            print("Vector({}) ".format(a.shape[0]), end="")
        elif len(a.shape) == 2:
            print("Matrix{} ".format(a.shape), end="")
        else:
            print(a.shape, end=" ")
    print("")


def predict(params, image_sequence):
    return model.predict(params, image_sequence[0])


def loss(params, train_data):
    input_sequence, y_image = train_data
    #prediction, loss_prediction = predict(params, input_sequence)
    prediction = predict(params, input_sequence)
    prediction_loss = np.mean(np.square(y_image - prediction))
    loss_prediction_loss = 0#apr1(np.square(prediction_loss - loss_prediction))
    return prediction_loss + loss_prediction_loss
