import jax.numpy as np
from nn_layers import FeedbackUnit
from nn_activations import apr1


def build_model(input_size, latent_vector_sizes=(32,), param_init_scale=1e-2):
    if len(latent_vector_sizes) < 1:
        raise Exception("There needs to mbe at least one latent vector. Please supply a size like [32].")
    params = []
    global model
    model = FeedbackUnit(
        input_size, latent_vector_sizes[0], params,
        param_init_scale=param_init_scale)
    last_layer = model
    for latent_vector_size_base, latent_vector_size_next in zip(latent_vector_sizes[:-1], latent_vector_sizes[1:]):
        last_layer.stacked_unit = FeedbackUnit(
            latent_vector_size_base, latent_vector_size_next, params,
            param_init_scale=param_init_scale)
    return params


def predict(params, image_sequence):
    return model.predict(params, image_sequence[0])


def loss(params, train_data):
    input_sequence, y_image = train_data
    prediction, loss_prediction = predict(params, input_sequence)
    prediction_loss = np.square(y_image - prediction)
    loss_prediction_loss = apr1(np.square(prediction_loss - loss_prediction))
    return np.mean(prediction_loss + loss_prediction_loss)
