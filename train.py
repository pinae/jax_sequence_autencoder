import model
from cifar10_loader import get_train_batches, get_test_batches
from model import build_model, loss, print_params
import jax.numpy as np
import tensorflow as tf
import pandas as pd
from jax import grad, jit, vmap


learning_rate = 0.5
num_epochs = 10
batch_size = 512
param_init_scale = 0.1


def make_sequence_batch(image_batch, sequence_length=5):
    sequence_batch = np.asarray(image_batch, dtype=np.float32) * np.float32(1/256)
    num_flattened_values = sequence_batch.shape[-3] * sequence_batch.shape[-2] * sequence_batch.shape[-1]
    batch_of_single_flattened_images = np.reshape(sequence_batch, (sequence_batch.shape[0], 1, num_flattened_values))
    sequence_batch = np.concatenate([batch_of_single_flattened_images for _ in range(sequence_length)], axis=1)
    return sequence_batch


def make_y_batch(input_data):
    scaled_image_batch = np.asarray(input_data, dtype=np.float32) * np.float32(1/256)
    num_flattened_values = scaled_image_batch.shape[-3] * scaled_image_batch.shape[-2] * scaled_image_batch.shape[-1]
    scaled_image_batch = np.reshape(scaled_image_batch, (scaled_image_batch.shape[0], num_flattened_values))
    return scaled_image_batch


@jit
def mean_loss(par, data):
    batched_loss = vmap(loss, in_axes=(None, 0))
    mean_loss_of_batch = np.mean(batched_loss(par, data))
    return mean_loss_of_batch


def test_performance(params, epoch_results):
    test_losses = []
    ssim_scores = []
    for x in get_test_batches(batch_size=batch_size):
        y_batch = make_y_batch(x['image'])
        seq_batch = make_sequence_batch(x['image'], sequence_length=1)
        batched_predict = vmap(model.predict, in_axes=(None, 0))
        #predictions_batch, prediction_prediction_batch = batched_predict(params, seq_batch)
        predictions_batch = batched_predict(params, seq_batch)
        real_batch_size = x['image'].shape[0]
        ssim_scores.append(np.mean(tf.image.ssim(
            np.reshape(predictions_batch, (real_batch_size, 32, 32, 3)),
            np.reshape(y_batch, (real_batch_size, 32, 32, 3)),
            max_val=1.0).numpy()))
        test_losses.append(mean_loss(params, (seq_batch, y_batch)))
    mean_test_loss = np.mean(np.asarray(test_losses, dtype=np.float32))
    mean_ssim_score = np.mean(np.asarray(ssim_scores, dtype=np.float32))
    print("Test Loss:", mean_test_loss, "\tSSIM score:", mean_ssim_score)
    epoch_results["mean test loss"].append(mean_test_loss)
    epoch_results["mean SSIM score"].append(mean_ssim_score)


@jit
def update(params, train_data):
    grads = grad(mean_loss)(params, train_data)
    return [p - learning_rate * dp for p, dp in zip(params, grads)]


def train(epochs=10, latent_vector_sizes=(1024, 256, 32)):
    global learning_rate
    params = build_model(input_size=32 * 32 * 3, latent_vector_sizes=latent_vector_sizes)
    print_params(params)
    epoch_results = {"mean training loss": [None], "mean test loss": [], "mean SSIM score": []}
    test_performance(params, epoch_results)
    for epoch in range(epochs):
        print("Learning rate:", learning_rate)
        epoch_history = {"batch_mean_loss": []}
        for bn, x in enumerate(get_train_batches(batch_size=batch_size)):
            y_batch = make_y_batch(x['image'])
            seq_batch = make_sequence_batch(x['image'], sequence_length=1)
            batch_loss = mean_loss(params, (seq_batch, y_batch))
            params = update(params, (seq_batch, y_batch))
            epoch_history["batch_mean_loss"].append(batch_loss)
            print("training batch", bn, "loss:", batch_loss)
        epoch_results["mean training loss"].append(
            np.mean(np.asarray(epoch_history["batch_mean_loss"], dtype=np.float32)))
        test_performance(params, epoch_results)
        learning_rate = learning_rate * 0.8
    return epoch_results, params


if __name__ == "__main__":
    training_results, params = train(num_epochs, latent_vector_sizes=[3072])
    training_results = pd.DataFrame(training_results)
    print(training_results)
