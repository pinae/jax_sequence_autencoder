import model
from cifar10_loader import get_train_batches, get_test_batches
from model import build_model, loss
import jax.numpy as np
import tensorflow as tf
import pandas as pd
from jax import grad, jit, vmap


learning_rate = 0.01
num_epochs = 20
batch_size = 512
param_init_scale = 0.1


def make_sequence_batch(image_batch):
    sequence_batch = np.asarray(image_batch, dtype=np.float32) * np.float32(1/256)
    num_flattened_values = sequence_batch.shape[-3] * sequence_batch.shape[-2] * sequence_batch.shape[-1]
    sequence_batch = np.reshape(sequence_batch, (sequence_batch.shape[0], 1, num_flattened_values))
    return sequence_batch


def make_y_batch(input_data):
    scaled_image_batch = np.asarray(input_data, dtype=np.float32) * np.float32(1/256)
    num_flattened_values = scaled_image_batch.shape[-3] * scaled_image_batch.shape[-2] * scaled_image_batch.shape[-1]
    scaled_image_batch = np.reshape(scaled_image_batch, (scaled_image_batch.shape[0], num_flattened_values))
    return scaled_image_batch


def mean_loss(par, data):
    batched_loss = vmap(loss, in_axes=(None, 0))
    return np.mean(batched_loss(par, data))


def test_performance(params, epoch_results):
    test_losses = []
    ssim_scores = []
    for x in get_test_batches(batch_size=batch_size):
        y_batch = make_y_batch(x['image'])
        seq_batch = make_sequence_batch(x['image'])
        batched_predict = vmap(model.predict, in_axes=(None, 0))
        predictions_batch, prediction_prediction_batch = batched_predict(params, seq_batch)
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


def train():
    params = build_model(input_size=32 * 32 * 3, latent_vector_sizes=[64])
    epoch_results = {"mean test loss": [], "mean SSIM score": []}
    test_performance(params, epoch_results)
    for epoch in range(num_epochs):
        for x in get_train_batches(batch_size=batch_size):
            y_batch = make_y_batch(x['image'])
            seq_batch = make_sequence_batch(x['image'])
            params = update(params, (seq_batch, y_batch))
        test_performance(params, epoch_results)
    return epoch_results


if __name__ == "__main__":
    training_results = pd.DataFrame(train())
    print(training_results)
