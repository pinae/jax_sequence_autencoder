from cifar10_loader import get_train_batches, get_test_batches
from model import build_model, loss
import jax.numpy as np
from jax import grad, jit, vmap


learning_rate = 0.01
num_epochs = 8
batch_size = 256
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


@jit
def update(params, train_data):
    grads = grad(mean_loss)(params, train_data)
    return [p - learning_rate * dp for p, dp in zip(params, grads)]


if __name__ == "__main__":
    params = build_model(input_size=32*32*3, latent_vector_sizes=[64])
    for epoch in range(num_epochs):
        for x in get_train_batches(batch_size=batch_size):
            y_batch = make_y_batch(x['image'])
            seq_batch = make_sequence_batch(x['image'])
            params = update(params, (seq_batch, y_batch))
        test_losses = []
        for x in get_test_batches(batch_size=batch_size):
            y_batch = make_y_batch(x['image'])
            seq_batch = make_sequence_batch(x['image'])
            test_losses.append(mean_loss(params, (seq_batch, y_batch)))
        print(np.mean(np.asarray(test_losses, dtype=np.float32)))
