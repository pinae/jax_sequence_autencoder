import tensorflow_datasets as tfds
import jax.numpy as np

data_dir = "data/cifar10"


def get_train_batches(batch_size=10):
    dataset = tfds.load(name="cifar10", split="train", as_supervised=False, data_dir=data_dir)
    dataset = dataset.batch(batch_size).prefetch(1)
    return tfds.as_numpy(dataset)


def get_test_batches(batch_size=10):
    dataset = tfds.load(name="cifar10", split="test", as_supervised=False, data_dir=data_dir)
    dataset = dataset.batch(batch_size).prefetch(1)
    return tfds.as_numpy(dataset)


if __name__ == "__main__":
    cifar10_data, dataset_info = tfds.load(name="cifar10", batch_size=-1, data_dir=data_dir, with_info=True)
    cifar10_data = tfds.as_numpy(cifar10_data)
    train_data, test_data = cifar10_data['train'], cifar10_data['test']
    print(dataset_info.features['image'].shape, type(dataset_info.features['image'].shape))
    h, w, c = dataset_info.features['image'].shape
    num_pixels = h * w * c

    train_images = np.reshape(train_data['image'], (len(train_data['image']), num_pixels))
    test_images = np.reshape(test_data['image'], (len(test_data['image']), num_pixels))

    print('Train:', train_images.shape)
    print('Test:', test_images.shape)
