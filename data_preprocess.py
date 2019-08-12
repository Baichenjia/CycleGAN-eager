import tensorflow as tf
import os
import tensorflow_datasets as tfds
# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# pre-process
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image


def normalize(image):
    """normalizing the images to [-1, 1]"""
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


def return_dataset():
    """
    load dataset from tensorflow_dataset, and preprocess through pre-defined functions.
    :return:
    """
    dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']
    train_horses = train_horses.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    train_zebras = train_zebras.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    test_horses = test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    return train_horses, train_zebras, test_horses, test_zebras

#
# if __name__ == '__main__':
#     train_horses, train_zebras, test_horses, test_zebras = return_dataset()
#     # test
#     sample_horse = next(iter(train_horses))   # (1, 256, 256, 3)
#     sample_zebra = next(iter(train_zebras))   # (1, 256, 256, 3)
#     print("sample_horse.shape:", sample_horse.shape, ", sample_zebra.shape:", sample_zebra.shape)
#     plt.figure(figsize=(12, 10))
#     plt.subplot(221)
#     plt.title('Horse')
#     plt.imshow(sample_horse[0] * 0.5 + 0.5)
#
#     plt.subplot(222)
#     plt.title('Horse with random jitter')
#     plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
#
#     plt.subplot(223)
#     plt.title('Zebra')
#     plt.imshow(sample_zebra[0] * 0.5 + 0.5)
#
#     plt.subplot(224)
#     plt.title('Zebra with random jitter')
#     plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)
#
#     plt.show()

