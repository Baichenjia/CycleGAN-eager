import tensorflow as tf
from CycleGAN.instance_norm import InstanceNormalization
from CycleGAN.CycleGenerator import EncoderLayer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The Discriminator is a PatchGAN.
Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
The shape of the output after the last layer is (batch_size, 30, 30, 1)
Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called PatchGAN).
Discriminator receives 2 inputs.
Input image and the target image, which it should classify as real.
Input image and the generated image (output of generator), which it should classify as fake.
We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
"""


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        # downsample
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False) # (bs, 128, 128, 64)
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)        # (bs, 64, 64, 128)
        self.encoder_layer_3 = EncoderLayer(filters=128, kernel_size=4)        # (bs, 32, 32, 256)

        # conv block1
        self.zero_pad1 = layers.ZeroPadding2D()                                # (bs, 34, 34, 256)
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        self.bn1 = InstanceNormalization()                                 # (bs, 31, 31, 512)
        self.ac = layers.LeakyReLU()

        # block2
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()                       # (bs, 33, 33, 512)
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)  # (bs, 30, 30, 1)

    def call(self, x):
        """inputs can be real image and generated image.
           concat the inputs and target as input to the discriminator """
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        x = self.zero_pad2(x)
        x = self.last(x)
        return x


# if __name__ == "__main__":
#     from CycleGAN.data_preprocess import return_dataset
#     import matplotlib.pyplot as plt
#     train_horses, train_zebras, _, _ = return_dataset()
#     sample_horse = next(iter(train_horses))  # (1, 256, 256, 3)
#     sample_zebra = next(iter(train_zebras))  # (1, 256, 256, 3)
#     #
#     discriminator_x = Discriminator()
#     discriminator_y = Discriminator()
#     plt.figure(figsize=(8, 8))
#     plt.subplot(121)
#     plt.title('Is a real zebra?')
#     plt.imshow(discriminator_y(sample_zebra).numpy()[0, ..., -1], cmap='RdBu_r')
#     plt.subplot(122)
#     plt.title('Is a real horse?')
#     plt.imshow(discriminator_x(sample_horse).numpy()[0, ..., -1], cmap='RdBu_r')
#     plt.show()





