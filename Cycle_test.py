import tensorflow as tf
import os
import numpy as np
from CycleGAN.CycleGenerator import Generator
from CycleGAN.data_preprocess import return_dataset
from CycleGAN.cycleGAN import generate_images
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

BASE_PATH = "/home/bai/Workspace/program/GANFamily/CycleGAN"

# Network.
print("load weights.")
generator_g = Generator()
generator_f = Generator()
y_g = generator_g(tf.convert_to_tensor(np.random.random((1, 256, 256, 3)), dtype=tf.float32))
y_f = generator_f(tf.convert_to_tensor(np.random.random((1, 256, 256, 3)), dtype=tf.float32))
generator_g.load_weights(os.path.join(BASE_PATH, "weights/Generator_g_50.h5"))
generator_f.load_weights(os.path.join(BASE_PATH, "weights/Generator_f_50.h5"))
print("load done.")

# data
train_horses, train_zebras, test_horses, test_zebras = return_dataset()

horses_iter = iter(test_horses)
zebras_iter = iter(test_zebras)
for i in range(50):
    sample_horse = next(horses_iter)                       # (1, 256, 256, 3)
    generate_images(generator_g, sample_horse, t=i)




