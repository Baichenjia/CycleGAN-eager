import tensorflow as tf
import os
from CycleGAN.data_preprocess import return_dataset
from CycleGAN.CycleGenerator import Generator
from CycleGAN.CycleDiscriminator import Discriminator
import matplotlib.pyplot as plt
# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers
BASE_PATH = "/home/bai/Workspace/program/GANFamily/CycleGAN"

# load dataset
train_horses, train_zebras, test_horses, test_zebras = return_dataset()

# Network. 包括两个Generator: G(X->Y) 和 F(Y->X). 两个Discriminator: D_X (分辨X和F(Y)), D_Y (分辨Y和G(X))
generator_g = Generator()
generator_f = Generator()
discriminator_x = Discriminator()
discriminator_y = Discriminator()


def discriminator_loss(real, generated):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real), logits=real)
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(generated), logits=generated)
    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    return total_disc_loss*0.5


def generator_loss(generated):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.ones_like(generated), logits=generated))


def cal_cycle_loss(real_image, cycle_image, cycle_weight=10.):
    return cycle_weight * tf.reduce_mean(tf.abs(real_image - cycle_image))


def identity_loss(real_image, same_image, identity_weight=5.):
    """
        As shown above, generator G is responsible for translating image X to image Y.
        Identity loss says that, if you fed image Y to generator G, it should yield
        the real image $Y$ or something close to image Y.
    """
    return identity_weight * tf.reduce_mean(tf.abs(real_image - same_image))


# optimizer
generator_g_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
generator_f_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

discriminator_x_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
discriminator_y_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)


def generate_images(model, test_input, t=0):
    """测试当前Generator的效果"""
    prediction = model(test_input)
    plt.figure(figsize=(12, 6))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(os.path.join(BASE_PATH, "generated_img/"+str(t)+".jpg"))
    plt.show()


def train_step(real_x, real_y):
    # Gradient Tape 在这里要计算 4 个部件
    # persistent is set to True because the tape is used more than once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x)    # G: X->Y
        cycle_x = generator_f(fake_y)   # F: Y->X

        fake_x = generator_f(real_y)    # F: Y->X
        cycle_y = generator_g(fake_x)   # G: X->Y

        # disc
        disc_real_x = discriminator_x(real_x)   # D_X(real_x)
        disc_real_y = discriminator_y(real_y)   # D_Y(real_y)
        disc_fake_x = discriminator_x(fake_x)   # D_X(fake_x)
        disc_fake_y = discriminator_y(fake_y)   # D_Y(fake_y)

        # generator loss
        gen_g_loss = generator_loss(disc_fake_y)  # G 产生的是 fake_y, 送入 D_Y 中得到的值计算损失
        gen_f_loss = generator_loss(disc_fake_x)  # F 产生的是 fake_x, 送入 D_X 中得到的值计算损失

        # cycle loss
        total_cycle_loss = cal_cycle_loss(real_x, cycle_x) + cal_cycle_loss(real_y, cycle_y)

        # identity loss
        same_x = generator_f(real_x)  # F: input x should output x
        same_y = generator_g(real_y)  # G: input y should output y

        # 分别统计每个组件的损失. 这里需要画图才能看清楚每个损失会影响哪几个单元.
        # Generator 会受到 Cycle,Generator,Identity损失的影响, Discriminator受到的影响单一
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)  # D_X 判断 real_x 和 fake_x
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)  # D_Y 判断 real_y 和 fake_y
        # total_losses = total_gen_f_loss + total_gen_g_loss + disc_x_loss + disc_y_loss

    # gradient
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # apply gradient
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))
    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


test_iter = iter(train_horses)
if __name__ == '__main__':
    for epoch in range(51):
        print("-----\n EPOCH:", epoch)
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
            total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(image_x, image_y)
            n += 1
            if n % 100 == 0:
                print("gen_g_loss:", total_gen_g_loss.numpy(), ", gen_f_loss:", total_gen_f_loss.numpy(),
                      ", disc_x_loss:", disc_x_loss.numpy(), ", disc_y_loss:", disc_y_loss.numpy())

        # test and save
        sample_horse = next(test_iter)  # (1, 256, 256, 3)
        generate_images(generator_g, sample_horse, t=epoch)
        if epoch % 10 == 0:
            generator_g.save_weights(os.path.join(BASE_PATH, "weights/Generator_g_"+str(epoch)+".h5"))
            generator_f.save_weights(os.path.join(BASE_PATH, "weights/Generator_f_"+str(epoch)+".h5"))
            discriminator_x.save_weights(os.path.join(BASE_PATH, "weights/Discriminator_x_"+str(epoch)+".h5"))
            discriminator_y.save_weights(os.path.join(BASE_PATH, "weights/Discriminator_y_"+str(epoch)+".h5"))
