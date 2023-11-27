from logging import info
from datetime import datetime
from os.path import join
from time import time
import tensorflow as tf

from pixtopix.processimages import dump_images


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters,
                               size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters,
                                        size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def get_down_stack(apply_batchnorm=False):
    return [
        downsample(64, 4, apply_batchnorm=apply_batchnorm),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]


def get_up_stack(apply_dropout=True):
    return [
        upsample(512, 4, apply_dropout=apply_dropout),
        upsample(512, 4, apply_dropout=apply_dropout),
        upsample(512, 4, apply_dropout=apply_dropout),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]


def build_generator(output_channels, shape=[256, 256, 3]):
    inputs = tf.keras.layers.Input(shape=shape)

    down_stack = get_down_stack()

    up_stack = get_up_stack()

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels,
                                           4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    outputs = inputs

    skips = []
    for down in down_stack:
        outputs = down(outputs)
        skips.append(outputs)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        outputs = up(outputs)
        outputs = tf.keras.layers.Concatenate()([outputs, skip])

    outputs = last(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def generator_loss(disc_generated_output,
                   gen_output,
                   target,
                   loss_object=None,
                   lambda_loss=100):
    if not loss_object:
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output),
                           disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (lambda_loss * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def build_discriminator(shape):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=shape, name='input_image')
    tar = tf.keras.layers.Input(shape=shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512,
                                  4,
                                  strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1,
                                  4,
                                  strides=1,
                                  kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output,
                       disc_generated_output,
                       loss_object=None):
    if not loss_object:
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output),
                                 disc_generated_output)

    return real_loss + generated_loss


class Trainer:

    def __init__(self, generator, discriminator, generator_optimizer,
                 discriminator_optimizer, log_dir: str,
                 checkpoint_directory: str, checkpoint_prefix: str):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint = self.init_checkpoints()

    @classmethod
    def from_config(cls, cfg):
        return cls(build_generator(cfg.output_channels),
                   build_discriminator(cfg.shape),
                   tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                   tf.keras.optimizers.Adam(2e-4, beta_1=0.5), cfg.log_dir,
                   cfg.checkpoint_dir, cfg.checkpoint_prefix)

    def train_step(self, input_image, target, step):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target],
                                                  training=True)
            disc_generated_output = self.discriminator([input_image, target],
                                                       training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output,
                                           disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)

        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables))

        summary_writer = tf.summary.create_file_writer(
            join(self.log_dir, "fit",
                 datetime.now().strftime("%Y%m%d-%H%M%S")))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss',
                              gen_total_loss,
                              step=step // 1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

    def fit(self, train_ds, test_ds, steps, save_every_n_step,
            print_image_every_n_step):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time()

        for step, (input_image,
                   target) in train_ds.repeat().take(steps).enumerate():
            self.train_step(input_image, target, step)

            if (step) % print_image_every_n_step == 0:
                #Display is some IPython thing that I'll probably delete
                #display.clear_output(wait=True)

                if step != 0:
                    info(
                        f'Time taken for 1000 steps: {time() - start:.2f} sec')
                    print()

                start = time()

                predicted = generate_images(self.generator, example_input)
                dump_images(example_input[0], example_target[0], predicted,
                            join(self.log_dir, f'image_dump_{step}.jpg'))

                info(f'Steps: {step//1000}k')

            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)
                if (step + 1) % 100 == 0:
                    print()
            if (step + 1) % save_every_n_step == 0:
                self.checkpoint.save(file_prefix=join(self.checkpoint_directory,
                                                 self.checkpoint_prefix))

    def init_checkpoints(self):
        return tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

    def restore_from_checkpoint(self, file_path):
        self.checkpoint.restore(file_path)


def generate_images(model, test_input):
    prediction = model(test_input if len(test_input.shape) == 4 else
                       tf.expand_dims(test_input, 0),
                       training=True)
    return prediction[0]


def neg_one_to_one(tensor):
    """This will take a tensor that's -1 to 1 and will convert it to 0 to 1."""
    return tensor * 0.5 + 0.5
