from logging import info, debug
from datetime import datetime
from os.path import join
from time import time
import tensorflow as tf

from pixtopix.processimages import (dump_images, write_images, load,
                                    normalize_to_255, normalize_to_1,
                                    normalize)


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

    # I think this adds skip connections but I don't understand how
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
        self.generator.compile(optimizer=self.generator_optimizer)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_prefix = checkpoint_prefix
        self.steps = tf.Variable(0, dtype=tf.int64, name='steps')
        self.checkpoint = self.init_checkpoints()

    @classmethod
    def from_config(cls, cfg):
        return cls(build_generator(cfg.output_channels),
                   build_discriminator(cfg.shape),
                   tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                   tf.keras.optimizers.Adam(2e-4, beta_1=0.5), cfg.log_dir,
                   cfg.checkpoint_dir, cfg.checkpoint_prefix)

    @classmethod
    def from_checkpoint(cls, file_path, config):
        train = cls.from_config(config)
        checkpoint = train.init_checkpoints()
        checkpoint.restore(file_path)

        return train

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
        #hardcoded_target, hardcoded_input = load('/home/zac/Documents/classes/CS548-12/CS548-Final-Project/heartread.jpg')
        #hardcoded_input = tf.cast(hardcoded_input, tf.uint8)
        #hardcoded_target = tf.cast(hardcoded_target, tf.uint8)

        start = time()

        for step, (input_image, target) in train_ds.repeat().take(
                abs(self.steps - steps)).enumerate(start=self.steps):
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

                #hardcoded = normalize(hardcoded_input.numpy())
                #predicted = normalize_to_255(generate_images(self.generator, tf.expand_dims(hardcoded, 0)).numpy())
                #write_images(tf.cast(hardcoded_input, tf.uint8), tf.cast(hardcoded_target, tf.uint8), predicted).save(join(self.log_dir, f'hardcode_{step}.jpg'))

                info(f'Steps: {step//1000}k')

            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)
                if (step + 1) % 100 == 0:
                    print()
            if (step + 1) % save_every_n_step == 0:
                self.steps.assign(step)
                chk_file = self.init_checkpoints().save(file_prefix=join(
                    self.checkpoint_dir, self.checkpoint_prefix))
                info(f'Saved checkpoint file to {chk_file}')
                gen_file = join(self.log_dir, f'model-{step + 1}.keras')
                self.generator.save(gen_file)
                info(f'Saved model file to {gen_file}')

    def init_checkpoints(self):
        to_save = {}
        for i in self.checkpoint_keys():
            to_save[i] = getattr(self, i)
        chk = tf.train.Checkpoint(**to_save)
        return chk

    @classmethod
    def checkpoint_keys(cls):
        return [
            'generator_optimizer',
            'discriminator_optimizer',
            'generator',
            'discriminator',
            'steps'
        ]

    def restore_from_checkpoint(self, file_path):
        vals = self.checkpoint.restore(file_path)
        return vals

    def build_tensorboard(self):
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                          histogram_freq=1)

    def write_model_visualizer(self):
        tf.keras.utils.plot_model(self.generator,
                                  show_shapes=True,
                                  to_file=join(self.log_dir, 'generator.jpg'),
                                  dpi=64)
        tf.keras.utils.plot_model(self.discriminator,
                                  show_shapes=True,
                                  to_file=join(self.log_dir,
                                               'discriminator.jpg'),
                                  dpi=64)


def generate_images(model, test_input):
    prediction = model(test_input if len(test_input.shape) == 4 else
                       tf.expand_dims(test_input, 0),
                       training=True)
    return prediction[0]


def neg_one_to_one(tensor):
    """This will take a tensor that's -1 to 1 and will convert it to 0 to 1."""
    return tensor * 0.5 + 0.5
