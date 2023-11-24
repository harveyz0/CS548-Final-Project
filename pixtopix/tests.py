from logging import debug, warning, info, error
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from time import time
from PIL import Image

from pixtopix.pipeline import Generator, neg_one_to_one
from pixtopix.processimages import (random_jitter, load, load_dataset,
                                    load_test_dataset, load_train_dataset, normalize_to_255, write_images)


def test_generator(input_image_file="train/100.jpg"):
    debug(f'Loading image {input_image_file}')
    input_image, real_image = load(input_image_file)
    generator = Generator()
    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
    gen_output = generator(input_image[tf.newaxis, ...], training=False)
    plt.imshow(gen_output[0, ...])
    plt.show()


def show_image(*images):
    plt.figure(figsize=(6, 6))
    i = 1
    subplt = 1
    while i < len(images):
        input_image, real_image = random_jitter(images[i - 1], images[i])
        plt.subplot(subplt, 2, 1)
        plt.imshow(input_image / 255.0)
        plt.axis('off')
        plt.subplot(subplt, 2, 2)
        plt.imshow(real_image / 255.0)
        plt.axis('off')
        i += 2
        subplt += 1
    plt.show()


def test_generate_images(input_image_file):
    #img = load_test_dataset("facades")
    #return img
    input_image, real_image = load(input_image_file)
    generator = Generator()
    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
    uint_generate_images(generator, input_image, real_image)


def dump_images(*images):
    #norm_one = normalize_to_1(images[0].numpy())
    #print(norm_one.shape, norm_one.dtype, np.min(norm_one), np.max(norm_one))
    #Image.fromarray(norm_one).save(f'./poops_one{int(time())}.jpg')
    #norm_twofive = images[0].numpy()
    #Image.fromarray(norm_twofive).save(f'./poops_twofive{int(time())}.jpg')
    write_images(*images).save('./stich.jpg')
    #with open(f'./poops{int(time())}.jpg', 'bw') as f:
    #  f.write(tf.io.encode_jpeg(images[0]))


def uint_generate_images(model, test_input, target_image):
    #test_input = tf.expand_dims(test_input, axis=0)
    #target_image = tf.expand_dims(target_image, axis=0)
    prediction = model(tf.expand_dims(test_input, 0), training=True)
    plt.figure(figsize=(15, 15))

    #dump_images(*display_list)
    write_images(test_input.numpy(), target_image.numpy(), normalize_to_255(prediction[0].numpy())).save('./stitch.jpg')
    #dump_images(convertScaleAbs(neg_one_to_one(prediction[0]).numpy()))

    return
    display_list = [test_input, target_image, prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicated Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        #plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def test_load_dataset(data_dir_path: str):
    train_dataset = load_train_dataset(data_dir_path)
    test_dataset = load_train_dataset(data_dir_path)


def runable():
    sample_image = tf.io.read_file(str(PATH / 'train/1.jpg'))

    sample_image = tf.io.decode_jpeg(sample_image)
    print(sample_image.shape)

    plt.figure()
    plt.imshow(sample_image)

    inp, re = load(str(PATH / 'train/100.jpg'))
    # Casting to int for matplotlib to display the images
    plt.figure()
    plt.imshow(inp / 255.0)
    plt.figure()
    plt.imshow(re / 255.0)

    # As described in the [pix2pix paper](https://arxiv.org/abs/1611.07004){:.external}, you need to apply random jittering and mirroring to preprocess the training set.
    #
    # Define several functions that:
    #
    # 1. Resize each `256 x 256` image to a larger height and width—`286 x 286`.
    # 2. Randomly crop it back to `256 x 256`.
    # 3. Randomly flip the image horizontally i.e. left to right (random mirroring).
    # 4. Normalize the images to the `[-1, 1]` range.

    # The facade training set consist of 400 images
    BUFFER_SIZE = 400
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    # Each image is 256x256 in size
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    plt.figure(figsize=(6, 6))
    for i in range(4):
        rj_inp, rj_re = random_jitter(inp, re)
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_inp / 255.0)
        plt.axis('off')
    plt.show()

    train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    try:
        test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # ## Build the generator
    #
    # The generator of your pix2pix cGAN is a _modified_ [U-Net](https://arxiv.org/abs/1505.04597){:.external}. A U-Net consists of an encoder (downsampler) and decoder (upsampler). (You can find out more about it in the [Image segmentation](../images/segmentation.ipynb) tutorial and on the [U-Net project website](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/){:.external}.)
    #
    # - Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
    # - Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
    # - There are skip connections between the encoder and decoder (as in the U-Net).

    # Define the downsampler (encoder):

    OUTPUT_CHANNELS = 3

    down_model = downsample(3, 4)
    down_result = down_model(tf.expand_dims(inp, 0))
    print(down_result.shape)

    # Define the upsampler (decoder):

    up_model = upsample(3, 4)
    up_result = up_model(down_result)
    print(up_result.shape)

    generator = Generator()
    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

    # Test the generator:

    gen_output = generator(inp[tf.newaxis, ...], training=False)
    plt.imshow(gen_output[0, ...])

    LAMBDA = 100

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

    # Test the discriminator:

    disc_out = discriminator([inp[tf.newaxis, ...], gen_output],
                             training=False)
    plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

    for example_input, example_target in test_dataset.take(1):
        generate_images(generator, example_input, example_target)

    # ## Training
    #
    # - For each example input generates an output.
    # - The discriminator receives the `input_image` and the generated image as the first input. The second input is the `input_image` and the `target_image`.
    # - Next, calculate the generator and the discriminator loss.
    # - Then, calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
    # - Finally, log the losses to TensorBoard.

    log_dir = "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
