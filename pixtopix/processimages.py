import sys
import logging
from logging import debug, warning, info, error

import tensorflow as tf

from os.path import splitext
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

from .defaults import get_default_width, get_default_height


def load_dataset(url='http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/',
                 dataset='edges2handbags',
                 extension='.tar.gz'):
    return pathlib.Path(
        tf.keras.utils.get_file(fname=dataset + extension,
                                origin=f"{url}{dataset}{extension}",
                                extract=True)).parent / dataset


def load_image(file_path: str, extension=None):
    """Decode an image using the extension as guidance for which
    decode function to use."""
    decode_fn = None
    if not extension:
        extension = splitext(file_path)[-1]
    debug(f"Decoding as extension {extension}")

    match extension:
        case '.jpg' | '.jpeg':
            decode_fn = tf.io.decode_jpeg
        case '.png':
            decode_fn = tf.io.decode_png
        case '.gif':
            decode_fn = tf.io.decode_gif
        case '.bmp':
            decode_fn = tf.io.decode_bmp
        case _:
            raise ValueError(f"Do not support type {extension}")

    return decode_fn(tf.io.read_file(file_path))


def split_image(raw_image_data):
    """Take an image from a decode function. Then will split in
    image in half and return the images as (left side, right side)"""
    half_width = tf.shape(raw_image_data)[1] // 2

    return (tf.cast(raw_image_data[:, half_width:, :], tf.float32),
            tf.cast(raw_image_data[:, :half_width, :], tf.float32))


def resize(image,
           height=get_default_height(),
           width=get_default_width()):
    return tf.image.resize(image, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def resize_all(*images,
               height=get_default_height(),
               width=get_default_width()):
    """Tried to just make it loop through but then Tensorflow yelled at me
    for using loops."""
    return tuple(resize(img, height, width) for img in images)


def load(file_path: str, extension=None):
    return split_image(load_image(file_path, extension))


def random_crop(*images,
                height=get_default_height(),
                width=get_default_width()):
    stack_values = tf.stack(images, axis=0)
    return tf.image.random_crop(stack_values,
                                size=[len(stack_values), height, width,
                                      3])[0:2]


def normalize(image):
    return (image / 127.5) - 1


@tf.function()
def random_jitter(input_image, real_image):
    input_image = resize(input_image, 286, 286)

    real_image = resize(real_image, 286, 286)
    images = random_crop(input_image, real_image)
    input_image = images[0]
    real_image = images[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image = resize(input_image)
    real_image = resize(real_image)
    return normalize(input_image), normalize(real_image)


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    return normalize(input_image), normalize(real_image)


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


def main(args):
    all_images = []
    data_path = load_dataset(dataset='facades')
    input_image, real_image = split_image(
        load_image(str(data_path / 'train/100.jpg')))
    all_images += random_jitter(input_image, real_image)
    input_image, real_image = split_image(
        load_image(str(data_path / 'train/101.jpg')))
    all_images += random_jitter(input_image, real_image)
    show_image(*all_images)


if __name__ == '__main__':
    main(sys.argv)
