import sys
import logging
from logging import debug, warning, info, error

import tensorflow as tf
import numpy as np

from PIL import Image

from os.path import splitext, join
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

from pixtopix.defaults import (get_default_width, get_default_height, get_default_batch_size, get_default_buffer_size)

def load_online_dataset(url='http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/',
                 dataset='edges2handbags',
                 extension='.tar.gz'):
    return pathlib.Path(
        tf.keras.utils.get_file(fname=dataset + extension,
                                origin=f"{url}{dataset}{extension}",
                                extract=True)).parent / dataset


#@tf.function()
#def load_image(file_path: str, extension=None):
#    """Decode an image using the extension as guidance for which
#    decode function to use.
#    Except there seems to be a problem with the first item out of map.
#    Giving a Tensor("args_0:0", shape=(), dtype=string) and I don't
#    understand why. So we're just gonna short cut it that if it's not string
#    then just throw it at a decoder and see what happens. Good Luck!"""
#    #decode_fn = None
#    decode_fn = tf.io.decode_jpeg
#    if not extension and isinstance(file_path, str):
#        extension = splitext(file_path)[-1]
#    elif not isinstance(file_path, str):
#        debug(f"We don't know what this type of file_path is {type(file_path)} so we're just gonna give it to tensorflow")
#        return decode_fn(tf.io.read_file(file_path))
#    debug(f"Decoding as extension {extension} for file_path {file_path}")
#
#    match extension:
#        case '.jpg' | '.jpeg':
#            decode_fn = tf.io.decode_jpeg
#        case '.png':
#            decode_fn = tf.io.decode_png
#        case '.gif':
#            decode_fn = tf.io.decode_gif
#        case '.bmp':
#            decode_fn = tf.io.decode_bmp
#        case _:
#            raise ValueError(f"Do not support type {extension}")
#
#    return decode_fn(tf.io.read_file(file_path))


@tf.function()
def load_jpg_image(file_path: str):
    image = tf.io.decode_jpeg(tf.io.read_file(file_path))
    return image


def load_image_from_tensor(file_path, decode_fn=tf.io.decode_jpeg):
    return decode_fn(tf.io.read_file(file_path))


def split_image(raw_image_data):
    """Take an image from a decode function. Then will split in
    image in half and return the images as (left side, right side)"""
    half_width = tf.shape(raw_image_data)[1] // 2

    debug(f'type of the raw_image_data {raw_image_data.dtype}')

    #show_images([raw_image_data[:, half_width:, :], raw_image_data[:, :half_width, :]], ['left', 'right'])
    #It appears the cast must be into float32 otherwise the normalization does not work.
    return (tf.cast(raw_image_data[:, half_width:, :], tf.float32),
            tf.cast(raw_image_data[:, :half_width, :], tf.float32))


@tf.function()
def resize(image, height=256, width=256, channels=3):
    return tf.image.resize(image, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


@tf.function()
def resize_all(*images, height=get_default_height(),
               width=get_default_width()):
    """Tried to just make it loop through but then Tensorflow yelled at me
    for using loops."""
    return tuple(resize(img, height, width) for img in images)


def load(file_path: str, extension=None):
    return split_image(load_jpg_image(file_path))


def random_crop(*images,
                height=get_default_height(),
                width=get_default_width()):
    stack_values = tf.stack(images, axis=0)
    return tf.image.random_crop(stack_values,
                                size=[len(stack_values), height, width,
                                      3])[0:2]


def show_images(images, labels):
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(labels[i])
        plt.imshow(images[i].numpy())
        plt.axis('off')
    plt.show()


def normalize(image):
    return (image / 127.5) - 1


@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize_all(input_image, real_image, height=286, width=286)
    images = random_crop(input_image, real_image)
    input_image = images[0]
    real_image = images[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


@tf.function()
def load_image_test(image_file, real_right=False):
    images = load(image_file)
    real_index = 1
    input_index = 0
    if real_right:
        real_index = 0
        input_index = 1
    input_image = resize(images[input_index])
    real_image = resize(images[real_index])
    #show_images([input_image, real_image], ['input', 'real'])
    return normalize(input_image), normalize(real_image)


@tf.function()
def load_image_train(image_file, real_right=False):
    images = load(image_file)
    real_index = 1
    input_index = 0
    if real_right:
        real_index = 0
        input_index = 1
    input_image, real_image = random_jitter(images[input_index], images[real_index])
    #show_images([input_image, real_image], ['input', 'real'])
    return normalize(input_image), normalize(real_image)


def load_train_dataset(path_to_data: str,
                       buffer_size,
                       batch_size,
                       real_right=False):
    return tf.data.Dataset.list_files(join(path_to_data, "train", "*.jpg")) \
                          .map(lambda img: load_image_train(img, real_right),
                               num_parallel_calls=tf.data.AUTOTUNE) \
                          .shuffle(buffer_size) \
                          .batch(batch_size)


def load_test_dataset(path_to_data: str,
                      batch_size,
                      real_right=False):
    test_dataset = None
    try:
        test_dataset = tf.data.Dataset.list_files(join(path_to_data, "test", "*.jpg"))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(join(path_to_data, "val", "*.jpg"))
    return test_dataset.map(lambda img: load_image_test(img, real_right)) \
                       .batch(batch_size)


#def load_dataset(path_to_file: str,
#                 batch_size,
#                 default_map_to=lambda img: load_image_train(img, ):
#    files = tf.data.Dataset.list_files(path_to_file)
#    #files.map(load_image_from_tensor)
#    files.map(default_map_to)
#    return files.batch(batch_size)


def normalize_to_1(np_arr):
    np_arr += 1.0
    np_arr /= 2.0
    return np_arr


def normalize_to_255(np_arr):
    np_arr = normalize_to_1(np_arr)
    np_arr *= 255
    np_arr = np_arr.astype(np.uint8)
    return np_arr


def dump_images(image_input, image_target, predicted_image, file_path):
    image = write_images(normalize_to_255(image_input.numpy()),
                         normalize_to_255(image_target.numpy()),
                         normalize_to_255(predicted_image.numpy()))
    image.save(file_path)


def write_images(*images):
    """I just want a bunch of images strung together. I mostly
    stole this from make_imge_grid
    https://github.com/huggingface/diffusers/blob/2a7f43a73bda387385a47a15d7b6fe9be9c65eb2/src/diffusers/utils/pil_utils.py#L53
    """
    max_height = 0
    pil_images = []
    for img in images:
        pil_images.append(Image.fromarray(img))
        if max_height < pil_images[-1].size[1]:
            max_height = pil_images[-1].size[1]

    width = pil_images[0].size[0]
    grid = Image.new("RGB", size=(width * len(pil_images), max_height))

    for step, img in enumerate(pil_images):
        grid.paste(img, box=(width * step, 0))
    return grid
