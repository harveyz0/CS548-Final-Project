import pathlib
from os.path import join
from logging import debug
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def load_online_dataset(
        url='http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/',
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


def split_image(raw_image_data, real_right=False):
    """Take an image from a decode function. Then will split in
    image in half and always return the real image as the last
    item in the tuple, (fake, real)"""
    half_width = tf.shape(raw_image_data)[1] // 2

    #show_images([raw_image_data[:, half_width:, :], raw_image_data[:, :half_width, :]], ['left', 'right'])
    #It appears the cast must be into float32 otherwise the normalization does not work.
    right = tf.cast(raw_image_data[:, half_width:, :], tf.float32)
    left = tf.cast(raw_image_data[:, :half_width, :], tf.float32)
    if real_right:
        return left, right
    return right, left


@tf.function()
def resize(image, height=256, width=256, channels=3):
    return tf.image.resize(image, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


@tf.function()
def resize_all(*images, height, width):
    """Tried to just make it loop through but then Tensorflow yelled at me
    for using loops."""
    return tuple(resize(img, height, width) for img in images)


def load(file_path: str, extension=None, real_right=False):
    return split_image(load_jpg_image(file_path), real_right=real_right)


def random_crop(*images, height, width):
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
def random_jitter(input_image, real_image, height, width, resize_height,
                  resize_width):
    input_image, real_image = resize_all(input_image,
                                         real_image,
                                         height=resize_height,
                                         width=resize_width)
    images = random_crop(input_image, real_image, height=height, width=width)
    input_image = images[0]
    real_image = images[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image


@tf.function()
def load_normalized_images(image_file, real_right=False):
    input_image, real_image = load(image_file, real_right=real_right)
    return normalize(input_image), normalize(real_image)


@tf.function()
def load_image_test(image_file,
                    resize_height,
                    resize_width,
                    channels=3,
                    real_right=False):
    input_image, real_image = load(image_file)
    return normalize(resize(input_image, resize_height, resize_width, channels)), normalize(resize(real_image, resize_height, resize_width, channels))


@tf.function()
def load_image_train(image_file,
                     height,
                     width,
                     resize_height,
                     resize_width,
                     real_right=False):
    input_image, real_image = load(image_file, real_right=real_right)
    input_image, real_image = random_jitter(input_image, real_image, height, width, resize_height, resize_width)
    return normalize(input_image), normalize(real_image)


def load_train_dataset(path_to_data: str,
                       buffer_size: int,
                       batch_size: int,
                       height: int,
                       width: int,
                       resize_height: int,
                       resize_width: int,
                       real_right=False):
    return tf.data.Dataset.list_files(join(path_to_data, "train", "*.jpg")) \
                          .map(lambda img: load_image_train(img, height, width, resize_height, resize_width, real_right),
                               num_parallel_calls=tf.data.AUTOTUNE) \
                          .shuffle(buffer_size) \
                          .batch(batch_size)


def load_test_dataset(path_to_data: str,
                      batch_size: int,
                      resize_height: int,
                      resize_width: int,
                      channels: int,
                      real_right=False):
    test_dataset = None
    path = join(path_to_data, "val", "*.jpg")
    try:
        test_dataset = tf.data.Dataset.list_files(path)
    except tf.errors.InvalidArgumentError:
        path = join(path_to_data, "test", "*.jpg")
        test_dataset = tf.data.Dataset.list_files(path)
    debug(f'Loaded from {path}')
    return test_dataset.map(lambda img: load_image_test(img, resize_height, resize_width, channels, real_right=real_right)) \
                       .batch(batch_size)


def load_dataset(path_to_data: str, batch_size: int, real_right=False):
    glob = join(path_to_data, "*.jpg")
    dataset = tf.data.Dataset.list_files(glob)
    return dataset.map(lambda img: load_normalized_images(img, real_right=real_right)) \
                  .batch(batch_size)


def normalize_to_1(np_arr):
    if isinstance(np_arr, tf.Tensor):
        np_arr = np_arr.numpy()
    np_arr = np_arr.astype(np.float32)
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
    debug(f'Saving file to {file_path}')
    image.save(file_path)


def write_images(*images):
    """I just want a bunch of images strung together. I mostly
    stole this from make_imge_grid
    https://github.com/huggingface/diffusers/blob/2a7f43a73bda387385a47a15d7b6fe9be9c65eb2/src/diffusers/utils/pil_utils.py#L53
    """
    max_height = 0
    pil_images = []
    for img in images:
        arr = img
        if isinstance(img, tf.Tensor):
            arr = img.numpy()
        pil_images.append(Image.fromarray(arr))
        if max_height < pil_images[-1].size[1]:
            max_height = pil_images[-1].size[1]

    width = pil_images[0].size[0]
    grid = Image.new("RGB", size=(width * len(pil_images), max_height))

    for step, img in enumerate(pil_images):
        grid.paste(img, box=(width * step, 0))
    return grid
