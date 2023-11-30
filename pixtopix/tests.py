from logging import debug, warning, info, error
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from time import time
from PIL import Image
from os.path import join

from pixtopix.defaults import get_default_config, load_config, save_config
from pixtopix.setup import (build_log_directories,
                            build_checkpoint_directories)

from pixtopix.pipeline import (build_generator, build_discriminator,
                               neg_one_to_one, Trainer)
from pixtopix.processimages import (random_jitter, load, load_test_dataset,
                                    load_train_dataset, normalize_to_255,
                                    write_images, load_online_dataset)


def test_generator(input_image_file="train/100.jpg"):
    debug(f'Loading image {input_image_file}')
    input_image, real_image = load(input_image_file)
    generator = build_generator(3)
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
    generator = build_generator(3)
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
    write_images(test_input.numpy(), target_image.numpy(),
                 normalize_to_255(prediction[0].numpy())).save('./stitch.jpg')
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


def full_run(cfg_file=None, chk_file=None):
    cfg = None
    if cfg_file:
        cfg = load_config(cfg_file)
    else:
        cfg = get_default_config()
    cfg.log_dir = build_log_directories(cfg.log_dir, cfg.log_dir_add_timestamp,
                                        cfg.root_dir)

    cfg.checkpoint_dir = build_checkpoint_directories(
        cfg.checkpoint_dir, cfg.checkpoint_dir_add_timestamp, cfg.log_dir)

    path = load_online_dataset(cfg.url, cfg.dataset, cfg.extension)
    train_dataset = load_train_dataset(path, cfg.buffer_size, cfg.batch_size,
                                       cfg.img_height, cfg.img_width,
                                       cfg.img_resize_height,
                                       cfg.img_resize_width, cfg.real_right)
    test_dataset = load_test_dataset(path, cfg.batch_size, cfg.img_height,
                                     cfg.img_width, cfg.img_channels,
                                     cfg.real_right)

    trainer = Trainer.from_config(cfg)

    if chk_file:
        trainer.restore_from_checkpoint(chk_file)
        cfg.checkpoint_file_start = chk_file

    save_config(cfg, join(cfg.log_dir, 'operation.cfg'))

    trainer.fit(train_dataset, test_dataset, cfg.max_steps,
                cfg.save_every_n_step, cfg.print_image_every_n_step)
