from dataclasses import dataclass
from json import load, dump
from logging import debug, info, error

from os.path import exists

_load_default_config = True
_default_config_file_path = './cfg.json'


@dataclass
class Configs:
    img_height: int = 256
    img_width: int = 256
    img_channels: int = 3

    img_resize_height: int = 286
    img_resize_width: int = 286

    batch_size: int = 16
    buffer_size: int = 16

    output_channels: int = 3

    save_at_step: int = 1000
    log_dir: str = "logs"
    log_dir_add_timestamp: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_dir_add_timestamp: bool = False
    checkpoint_prefix: str = 'ckpt'
    root_dir: str = "."

    max_steps: int = 40000
    save_every_n_step: int = 5000

    print_image_every_n_step: int = 1000

    url: str = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/'
    dataset: str = 'edges2handbags'
    extension: str = '.tar.gz'

    shape = [256, 256, 3]

    real_right: bool = True

    checkpoint_file_start: str = ""
    config_file_path: str = ""


#_default_config = Configs()


#def load_to_default_config(file_path):
#    _default_config = load_config(file_path)
#
#
#def set_default_config(config):
#    _default_config = config


def get_default_config():
    return Configs()


def load_config(file_path):
    cfg = None
    with open(file_path) as f:
        cfg = Configs(**load(f))
        cfg.config_file_path = file_path
    return cfg


def save_config(config, file_path):
    with open(file_path, 'w') as f:
        dump(config.__dict__, f)
    return config


#def get_default_width():
#    return get_default_config().img_width
#
#
#def get_default_height():
#    return get_default_config().img_height
#
#
#def get_default_buffer_size():
#    return get_default_config().buffer_size
#
#
#def get_default_batch_size():
#    return get_default_config().batch_size
#
#
#def get_default_output_channels():
#    return get_default_config().output_channels
#
#
#def get_default_channels():
#    return get_default_config().buffer_size
#
#
#def get_default_shape():
#    return [get_default_height(), get_default_width(), get_default_channels()]
#
#
#if _load_default_config:
#    if exists(_default_config_file_path):
#        debug(f'Loading default config file {_default_config_file_path}')
#        load_to_default_config(_default_config_file_path)
#    else:
#        debug(
#            f'Default config file does not exists {_default_config_file_path}')
#else:
#    debug('Skipping configuration loading')
