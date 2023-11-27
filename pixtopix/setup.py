import tensorflow as tf

from pixtopix.defaults import get_default_config
from time import time
from os import makedirs
from os.path import join
from logging import debug


def build_log_directories(log_dir: str,
                          log_dir_add_timestamp: bool,
                          root_dir: str = "."):
    name = None

    if log_dir:
        name = join(root_dir, log_dir)
        if log_dir_add_timestamp:
            epoch = time()
            name = join(root_dir, f'{log_dir}-{epoch}')
        debug(f'Building logs directory tree {name}')
        makedirs(name, exist_ok=True)
        log_dir = name
    return name


def build_checkpoint_directories(checkpoint_dir,
                                 checkpoint_dir_add_timestamp,
                                 root_dir="."):
    name = None

    if checkpoint_dir:
        name = join(root_dir, checkpoint_dir)
        if checkpoint_dir_add_timestamp:
            epoch = time()
            name = join(root_dir, f'{checkpoint_dir}-{epoch}')
        debug(f'Building checkpoint directory tree {name}')
        makedirs(name, exist_ok=True)
        checkpoint_dir = name

    return name


