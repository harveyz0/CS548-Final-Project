_DEFAULT_IMG_HEIGHT = 256
_DEFAULT_IMG_WIDTH = 256
_DEFAULT_IMG_CHANNELS = 3

_BATCH_SIZE = 32
_BUFFER_SIZE = 32

_OUTPUT_CHANELS = 3


def get_default_width():
    return _DEFAULT_IMG_WIDTH


def get_default_height():
    return _DEFAULT_IMG_HEIGHT


def get_default_buffer_size():
    return _BUFFER_SIZE


def get_default_batch_size():
    return _BATCH_SIZE


def get_default_output_channels():
    return _OUTPUT_CHANELS


def get_default_channels():
    return _DEFAULT_IMG_CHANNELS


def get_default_shape():
    return [get_default_height(),
            get_default_width(),
            get_default_channels()]
