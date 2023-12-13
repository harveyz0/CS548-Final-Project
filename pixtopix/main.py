import sys
from argparse import ArgumentParser
from logging import DEBUG, getLogger, ERROR, INFO, error
from pixtopix.defaults import get_default_config, load_config, dump_config

getLogger().setLevel(level=DEBUG)
getLogger("tensorflow").setLevel(ERROR)
getLogger("matplotlib.font_manager").setLevel(ERROR)
getLogger("parso.python.diff").setLevel(ERROR)

def arg_parser(args):
    parser = ArgumentParser(prog='pixtopix', description='Module to build and run a pix2pix model')
    parser.add_argument('-c', '--checkpoint', action='store', help='Start from this checkpoint file. Do NOT add the .index to the file path you only need the ckpt-1 or whatever number.')
    parser.add_argument('-f', '--config', action='store', help='Load the provided config file')
    parser.add_argument('-d', '--dump-config', action='store_true', help='Dump the config to stdout')
    parser.add_argument('-l', '--load-model', nargs='*', help='Load a model from a directory and then try to push it through all the images passed in')
    parser.add_argument('-g', '--generate', action='store_true', help='Generate a bunch of images from a directory of inputs. Uses the url dataset and extension values in the passed in config')
    parser.add_argument('-e', '--eval', action='store', help='Eval a directory and print FID and KID')
    #parser.add_argument('images', metavar='I', type=str, nargs='+', help='A model and images for load model')
    return parser.parse_args(args)


def default_run(parser):
    from pixtopix.tests import full_run
    full_run(parser.config, parser.checkpoint)


def eval_arg(parser):
    from pixtopix.eval import eval_generated
    eval_generated(parser.eval)

def dump_stdout_config(parser):
    print(dump_config(load_config_arg(parser)))


def load_config_arg(parser):
    cfg = None
    cfg_file = parser.config
    if cfg_file:
        cfg = load_config(cfg_file)
    else:
        cfg = get_default_config()

    if parser.load_model:
        cfg.model = parser.load_model[0]
        if 1 < len(parser.load_model):
            cfg.images = parser.load_model[1:]
    return cfg


def load_model_arg(parser):
    from pixtopix.tests import load_model
    cfg = load_config_arg(parser)
    if 1 < len(cfg.model):
        error('Load model needs list of images. Please pass in image files.')
    load_model(cfg.model, *cfg.images)


def generate_n_images_arg(parser):
    from pixtopix.tests import generate_n_images
    generate_n_images(load_config_arg(parser))


def main(args=sys.argv):
    parser = arg_parser(args[1:])

    if parser.dump_config:
        dump_stdout_config(parser)
    elif parser.generate:
        generate_n_images_arg(parser)
    elif parser.load_model:
        load_model_arg(parser)
    elif parser.eval:
        eval_arg(parser)
    else:
        default_run(parser)
