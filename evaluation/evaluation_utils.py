import utils
import BigGAN as model
import torch
import numpy as np
import glob
import os


def get_config():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config["experiment_name"] = input("Please input experiment_name: ")
    config["model_parameter_name"] = find_best_model_parameter(config["experiment_name"])
    return config


def find_best_model_parameter(experiment_name, ema=True):
    """
    Return the full path!
    """
    prefix = "G_"
    if ema:
        prefix += "ema_"
    prefix += "best"
    dir_name = f"./weights/{experiment_name}/"
    files = glob.glob(dir_name + prefix + "*.pth")
    files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
    print(files[0])
    return files[0]


def load_generator(config, device='cuda'):
    print("Loading model...")
    generator = model.Generator(**config).to(device)
    generator.load_state_dict(torch.load(config['model_parameter_name']), strict=True)
    print("Loading done.")
    return generator


def sample_labels(num_classes, batch_size):
    labels = np.random.randint(low=0, high=num_classes, size=batch_size)
    pseudo_labels = torch.from_numpy(labels)
    pseudo_labels = pseudo_labels.type(torch.long).cuda()
    return pseudo_labels


def print_banner(title="", char="*", total_length=100):
    left_num = (total_length - len(title) - 2) // 2
    right_num = left_num if len(title) % 2 == 0 else left_num + 1
    print(left_num * char + " " + title + " " + right_num * char)
