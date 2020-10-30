import utils
import BigGAN as model
import torch
import numpy as np
import glob
import os
import datetime
import pickle
import pprint

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


def get_logger(file_path="./", filename=None):
    filename = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + ".log" if not filename else filename
    full_path = file_path + filename

    def get_timestamp():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def logger(content, print_screen=True):
        current_time = get_timestamp()
        log = f"[{current_time}]: {pprint.pformat(content)}"
        if print_screen:
            print(log)
        with open(full_path, 'a+') as f:
            f.write(log + "\n")

    return logger


def load_cifar_samples(mode="train", data_dir='./data/cifar/cifar-10-batches-py'):
    HEIGHT = WIDTH = 32
    train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_filenames = ['test_batch']
    if mode == "all":
        filenames = train_filenames + test_filenames
    elif mode == "test":
        filenames = test_filenames
    else:
        filenames = train_filenames

    def unpickle(file):
        fo = open(file, 'rb')
        data_dict = pickle.load(fo, encoding='latin1')
        fo.close()
        return data_dict['data'], data_dict['labels']

    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0).reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1]).reshape(
        [-1, 32 * 32 * 3])
    labels = np.concatenate(all_labels, axis=0)

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    # images = 2. / 255 * images - 1
    assert (np.min(images[0]) >= 0 and np.max(images[0]) > 10)
    images = images[:].reshape([-1, HEIGHT, WIDTH, 3]).transpose([0, 3, 1, 2])
    return images, labels


def load_f100_samples(mode="train", dataset_path=r'D:\Data\facescrub100\facescrub100_64.npz'):
    dataset = np.load(dataset_path)
    train_data, train_labels, test_data, test_labels = dataset['trainx'], dataset['trainy'], dataset['testx'], dataset[
        'testy']
    if mode == "all":
        images = train_data + test_data
        labels = train_labels + test_labels
    elif mode == "test":
        images = test_data
        labels = test_labels
    else:
        images = train_data
        labels = train_labels

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    images = np.moveaxis(images, -1, 1)
    images = ((images + 1) / 2 * 255).astype(np.uint8)

    return images, labels


def load_generated_samples(experiment_name, file_name):
    npz_path = rf'.\samples\{experiment_name}\{file_name}.npz'
    images = np.load(npz_path)["samples"]
    images = np.moveaxis(images, -1, 1)
    return images


if __name__ == '__main__':
    load_f100_samples()
