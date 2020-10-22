import numpy as np
import pickle
from evaluation.inception_score import get_inception_score
from evaluation.fid import get_fid

DATA_DIR = './data/cifar/cifar-10-batches-py'
HEIGHT = WIDTH = 32
DATA_DIM = HEIGHT * WIDTH * 3
BATCH_SIZE = 50


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']


def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0).reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1]).reshape(
        [-1, 32 * 32 * 3])
    labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) // batch_size):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], batch_size,
                        data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )


def inf_gen(MODE='TRAIN', BATCH_SIZE=BATCH_SIZE):
    if MODE == 'TRAIN':
        train_gen, _ = load(BATCH_SIZE, data_dir=DATA_DIR)
        while True:
            for original_images, labels in train_gen():
                yield 2. / 255 * original_images - 1, labels
    elif MODE == 'TEST':
        _, test_gen = load(BATCH_SIZE, data_dir=DATA_DIR)
        while True:
            for original_images, labels in test_gen():
                yield 2. / 255 * original_images - 1, labels


train_gen = inf_gen('TRAIN')
test_gen = inf_gen('TEST')


def get_training_set_is(n=50000, splits=10):
    all_samples = np.zeros([int(np.ceil(float(n) / BATCH_SIZE) * BATCH_SIZE), DATA_DIM], dtype=np.uint8)
    for i in range(int(np.ceil(float(n) / BATCH_SIZE))):  # inception score for num_batches of real data
        all_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = ((next(train_gen)[0] + 1) / 2 * 255).astype(np.uint8)
    return get_inception_score(all_samples[:n].reshape([-1, HEIGHT, WIDTH, 3]).transpose([0, 3, 1, 2]), splits)


def train_test_sets_fid(n, gen1, gen2):
    all_real_samples = np.zeros([n // BATCH_SIZE * BATCH_SIZE, DATA_DIM], dtype=np.uint8)
    all_fake_samples = np.zeros([n // BATCH_SIZE * BATCH_SIZE, DATA_DIM], dtype=np.uint8)
    for i in range(int(np.ceil(float(n) / BATCH_SIZE))):
        all_real_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = ((next(gen1)[0] + 1) / 2 * 255).astype(np.uint8)
        all_fake_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = ((next(gen2)[0] + 1) / 2 * 255).astype(np.uint8)
    sample_size = min(all_fake_samples.shape[0], all_real_samples.shape[0])
    return get_fid(all_real_samples[:sample_size].reshape([-1, HEIGHT, WIDTH, 3]).transpose([0, 3, 1, 2]),
                   all_fake_samples[:sample_size].reshape([-1, HEIGHT, WIDTH, 3]).transpose([0, 3, 1, 2]))


# Inception Score of the training set in 10 splits
is_mean, is_std = get_training_set_is()
print(f'Inception Score: {is_mean}+{is_std}')

# FID between training set and test set
_fid = train_test_sets_fid(50000, train_gen, test_gen)
print('FID: %f' % _fid)
