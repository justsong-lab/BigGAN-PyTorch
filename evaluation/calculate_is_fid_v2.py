import tensorflow.compat.v1 as tf
from evaluation.fid_v2 import *
from evaluation.inception_score_v2 import *
from glob import glob
from evaluation.calculate_is_fid import get_generated_samples, train_gen
import os


def load_images_from_directory(img_dir: str):
    filenames = glob(os.path.join(img_dir, '*.*'))
    images = [get_images(filename) for filename in filenames]
    images = np.transpose(images, axes=[0, 3, 1, 2])
    return images


def get_training_samples():
    HEIGHT = WIDTH = 32
    DATA_DIM = HEIGHT * WIDTH * 3
    BATCH_SIZE = 50
    n = 50000
    samples = np.zeros([int(np.ceil(float(n) / BATCH_SIZE) * BATCH_SIZE), DATA_DIM], dtype=np.uint8)
    for i in range(int(np.ceil(float(n) / BATCH_SIZE))):
        samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = ((next(train_gen)[0] + 1) / 2 * 255).astype(np.uint8)
    return samples[:n].reshape([-1, HEIGHT, WIDTH, 3]).transpose([0, 3, 1, 2])


def inception_score(images: np.array):
    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])

    logits = inception_logits(inception_images)

    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)

    print()
    print(f"IS : {IS[0]:.6f}+{IS[1]:.6f}")


def frechet_inception_distance(real_images: np.array, fake_images: np.array):
    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation,
                  activations)

    print()
    print(f"FID : {FID:.6f}")


def main():
    experiment_name = "BigGAN_C10_seed0_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema"
    # file_name = "2020-10-23_15_24_32"
    # file_name = "2020-10-22_21_18_55"
    file_name = "2020-10-26_15_35_39"
    npz_path = rf'.\samples\{experiment_name}\{file_name}.npz'
    generated_samples = get_generated_samples(npz_path)
    training_samples = get_training_samples()
    inception_score(generated_samples)
    frechet_inception_distance(training_samples, generated_samples)


if __name__ == '__main__':
    main()
